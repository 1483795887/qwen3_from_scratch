import json
import os
import time

import jinja2
import torch
from tokenizers import Tokenizer

from qwen3_from_scratch.factory.config import load_from_file
from qwen3_from_scratch.inference.engine import InferenceEngine
from qwen3_from_scratch.inference.sampler import GreedySampler
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()


def warmup(engine: InferenceEngine, num_steps: int = 10):
    """用短输入做若干轮 warmup（贪婪解码，避免随机性干扰）。"""
    greedy = GreedySampler()
    engine.sampler = greedy
    idx = torch.tensor([[0, 1, 2, 3]]).to(engine.device)
    with torch.no_grad():
        for _ in range(5):
            first = engine.prefill(idx)
            cur = first
            for _ in range(num_steps - 1):
                cur = engine.step(cur)


def benchmark(
    engine: InferenceEngine,
    config,
    prompt_tokens,
    max_new_tokens,
    device,
    num_runs=3,
):
    """逐 token 计时，记录 prefill 和 decode 时间。"""
    prefill_times = []
    decode_times = []
    total_decode_time_first_run = None

    for run in range(num_runs):
        idx = torch.tensor([prompt_tokens]).to(device)
        token_count = 0
        run_decode_times = []

        for _ in range(max_new_tokens):
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.perf_counter()

            with torch.no_grad():
                if token_count == 0:
                    first = engine.prefill(idx)
                    nxt = first
                else:
                    nxt = engine.step(prev)

            if nxt == config.eos_token_id:
                break

            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.perf_counter() - start
            token_count += 1
            prev = nxt

            if token_count == 1:
                prefill_times.append(elapsed)
            else:
                run_decode_times.append(elapsed)

        decode_times.append(run_decode_times)
        print(
            f"  Run {run + 1}: {token_count} tokens, "
            f"prefill={prefill_times[-1]:.4f}s, "
            f"avg decode={sum(run_decode_times) / len(run_decode_times):.4f}s "
            f"({len(run_decode_times) / sum(run_decode_times):.1f} tokens/s)"
            if run_decode_times
            else f"  Run {run + 1}: {token_count} tokens, prefill={prefill_times[-1]:.4f}s"
        )

    avg_prefill = sum(prefill_times) / len(prefill_times)
    all_decode = [t for rt in decode_times for t in rt]
    avg_decode = sum(all_decode) / len(all_decode) if all_decode else 0
    toks_per_sec = 1.0 / avg_decode if avg_decode > 0 else 0
    total_time = (
        avg_prefill + sum(all_decode) / num_runs if all_decode else avg_prefill
    )

    print(f"\n  === Summary ===")
    print(f"  Time to First Token (TTFT): {avg_prefill:.4f}s")
    print(f"  Avg Decode: {avg_decode:.4f}s/token ({toks_per_sec:.2f} tokens/s)")
    print(f"  Total Time: {total_time:.4f}s")
    print(
        f"  Generated Tokens (per run): {sum(len(rt) for rt in decode_times) // num_runs}"
    )

    return {
        "ttft": avg_prefill,
        "avg_decode_time": avg_decode,
        "decode_tokens_per_sec": toks_per_sec,
        "total_time": total_time,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.environ.get("MODEL_PATH")
    print(f"Device: {device}")
    print(f"Model: {model_path}")

    config = load_from_file(model_path + "/config.json")
    config.decoder_layer.name = "my_op"
    engine = InferenceEngine.from_path(
        model_path, device=device, max_len=512
    )
    engine.model.config = config

    print("Warming up...")
    warmup(engine)

    with open(model_path + "/tokenizer_config.json") as f:
        data = json.load(f)
        template = jinja2.Template(data["chat_template"])
        prompts = {
            "short": template.render(
                messages=[{"role": "user", "content": "你好"}]
            ),
            "medium": template.render(
                messages=[
                    {"role": "user", "content": "介绍一下人工智能的发展历程"}
                ]
            ),
            "long": template.render(
                messages=[
                    {
                        "role": "user",
                        "content": "请详细介绍人工智能的发展历程、主要应用领域、当前面临的挑战以及未来的发展趋势。请尽可能详细地阐述每个方面，包括具体的技术、案例和观点。",
                    }
                ]
            ),
        }

    tokenizer = engine.tokenizer

    for name, prompt in prompts.items():
        inputs = tokenizer.encode(prompt)
        print(f"\n{'=' * 50}")
        print(f"Prompt: {name} ({len(inputs.ids)} tokens)")
        print(f"{'=' * 50}")
        results = benchmark(
            engine,
            config,
            inputs.ids,
            max_new_tokens=200,
            device=device,
            num_runs=3,
        )


if __name__ == "__main__":
    main()
