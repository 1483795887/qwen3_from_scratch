import json
import os
import time

import jinja2
import torch
from tokenizers import Tokenizer

from qwen3_from_scratch.factory.config import load_from_file
from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.inference.kv_cache.pre_allocated_kv_cache import (
    PreAllocatedKVCache,
)
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()


def warmup(model, config, device):
    context = ModelContext()
    context.use_cache = True
    context.dtype = torch.bfloat16
    context.kv_cache = PreAllocatedKVCache(128, config.num_hidden_layers)
    idx = torch.tensor([[0, 1, 2, 3]]).to(device)
    with torch.no_grad():
        for _ in range(5):
            context.cache_position = 0
            context.position_ids = None
            context.position_embeddings = None
            is_prefill = True
            cur = idx
            for _ in range(10):
                if is_prefill:
                    context.cache_position = 0
                    logits = model(cur, context=context)
                    is_prefill = False
                else:
                    context.cache_position = cur.shape[1] - 1
                    logits = model(cur[:, -1:], context=context)
                logits = logits[:, -1, :]
                probs = torch.softmax(logits / 0.7, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                cur = torch.cat((cur, nxt), dim=1)
            context.kv_cache.reset()


def benchmark(model, config, tokenizer, prompt_tokens, max_new_tokens, device, num_runs=3):
    context = ModelContext()
    context.use_cache = True
    context.dtype = torch.bfloat16
    context.kv_cache = PreAllocatedKVCache(
        max_new_tokens + len(prompt_tokens), config.num_hidden_layers
    )

    prefill_times = []
    decode_times = []
    total_decode_time_first_run = None

    for run in range(num_runs):
        if run > 0:
            context.cache_position = 0
            context.kv_cache.reset()
            context.position_ids = None
            context.position_embeddings = None
        else:
            context.cache_position = 0

        idx = torch.tensor([prompt_tokens]).to(device)
        is_prefill = True
        token_count = 0
        run_decode_times = []

        for _ in range(max_new_tokens):
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.perf_counter()

            with torch.no_grad():
                if is_prefill:
                    context.cache_position = 0
                    logits = model(idx, context=context)
                    is_prefill = False
                else:
                    context.cache_position = idx.shape[1] - 1
                    logits = model(idx[:, -1:], context=context)

            logits = logits[:, -1, :]
            probs = torch.softmax(logits / 0.7, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == config.eos_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = time.perf_counter() - start
            token_count += 1

            if token_count == 1:
                prefill_times.append(elapsed)
            else:
                run_decode_times.append(elapsed)

        decode_times.append(run_decode_times)
        print(f"  Run {run + 1}: {token_count} tokens, "
              f"prefill={prefill_times[-1]:.4f}s, "
              f"avg decode={sum(run_decode_times) / len(run_decode_times):.4f}s ({len(run_decode_times) / sum(run_decode_times):.1f} tokens/s)" if run_decode_times else "")

    avg_prefill = sum(prefill_times) / len(prefill_times)
    all_decode = [t for rt in decode_times for t in rt]
    avg_decode = sum(all_decode) / len(all_decode) if all_decode else 0
    toks_per_sec = 1.0 / avg_decode if avg_decode > 0 else 0
    total_time = avg_prefill + sum(all_decode) / num_runs if all_decode else avg_prefill

    print(f"\n  === Summary ===")
    print(f"  Time to First Token (TTFT): {avg_prefill:.4f}s")
    print(f"  Avg Decode: {avg_decode:.4f}s/token ({toks_per_sec:.2f} tokens/s)")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  Generated Tokens (per run): {sum(len(rt) for rt in decode_times) // num_runs}")

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

    loader = ParameterLoader()
    loader.load(model_path)
    config = load_from_file(model_path + "/config.json")
    config.decoder_layer.name = "my_op"
    model = Qwen3(config=config)
    model.load_state(loader)
    unused_keys = loader.get_unused_keys()
    assert len(unused_keys) == 0, f"Unused keys: {unused_keys}"
    model = model.to(device)

    print("Warming up...")
    warmup(model, config, device)

    with open(model_path + "/tokenizer_config.json") as f:
        data = json.load(f)
        template = jinja2.Template(data["chat_template"])
        prompts = {
            "short": template.render(
                messages=[{"role": "user", "content": "你好"}]
            ),
            "medium": template.render(
                messages=[{"role": "user", "content": "介绍一下人工智能的发展历程"}]
            ),
            "long": template.render(
                messages=[{"role": "user", "content": "请详细介绍人工智能的发展历程、主要应用领域、当前面临的挑战以及未来的发展趋势。请尽可能详细地阐述每个方面，包括具体的技术、案例和观点。"}]
            ),
        }

    tokenizer = Tokenizer.from_file(model_path + "/tokenizer.json")

    for name, prompt in prompts.items():
        inputs = tokenizer.encode(prompt)
        print(f"\n{'=' * 50}")
        print(f"Prompt: {name} ({len(inputs.ids)} tokens)")
        print(f"{'=' * 50}")
        results = benchmark(
            model, config, tokenizer, inputs.ids,
            max_new_tokens=200, device=device, num_runs=3
        )


if __name__ == "__main__":
    main()
