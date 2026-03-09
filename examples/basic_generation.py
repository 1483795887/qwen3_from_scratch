import json
import os

import jinja2
import torch
import torch.nn as nn
from qwen3_from_scratch.models.context import ModelContext
from tokenizers import Tokenizer

from qwen3_from_scratch.factory.config import load_from_file
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k=None,
    eos_id=None,
    tokenizer=None,
    use_cache: bool = True,
    device = "cuda",
):
    context = ModelContext()
    context.dtype = torch.bfloat16
    context.use_cache = use_cache
    is_prefill = True
    model = model.to(device)
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if is_prefill or not use_cache:
                context.cache_position = 0
                logits = model(idx, context=context)
                is_prefill = False
            else:
                context.cache_position = idx.shape[1] - 1
                logits = model(idx[:, -1:], context=context)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
        if tokenizer is not None and use_cache:
            print(
                tokenizer.decode(
                    idx.squeeze(0).tolist()[-1:], skip_special_tokens=False
                ),
                end="",
            )
    print()
    print(tokenizer.decode(idx.squeeze(0).tolist(), skip_special_tokens=False))
    return idx


def main():
    loader = ParameterLoader()
    model_path = os.environ.get("MODEL_PATH")
    loader.load(model_path)
    config = load_from_file(model_path + "/config.json")
    model = Qwen3(config=config)
    model.load_state(loader)
    unused_keys = loader.get_unused_keys()
    assert len(unused_keys) == 0, f"Unused keys: {unused_keys}"

    with open(model_path + "/tokenizer_config.json") as f:
        data = json.load(f)
        template = jinja2.Template(data["chat_template"])
        prompt = template.render(
            messages=[{"role": "user", "content": "介绍一下你自己"}]
        )
        tokenizer = Tokenizer.from_file(model_path + "/tokenizer.json")
        print(prompt)
        inputs = tokenizer.encode(prompt)
        with torch.no_grad():
            generate(
                model,
                torch.tensor([inputs.ids]),
                400,
                eos_id=config.eos_token_id,
                tokenizer=tokenizer,
                temperature=0.7,
                use_cache=True,
            )


if __name__ == "__main__":
    main()
