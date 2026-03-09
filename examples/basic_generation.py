import json
import os

import jinja2
import torch
from tokenizers import Tokenizer

from qwen3_from_scratch.factory.config import load_from_file
from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.inference.generate import generate
from qwen3_from_scratch.inference.kv_cache.pre_allocated_kv_cache import (
    PreAllocatedKVCache,
)
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()

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
        context = ModelContext()
        context.use_cache = True
        context.dtype = torch.bfloat16
        context.kv_cache = PreAllocatedKVCache(
            1024, config.num_hidden_layers
        )
        with torch.no_grad():
            result = generate(
                model,
                torch.tensor([inputs.ids]),
                400,
                context=context,
                eos_id=config.eos_token_id,
                tokenizer=tokenizer,
                temperature=0.7,
                stream=True,
            )
            for token in result:
                print(token, end="")


if __name__ == "__main__":
    main()
