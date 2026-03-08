import os
from pathlib import Path

import safetensors


def load_single_safetensors(file_path: os.PathLike):
    """加载单个safetensors文件"""
    tensors = {}
    with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
        # 获取所有键名
        keys = f.keys()
        # 逐个加载张量
        for key in keys:
            tensors[key] = f.get_tensor(key)

    return tensors


class ParameterLoader:
    def __init__(self):
        self.model_states = {}
        self.loaded_keys = set()

    def load(self, model_path: str):
        self.model_states = load_single_safetensors(
            Path(model_path) / "model.safetensors"
        )

    def get(self, key: str):
        self.loaded_keys.add(key)
        return self.model_states[key]

    def get_unused_keys(self):
        return set(self.model_states.keys()) - self.loaded_keys
