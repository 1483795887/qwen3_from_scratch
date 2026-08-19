import os
from pathlib import Path

import safetensors
from tqdm import tqdm


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

    def _find_safetensors(self, model_path: Path):
        files = []
        for file in model_path.rglob("model.*.safetensors"):
            files.append(file)
        if len(files) == 0:
            assert os.path.exists(model_path / "model.safetensors")
            return [model_path / "model.safetensors"]
        return sorted(files)  # 好像不排序也无所谓

    def load(self, model_path: str):
        self.model_states = {}
        all_files = self._find_safetensors(Path(model_path))
        with tqdm(total=len(all_files)) as pbar:
            for file in all_files:
                pbar.set_description(f"Loading {file}")
                self.model_states.update(load_single_safetensors(file))

    def get(self, key: str):
        self.loaded_keys.add(key)
        return self.model_states[key]

    def get_unused_keys(self):
        return set(self.model_states.keys()) - self.loaded_keys
