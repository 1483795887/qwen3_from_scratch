import os

from qwen3_from_scratch.factory.config import ModelConfig, load_from_file
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3


class ModelLoader:
    """从磁盘加载模型权重，返回 Qwen3 实例。

    职责单一：只负责加载模型，不管 GenerationConfig、tokenizer 或 chat template。
    """

    @staticmethod
    def load(
        model_path: str,
        device: str = "cpu",
        config: ModelConfig = None,
    ) -> Qwen3:
        if config is None:
            config = load_from_file(os.path.join(model_path, "config.json"))
        loader = ParameterLoader()
        loader.load(model_path)
        model = Qwen3(config=config)
        model.load_state(loader)
        model.to(device)
        return model
