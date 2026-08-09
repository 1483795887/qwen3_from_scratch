from transformers import AutoTokenizer

from qwen3_from_scratch.factory import BatchConfig
from qwen3_from_scratch.inference.logger import get_logger
from qwen3_from_scratch.inference.model_loader import ModelLoader

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, config: BatchConfig):
        self.config = config
        self.tokenizer_cache = {}

    def load_tokenizer(self, model_name: str):
        if model_name in self.tokenizer_cache:
            return self.tokenizer_cache[model_name]
        if model_name not in self.config.list_model_names():
            raise KeyError(f"模型 {model_name} 不可用")
        model_info = self.config.get_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_info.path)
        self.tokenizer_cache[model_name] = tokenizer
        return self.tokenizer_cache[model_name]

    def load_model(self, model_name: str):
        logger.info(f"加载模型 {model_name} 中")
        if model_name not in self.config.list_model_names():
            raise KeyError(f"模型 {model_name} 不可用")
        model_info = self.config.get_model(model_name)
        return ModelLoader.load(
            model_info.path, model_info.device, model_info.components
        )
