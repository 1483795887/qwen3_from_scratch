import os
from typing import Dict, Optional

from qwen3_from_scratch.factory.config import ComponentConfig, load_from_file
from qwen3_from_scratch.factory.factory import ComponentFactory
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3


class ModelLoader:
    """从磁盘加载模型权重，返回 Qwen3 实例。

    职责单一：只负责加载模型，不管 GenerationConfig、tokenizer 或 chat template。
    架构参数从 config.json 读，组件实现通过 components 参数运行时覆写。
    """

    @staticmethod
    def load(
        model_path: str,
        device: str = "cpu",
        components: Optional[Dict[str, ComponentConfig]] = None,
    ) -> Qwen3:
        config = load_from_file(os.path.join(model_path, "config.json"))
        for name, conf in (components or {}).items():
            if name not in ComponentFactory._registry:
                raise ValueError(
                    f"Unknown component field: {name}. "
                    f"Valid: {list(ComponentFactory._registry.keys())}"
                )
            if conf.name not in ComponentFactory._registry[name]:
                raise ValueError(
                    f"Unknown {name} implementation: {conf.name}. "
                    f"Registered: {list(ComponentFactory._registry[name].keys())}"
                )
            setattr(config, name, conf)
        loader = ParameterLoader()
        loader.load(model_path)
        model = Qwen3(config=config)
        model.load_state(loader)
        model.to(device)
        return model
