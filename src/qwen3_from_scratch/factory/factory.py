from typing import Dict, Literal, Type

import torch.nn as nn

from .config import ComponentConfig, ModelConfig

Components = Literal[
    "self_attn", "mlp", "norm", "attn", "rope", "decoder_layer"
]


class ComponentFactory:
    _registry: Dict[Components, Type[nn.Module]] = {
        "self_attn": {},
        "mlp": {},
        "norm": {},
        "attn": {},
        "rope": {},
        "decoder_layer": {},
    }

    @classmethod
    def register(cls, component_type: Components, name: str):
        def decorator(impl_cls):
            if name in cls._registry[component_type]:
                print(f"Warning: Overwriting {component_type}/{name}")
            cls._registry[component_type][name] = impl_cls
            return impl_cls

        return decorator

    @classmethod
    def create(
        cls,
        component_type: Components,
        config: ModelConfig,
        component_impl: str = None,
        **kwargs,
    ):
        if component_type not in cls._registry:
            raise ValueError(f"Unknown component type: {component_type}")
        if not hasattr(config, component_type):
            raise ValueError(f"{component_type} not found in config")
        component_conf: ComponentConfig = getattr(config, component_type)
        component_kwargs = component_conf.kwargs.copy()
        component_kwargs.update(kwargs)
        name = (
            component_conf.name if component_impl is None else component_impl
        )
        if name not in cls._registry[component_type]:
            raise ValueError(
                f"Unknown {component_type} implementation: {name}"
            )
        return cls._registry[component_type][name](config, **component_kwargs)
