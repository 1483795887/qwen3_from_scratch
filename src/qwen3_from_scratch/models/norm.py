import torch
from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig

@ComponentFactory.register("norm", "base")
class TorchRmsNorm(nn.Module):

    def __init__(self, config: ModelConfig, name: str, **kwargs):
        super().__init__()
        norm_dim = kwargs.get("dim", config.head_dim)
        self.eps: float = config.norm_params["eps"]
        self.weight = nn.Parameter(torch.ones(norm_dim))
        self.name = name

    def forward(self, x):
        input_dtype = x.dtype
        return nn.functional.rms_norm(
            x, self.weight.shape, self.weight, self.eps
        ).to(input_dtype)