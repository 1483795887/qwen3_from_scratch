import torch

# 核心：给 rms_norm_forward 函数写类型提示
def rms_norm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor: ...
