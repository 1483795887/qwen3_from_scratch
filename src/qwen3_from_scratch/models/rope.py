import torch
from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.inference.context import get_forward_context


@ComponentFactory.register("rope", "base")
class PythonRope(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.head_dim = config.head_dim
        self.base_freq = config.pos_embed_params["rope_theta"]
        self.max_seq_len = config.max_position_embeddings
        self.rope_type = config.pos_embed_params["rope_type"]

    def _rotate_half_neox(self, x: torch.Tensor) -> torch.Tensor:
        """NeoX风格的旋转：前后半段交叉"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _rotate_normal(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

    def _get_cos_sin(self, x: torch.Tensor):
        """直接读 ctx.cos / ctx.sin（引擎侧每步预取，与 CUDA graph capture 兼容）。

        路径固定、不再走 .cpu() 索引；要求 ctx 上 cos/sin 已就绪。
        """
        ctx = get_forward_context()
        assert ctx.cos is not None and ctx.sin is not None, (
            "context.cos / context.sin must be set before forward"
        )
        return ctx.cos, ctx.sin

    def forward(self, x: torch.Tensor):
        cos, sin = self._get_cos_sin(x)

        if x.dim() == 4:  # BHSD: (B, H, S, D)
            cos_e = cos[None, None, :, :]
            sin_e = sin[None, None, :, :]
        elif x.dim() == 3:  # SHD: (T, H, D)
            cos_e = cos[:, None, :]
            sin_e = sin[:, None, :]
        else:
            raise ValueError(f"Unexpected x.dim()={x.dim()}, expected 3 or 4")

        if self.rope_type == "neox":
            return (x * cos_e) + (self._rotate_half_neox(x) * sin_e)
        elif self.rope_type == "normal":
            return (x * cos_e) + (self._rotate_normal(x) * sin_e)
        else:
            raise ValueError(f"Unknown RoPE type: {self.rope_type}")


@ComponentFactory.register("rope", "my_op")
class MyRope(PythonRope):
    def forward(self, x: torch.Tensor):
        # neox_rope 是 triton kernel，只在 CUDA + neox 下走；其他情况落到基类
        if self.rope_type != "neox" or not x.is_cuda:
            return super().forward(x)

        cos, sin = self._get_cos_sin(x)
        cos_e = cos[None, None, :, :]
        sin_e = sin[None, None, :, :]
        from qwen3_from_scratch.kernels.triton.rope import neox_rope

        return neox_rope(x, cos_e, sin_e)
