import torch
from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.inference.context import get_forward_context
from qwen3_from_scratch.models.rotary import get_rope


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
        """从预计算的 RotaryEmbedding 获取 cos/sin，按 position_ids 索引。

        cos_sin_cache 结构: cat([cos(freqs), sin(freqs)])，各 head_dim//2。
        需要扩展为 head_dim 维（cat([cos, cos] 和 [sin, sin]），
        与旧 build_cos_sin_embed 的 cat([freqs, freqs]) 一致。
        """
        ctx = get_forward_context()
        assert ctx.position_ids is not None, "position_ids must be set before forward"
        rotary = get_rope(
            self.head_dim, self.head_dim, self.max_seq_len, self.base_freq
        )
        pos = ctx.position_ids.reshape(-1).to(x.device)
        cos_sin = rotary.cos_sin_cache[pos].to(x.device, x.dtype)
        half = cos_sin.shape[-1] // 2
        cos = torch.cat([cos_sin[..., :half], cos_sin[..., :half]], dim=-1)
        sin = torch.cat([cos_sin[..., half:], cos_sin[..., half:]], dim=-1)
        # cos_sin_cache 多了 unsqueeze(1) 的中间维度，squeeze 掉
        cos = cos.squeeze(1)  # (N, head_dim)
        sin = sin.squeeze(1)
        return cos, sin

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
        if self.rope_type == "normal" or not x.is_cuda:
            return super().forward(x)

        cos, sin = self._get_cos_sin(x)
        cos_e = cos[None, None, :, :]
        sin_e = sin[None, None, :, :]
        from qwen3_from_scratch.kernels.triton.rope import neox_rope
        return neox_rope(x, cos_e, sin_e)
