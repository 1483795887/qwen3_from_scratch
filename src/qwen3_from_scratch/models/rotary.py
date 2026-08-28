"""预计算 Rotary Embedding 模块。

通过 get_rope 工厂函数获取全局共享的 RotaryEmbedding 实例，
cos/sin 一次性预计算并注册为 buffer，后续只做索引。
"""

from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对 x 应用旋转位置编码（NeoX 风格）。"""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = positions.cpu()
        cos_sin = self.cos_sin_cache[pos].to(query.device, query.dtype)
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
) -> RotaryEmbedding:
    """获取全局共享的 RotaryEmbedding 实例。

    相同参数只创建一次，后续调用直接返回缓存实例。
    """
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def build_cos_sin_table(
    head_dim: int,
    max_pos: int,
    base: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一次性构建最大长度的 RoPE cos/sin 表，形状 (max_pos, head_dim)。

    引擎侧在 init/warmup 阶段调用一次；运行时每步按 position_ids 切片即可，
    避免每层每步的 .cpu() 同步（D2H 同步会在 CUDA graph capture 中失效）。
    cos/sin 在 head_dim 维通过 cat([x, x]) 扩到全长，与 _get_cos_sin
    旧路径的 cat([cos, cos], -1) / cat([sin, sin], -1) 一致。
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_half = freqs.cos()
    sin_half = freqs.sin()
    # 把 head_dim//2 扩到 head_dim（前半段拼后半段，旧 _get_cos_sin 同样做法）
    cos = torch.cat([cos_half, cos_half], dim=-1).to(device=device, dtype=dtype)
    sin = torch.cat([sin_half, sin_half], dim=-1).to(device=device, dtype=dtype)
    return cos, sin
