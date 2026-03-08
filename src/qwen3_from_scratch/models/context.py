from dataclasses import dataclass, field

import torch


@dataclass
class PositionEmbeddings:
    cos_embed: torch.Tensor
    sin_embed: torch.Tensor


@dataclass
class KVCache:
    k_cache: torch.Tensor
    v_cache: torch.Tensor


@dataclass
class ModelContext:
    dtype: torch.dtype = torch.float32
    use_cache: bool = False
    kv_cache: dict[int, KVCache] = field(default_factory=lambda: {})
    position_ids: torch.Tensor = None
    position_embeddings: PositionEmbeddings = None
    cache_position: int = 0
