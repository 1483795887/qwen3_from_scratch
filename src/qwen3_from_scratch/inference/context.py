from dataclasses import dataclass, field

import torch

from .kv_cache import KVCache, SimpleKVCache


@dataclass
class PositionEmbeddings:
    cos_embed: torch.Tensor
    sin_embed: torch.Tensor


@dataclass
class ModelContext:
    dtype: torch.dtype = torch.float32
    use_cache: bool = False
    kv_cache: KVCache = field(default_factory=SimpleKVCache)
    position_ids: torch.Tensor = None
    position_embeddings: PositionEmbeddings = None
    cache_position: int = 0
