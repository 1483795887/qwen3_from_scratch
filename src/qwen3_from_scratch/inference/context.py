from dataclasses import dataclass, field

import torch

from .kv_cache import KVCache, SimpleKVCache


@dataclass
class ModelContext:
    dtype: torch.dtype = torch.float32
    use_cache: bool = False
    kv_cache: KVCache = field(default_factory=SimpleKVCache)
    position_ids: torch.Tensor = None
    cache_position: int = 0
    block_tables: torch.Tensor | None = None
    block_size: int = 16
    cum_seq_lens_q: torch.Tensor = field(default=torch.Tensor)
    cum_seq_lens_kv: torch.Tensor = field(default=torch.Tensor)
    slot_mapping: torch.Tensor | None = None


_CONTEXT = ModelContext()


def get_forward_context() -> ModelContext:
    return _CONTEXT


def set_forward_context(context: ModelContext) -> None:
    global _CONTEXT
    _CONTEXT = context
