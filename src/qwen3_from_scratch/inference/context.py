from dataclasses import dataclass, field

import torch

from .kv_cache import KVCache, SimpleKVCache


@dataclass
class ModelContext:
    dtype: torch.dtype = torch.float32
    use_cache: bool = False
    use_decode_graph: bool = False
    kv_cache: KVCache = field(default_factory=SimpleKVCache)

    # ====== Batch 模式使用 Packed 不使用
    cache_position: int = 0

    position_ids: torch.Tensor = None
    block_tables: torch.Tensor | None = None
    block_size: int = 16
    # ====== 预填充和混合使用 ======
    cum_seq_lens_q: torch.Tensor = field(default=torch.Tensor)
    cum_seq_lens_kv: torch.Tensor = field(default=torch.Tensor)

    # ====== 纯解码使用 ======
    context_lens: torch.Tensor = field(default=torch.Tensor)

    slot_mapping: torch.Tensor | None = None
    # 引擎侧预取（每步一次，避免每层每步 D2H 同步；None 表示未预取，走旧路径）
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None
    # ====== 预填充和混合使用 ======
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0


_CONTEXT = ModelContext()


def get_forward_context() -> ModelContext:
    return _CONTEXT


def set_forward_context(context: ModelContext) -> None:
    global _CONTEXT
    _CONTEXT = context
