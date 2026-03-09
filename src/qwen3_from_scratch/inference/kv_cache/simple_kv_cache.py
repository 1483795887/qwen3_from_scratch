from collections import defaultdict

import torch

from .kv_cache import KVCache


class SimpleKVCache(KVCache):
    def __init__(self):
        self.k_cache: dict[int, torch.Tensor | None] = defaultdict(lambda: None)
        self.v_cache: dict[int, torch.Tensor | None] = defaultdict(lambda: None)

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        cache_position: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            self.k_cache[layer_idx] = torch.cat(
                [self.k_cache[layer_idx], k], dim=2
            )
            self.v_cache[layer_idx] = torch.cat(
                [self.v_cache[layer_idx], v], dim=2
            )
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
