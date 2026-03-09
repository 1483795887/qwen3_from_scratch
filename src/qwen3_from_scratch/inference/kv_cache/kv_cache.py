from abc import ABC, abstractmethod

import torch


class KVCache(ABC):
    @abstractmethod
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        cache_pos: int = 0
    )->tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass
