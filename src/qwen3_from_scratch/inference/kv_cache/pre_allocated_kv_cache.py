import torch

from .kv_cache import KVCache


class PreAllocatedKVCache(KVCache):
    def __init__(self, max_length: int, num_layers: int):
        self.max_length = max_length
        self.num_layers = num_layers
        self.k_cache: dict[int, torch.Tensor | None] = {}
        self.v_cache: dict[int, torch.Tensor | None] = {}
        self.current_length = 0

    def _initialize_cache(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
        if layer_idx not in self.k_cache or self.k_cache[layer_idx] is None:
            k_shape = list(k.shape)
            v_shape = list(v.shape)
            k_shape[2] = self.max_length
            v_shape[2] = self.max_length
            self.k_cache[layer_idx] = torch.zeros(
                k_shape, dtype=k.dtype, device=k.device
            )
            self.v_cache[layer_idx] = torch.zeros(
                v_shape, dtype=v.dtype, device=v.device
            )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        cache_position: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._initialize_cache(k, v, layer_idx)
        current_seq_len = k.shape[2]
        end_idx = cache_position + current_seq_len

        if end_idx > self.max_length:
            raise RuntimeError(
                f"Sequence length {end_idx} exceeds max_length {self.max_length}"
            )

        self.k_cache[layer_idx][:, :, cache_position:end_idx, :] = k
        self.v_cache[layer_idx][:, :, cache_position:end_idx, :] = v
        self.current_length = end_idx
        return (
            self.k_cache[layer_idx][:, :, :end_idx, :],
            self.v_cache[layer_idx][:, :, :end_idx, :],
        )

    def get(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k_cache[layer_idx][:, :, : self.current_length, :],
            self.v_cache[layer_idx][:, :, : self.current_length, :],
        )

    def reset(self):
        self.current_length = 0
