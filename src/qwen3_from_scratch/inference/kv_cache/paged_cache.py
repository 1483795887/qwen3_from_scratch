import torch

from .kv_cache import KVCache


class PagedKVCache(KVCache):
    def __init__(self, mem_size: int, layers: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float32,
                 block_size: int = 16, device="cuda"):
        block_size_in_bytes = layers * num_heads * head_dim * dtype.itemsize * block_size
        num_pages = mem_size // block_size_in_bytes // 2
        kv_cache = torch.empty((2, layers, num_pages, block_size, num_heads, head_dim), dtype=dtype, device=device)
        self.block_size = block_size
        self.page_size = block_size
        self.num_pages = num_pages
        self.k_cache = kv_cache[0]  # (layers, num_pages, block_size, num_heads, head_dim)
        self.v_cache = kv_cache[1]
        self.block_tables: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_pos: int = 0) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        将 k, v 写入分页缓存。

        k, v: (batch, seq, num_heads, head_dim) — BSHD 格式
        cache_pos: 本次写入在缓存中的起始位置（绝对位置）
        """
        batch_size, seq_len, num_heads, head_dim = k.shape
        for b in range(batch_size):
            pos = cache_pos
            remaining = seq_len
            while remaining > 0:
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                block_id = self.block_tables[b][block_idx].item()
                size = min(self.block_size - offset, remaining)
                src_start = seq_len - remaining
                self.k_cache[layer_idx, block_id, offset:offset + size] = k[b, src_start:src_start + size]
                self.v_cache[layer_idx, block_id, offset:offset + size] = v[b, src_start:src_start + size]
                pos += size
                remaining -= size
        return k, v

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 这个用法和其他不太一样，这里返回的是所有空间，需要根据块号获得具体内容
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
