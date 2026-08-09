import os
import sys

import torch

from .kv_cache import KVCache
from ..context import get_forward_context


class PagedKVCache(KVCache):
    def __init__(self, num_blocks: int, layers: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float32,
                 block_size: int = 16, device="cuda"):
        kv_cache = torch.empty((2, layers, num_blocks, block_size, num_heads, head_dim), dtype=dtype, device=device)
        self.block_size = block_size
        self.page_size = block_size
        self.num_pages = num_blocks
        self.k_cache = kv_cache[0]  # (layers, num_pages, block_size, num_heads, head_dim)
        self.v_cache = kv_cache[1]

    @staticmethod
    def get_block_num(mem_size: int, layers: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float32,
                 block_size: int = 16, device="cuda"):
        block_size_in_bytes = layers * num_heads * head_dim * dtype.itemsize * block_size
        num_pages = mem_size // block_size_in_bytes // 2
        return num_pages

    @staticmethod
    def get_available_mem() -> int:
        """获取可用显存/内存，以字节为单位。

        有 CUDA 设备时返回可用显存，否则返回可用系统内存。
        """
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free

        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys

        # Linux/Unix
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

    def _update_var_len(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
        """
        将 k,v 写入分页缓存

        k, v: (total_seq, num_heads, head_dim) - SHD 格式
        slot_mapping: 每个 头部所在的 索引，非 block_id，而是更直接的 idx
        """
        context = get_forward_context()
        assert context.slot_mapping is not None
        slot_mapping = context.slot_mapping
        assert k.shape[0] == slot_mapping.shape[0]
        if k.is_cuda:
            from qwen3_from_scratch.kernels.triton.paged_attn import update_paged_kv_cache

            update_paged_kv_cache(
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                k,
                v,
                slot_mapping,
            )
            return
        for i in range(k.shape[0]):
            slot = slot_mapping[i]
            if slot == -1:
                continue
            block_id, slot_id = slot // self.block_size, slot % self.block_size
            self.k_cache[layer_idx, block_id, slot_id] = k[i]
            self.v_cache[layer_idx, block_id, slot_id] = v[i]
        return k,v

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, cache_pos: int = 0) -> tuple[
        torch.Tensor, torch.Tensor]:
        if len(k.shape) == 4:
            k = k.reshape(-1, *k.shape[2:])
            v = v.reshape(-1, *v.shape[2:])

        return self._update_var_len(k, v, layer_idx)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 这个用法和其他不太一样，这里返回的是所有空间，需要根据块号获得具体内容
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
