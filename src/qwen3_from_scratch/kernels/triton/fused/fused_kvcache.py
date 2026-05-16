import triton
import triton.language as tl
from typing import Optional
from qwen3_from_scratch.kernels.triton.gemm import ActivationType
import torch

def fused_kv_cache(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cos_data: torch.Tensor,
  sin_data: torch.Tensor,
  k_cache: torch.Tensor,
  v_cache: torch.Tensor,
  cache_offset: int
):
  """
  融合这些操作：
  1. QK 的 Norm
  2. QK 的 RoPE
  3. 缓存写入 KV
  q, k, v : [B, H, N, D]
  写入到 k_cache 和 v_cache 的 [:, cache_offset:cache_offset+n] 的地方
  需要注意, H 和 N 维度转置了
  """
  assert len(k.shape) == len(v.shape) == len(q.shape)
  assert k.shape == v.shape
  B, Hq, N, D = q.shape
  Hk = k.shape[1]
  assert Hq // Hk == 0
  assert k.shape == (B, Hk, N, D)

  assert k.stride(1) == D
  assert v.stride(1) == D
  assert k.stride(1) == D

  