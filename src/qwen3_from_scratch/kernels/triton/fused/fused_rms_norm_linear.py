import triton
import triton.language as tl
from typing import Optional
from qwen3_from_scratch.kernels.triton.gemm import ActivationType
import torch

@triton.jit
def fused_rms_norm_linear_1d_kernel(
  x, w, gamma, output, bias,
  N, D:tl.constexpr, D1:tl.constexpr, eps,
  activation: tl.constexpr, is_bias_needed:tl.constexpr, BLOCK_SIZE_N:tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_D1: tl.constexpr
):
  """
  is_bias_needed 为 False 时， bias 的值不可信，完全可能是空指针
  D和D1当常量了，所以这里仅用于 Linear 固定权重尺寸，不要随便输入变动长度的W进来
  """
  w_dtype = w.dtype.element_ty
  a_dtype = x.dtype.element_ty
  # 第一步，计算 rms_norm 的 scale
  n_id = tl.program_id(0)
  d1_id = tl.program_id(1)

  x_ptr = tl.make_block_ptr(
    x,
    (N, D),
    (D, 1),
    (n_id * BLOCK_SIZE_N, 0),
    (BLOCK_SIZE_N, BLOCK_SIZE_D),
    (1, 0)
  )
  sum_sq = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), tl.float32)
  for _ in tl.range(tl.cdiv(D, BLOCK_SIZE_D)):
    items_x = tl.load(x_ptr, boundary_check=(0,1))
    sum_sq += items_x * items_x
    x_ptr = x_ptr.advance([0, BLOCK_SIZE_D])
  scale = tl.rsqrt(tl.sum(sum_sq, -1) / D + eps)

  acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D1), tl.float32)
  x_ptr = tl.make_block_ptr(
    x,
    (N, D),
    (D, 1),
    (n_id * BLOCK_SIZE_N, 0),
    (BLOCK_SIZE_N, BLOCK_SIZE_D),
    (1, 0)
  )
  w_ptr = tl.make_block_ptr(
    w,
    (D1, D),
    (D, 1),
    (d1_id * BLOCK_SIZE_D1, 0),
    (BLOCK_SIZE_D1, BLOCK_SIZE_D),
    (1, 0)
  )
  output_ptr = tl.make_block_ptr(
    output,
    (N, D1),
    (D1, 1),
    (n_id * BLOCK_SIZE_N, d1_id * BLOCK_SIZE_D1),
    (BLOCK_SIZE_N, BLOCK_SIZE_D1),
    (1, 0)
  )
  for off in tl.range(0, D, BLOCK_SIZE_D):
    items_x = tl.load(x_ptr, boundary_check=(0, 1)) * scale[:, None]
    offsets = off + tl.arange(0, BLOCK_SIZE_D)
    items_gamma = tl.load(gamma + offsets, offsets < D, 0.0)[None, :]
    items_x = items_x * items_gamma
    items_w = tl.load(w_ptr, boundary_check=(0, 1))
    acc = tl.dot(items_x.to(w_dtype), items_w.T, acc)
    x_ptr = x_ptr.advance([0, BLOCK_SIZE_D])
    w_ptr = w_ptr.advance([0, BLOCK_SIZE_D])

  if is_bias_needed:
    offsets = d1_id * BLOCK_SIZE_D1 + tl.arange(0, BLOCK_SIZE_D1)
    items_bias = tl.load(bias + offsets, offsets < D1, 0.0)
    acc += items_bias[None, :]
  tl.store(output_ptr, acc.to(a_dtype), boundary_check=(0,1))
  


def fused_rms_norm_linear_1d(
  x:torch.Tensor,
  w:torch.Tensor,
  gamma:torch.Tensor,
  output:torch.Tensor,
  activation: ActivationType = ActivationType.Nop,
  eps:float=1e-5,
  bias:Optional[torch.Tensor] = None
):
  """
  x: (N)xD, 不少于二维，使用会把其他维度都打包到一维中
  w: D1xD, 只支持二维, 原始权重，不要转置
  gamma: D1
  output: (N)xD1
  activation: 激活函数, 默认为Nop不做
  eps: float
  bias: D1, 可选
  """
  assert len(x.shape) >= 2
  assert len(w.shape) == 2
  D = x.shape[-1]
  assert D == w.shape[1]
  D1 = w.shape[0]
  assert D1 == output.shape[-1]
  assert gamma.shape == (D,)
  if bias is not None:
    assert bias.shape == (D1,)

  assert len(output.shape) == len(x.shape)
  N = x.numel() // D
  assert output.numel() // D1 == N
  assert x.stride(-2) == D, "x msut be contiguous at lease last dim"
  assert w.is_contiguous(), "w must be contiguous"
  assert output.stride(-2) == D1, "output msut be contiguous at lease last dim"

  BLOCK_SIZE_N = 64
  BLOCK_SIZE_D1 = 128
  BLOCK_SIZE_D = 32
  grid = [triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(D1, BLOCK_SIZE_D1)]
  fused_rms_norm_linear_1d_kernel[grid](x, w, gamma, output, bias, N, D, D1,eps, activation, bias is not None, BLOCK_SIZE_N, BLOCK_SIZE_D, BLOCK_SIZE_D1)