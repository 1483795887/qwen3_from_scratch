import triton
import triton.language as tl
import torch
import math
from typing import Optional
from enum import Enum

class ActivationType(Enum):
  Nop = 0
  GELU = 1
  SILU = 2

ACTIVATION_NOP = tl.constexpr(ActivationType.Nop.value)
ACTIVATION_GELU = tl.constexpr(ActivationType.GELU.value)
ACTIVATION_SILU = tl.constexpr(ActivationType.SILU.value)

_TRITON_IEEE_PRECISION = False

@triton.jit
def gemm_kernel(
    a, b, c, d, 
    alpha, beta,
    M, N, H: tl.constexpr, K, groups: tl.constexpr,
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_bb, stride_bh, stride_bk, stride_bn,
    stride_cb, stride_ch, stride_cm, stride_cn,
    stride_db, stride_dh, stride_dm, stride_dn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr, is_c_needed:tl.constexpr = True
):
    bh_id = tl.program_id(2)
    n_id = tl.program_id(1)
    m_id = tl.program_id(0)

    b_id = bh_id // H
    h_id = bh_id % H
    h_id_b = h_id // groups

    a_ptr = tl.make_block_ptr(
        a + b_id * stride_ab + h_id * stride_ah,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(m_id * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    b_ptr = tl.make_block_ptr(
        b + b_id * stride_bb + h_id_b * stride_bh,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, n_id * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1,0)
    )
    c_ptr = tl.make_block_ptr(
        c + b_id * stride_cb + h_id * stride_ch,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(m_id * BLOCK_SIZE_M, n_id * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0)
    )
    d_ptr = tl.make_block_ptr(
        d + b_id * stride_db + h_id * stride_dh,
        shape=(M, N),
        offsets=(m_id * BLOCK_SIZE_M, n_id * BLOCK_SIZE_N),
        strides=(stride_dm, stride_dn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0)
    )
    dtype = d.dtype.element_ty
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K):
        block_a = tl.load(a_ptr, boundary_check=(0,1))
        block_b = tl.load(b_ptr, boundary_check=(0,1))
        if USE_FP32_ACCUM:
            acc = tl.dot(block_a, block_b, acc, input_precision="ieee")
        else:
            acc = tl.dot(block_a, block_b, acc)
        a_ptr = a_ptr.advance([0, BLOCK_SIZE_K])
        b_ptr = b_ptr.advance([BLOCK_SIZE_K, 0])
    if is_c_needed:
      block_c = tl.load(c_ptr, boundary_check=(0,1))
      acc = alpha * acc + beta * block_c
    else:
      # beta为0也不能解决nan*0=nan这个问题
      acc = alpha * acc
    # 暂且不管激活值
    tl.store(d_ptr, acc.to(dtype), boundary_check=(0, 1))


# 不需要c的时候就把d也传给c,然后 is_c_needed 传False
def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    is_c_needed: bool = True,
    alpha: float=1.0,
    beta: float=1.0,
    activation: ActivationType = ActivationType.Nop,
):
    assert len(a.shape) >= 3
    assert len(b.shape) >= 3
    assert len(d.shape) >= 3

    if len(a.shape) < 4:
        a = a.unsqueeze(1)
    if len(b.shape) < 4:
        b = b.unsqueeze(1)
    if len(c.shape) < 4:
        c = c.unsqueeze(1)
    if len(d.shape) < 4:
        d = d.unsqueeze(1)
    
    assert a.shape[0] == b.shape[0] == c.shape[0] == d.shape[0], f"Batch size mismatch: {a.shape[0]} vs {b.shape[0]} vs {c.shape[0]} vs {d.shape[0]}"
    B, Ha, M, K = a.shape
    assert K == b.shape[-2], f"Inner matrix dimension mismatch: {K} vs {b.shape[-2]}"
    _, Hb, _, N = b.shape
    assert c.shape[-2:] == (M,N) == d.shape[-2:], f"Output shape mismatch: {c.shape[-2:]} vs {(M,N)}"
    assert Ha == c.shape[1] == d.shape[1], f"Feature dimension mismatch: {c.shape[1]} vs {d.shape[1]}"
    assert Ha % Hb == 0
    
    BLOCK_SIZE_M = 128 if M > 1 else 1
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_D = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N),  B * Ha)
    gemm_kernel[grid](
        a,b,c,d,
        alpha, beta,
        M, N, Ha, K, Ha // Hb,
        a.stride(0), a.stride(1), a.stride(-2), a.stride(-1),
        b.stride(0), b.stride(1), b.stride(-2), b.stride(-1),
        c.stride(0), c.stride(1), c.stride(-2), c.stride(-1),
        d.stride(0), d.stride(1), d.stride(-2), d.stride(-1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_D,
        _TRITON_IEEE_PRECISION, is_c_needed=is_c_needed
    )

def gemm_without_c(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    alpha: float=1.0,
    activation: ActivationType = ActivationType.Nop,
):
  return gemm(a,b,d,d, is_c_needed=False, alpha=alpha,beta=0.0,activation=activation)


if __name__ == "__main__":  
    M = 1
    N = 128
    K = 1024
    B = 2
    dtype = torch.float32
    a = torch.randn(B, M, K).cuda().to(dtype)
    b = torch.randn(B, N, K).cuda().to(dtype).transpose(-2, -1)
    c = torch.randn(B, M, N).cuda().to(dtype)
    d = torch.randn(B, M, N).cuda().to(dtype)
    
    gemm(a, b, c, d)
    ref_d = torch.matmul(a,b) + c
    diff = (ref_d - d).abs()
    print(diff[1].max())
    