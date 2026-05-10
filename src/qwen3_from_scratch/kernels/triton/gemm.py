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
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    stride_db, stride_dm, stride_dn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr,
):
    b_id = tl.program_id(2)
    n_id = tl.program_id(1)
    m_id = tl.program_id(0)
    a_ptr = tl.make_block_ptr(
        a + b_id * stride_ab,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(m_id * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(0,1)
    )
    b_ptr = tl.make_block_ptr(
        b + b_id * stride_bb,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, n_id * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1,0)
    )
    c_ptr = tl.make_block_ptr(
        c + b_id * stride_cb,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(m_id * BLOCK_SIZE_M, n_id * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0)
    )
    d_ptr = tl.make_block_ptr(
        d + b_id * stride_db,
        shape=(M, N),
        offsets=(m_id * BLOCK_SIZE_M, n_id * BLOCK_SIZE_N),
        strides=(stride_dm, stride_dn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1,0)
    )
    dtype = d.dtype.element_ty
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K):
        block_a = tl.load(a_ptr)
        block_b = tl.load(b_ptr)
        if USE_FP32_ACCUM:
            acc = tl.dot(block_a, block_b, acc, input_precision="ieee")
        else:
            acc = tl.dot(block_a, block_b, acc)
        a_ptr = a_ptr.advance([0, BLOCK_SIZE_K])
        b_ptr = b_ptr.advance([BLOCK_SIZE_K, 0])
    block_c = tl.load(c_ptr)
    acc = alpha * acc + beta * block_c
    # 暂且不管激活值
    tl.store(d_ptr, acc.to(dtype), boundary_check=(0, 1))


@triton.jit
def gemm_kernel_2(
    a, b, c, d, 
    alpha, beta,
    M, N, H: tl.constexpr, K, groups: tl.constexpr,
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_bb, stride_bh, stride_bk, stride_bn,
    stride_cb, stride_ch, stride_cm, stride_cn,
    stride_db, stride_dh, stride_dm, stride_dn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr,
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
        block_a = tl.load(a_ptr)
        block_b = tl.load(b_ptr)
        if USE_FP32_ACCUM:
            acc = tl.dot(block_a, block_b, acc, input_precision="ieee")
        else:
            acc = tl.dot(block_a, block_b, acc)
        a_ptr = a_ptr.advance([0, BLOCK_SIZE_K])
        b_ptr = b_ptr.advance([BLOCK_SIZE_K, 0])
    block_c = tl.load(c_ptr)
    acc = alpha * acc + beta * block_c
    # 暂且不管激活值
    tl.store(d_ptr, acc.to(dtype), boundary_check=(0, 1))


@triton.jit
def gemm_kernel_bt(
    a, b, c, d, 
    alpha, beta,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bn, stride_bk,
    stride_cb, stride_cm, stride_cn,
    stride_db, stride_dm, stride_dn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr,
):
    b_id = tl.program_id(2)
    n_id = tl.program_id(1)
    m_id = tl.program_id(0)
    a_ptr = tl.make_tensor_descriptor(
        a + b_id * stride_ab,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
    )
    b_ptr = tl.make_tensor_descriptor(
        b + b_id * stride_bb,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
    )
    c_ptr = tl.make_tensor_descriptor(
        c + b_id * stride_cb,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
    )
    d_ptr = tl.make_tensor_descriptor(
        d + b_id * stride_db,
        shape=(M, N),
        strides=(stride_dm, stride_dn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
    )
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K):
        block_a = a_ptr.load([m_id * BLOCK_SIZE_M, k])
        block_b = b_ptr.load([n_id * BLOCK_SIZE_N, k]).T
        if USE_FP32_ACCUM:
            acc = tl.dot(block_a, block_b, acc, input_precision="ieee")
        else:
            acc = tl.dot(block_a, block_b, acc)
    
    block_c = c_ptr.load([m_id * BLOCK_SIZE_M, n_id * BLOCK_SIZE_N]).to(d.dtype.element_ty)
    acc = alpha * acc + beta * block_c
    # 暂且不管激活值
    d_ptr.store([m_id * BLOCK_SIZE_M, n_id * BLOCK_SIZE_N], acc)

def gemm_v2(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
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
    assert K == b.shape[-2], f"Inner matrix dimension mismatch: {a.shape} vs {b.shape[-2]}"
    _, Hb, _, N = b.shape
    assert c.shape[-2:] == (M,N) == d.shape[-2:], f"Output shape mismatch: {c.shape[-2:]} vs {(M,N)}"
    assert Ha == c.shape[1] == d.shape[1], f"Feature dimension mismatch: {c.shape[1]} vs {d.shape[1]}"
    assert Ha % Hb == 0
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_D = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N),  B * Ha)
    gemm_kernel_2[grid](
        a,b,c,d,
        alpha, beta,
        M, N, Ha, K, Ha // Hb,
        a.stride(0), a.stride(1), a.stride(-2), a.stride(-1),
        b.stride(0), b.stride(1), b.stride(-2), b.stride(-1),
        c.stride(0), c.stride(1), c.stride(-2), c.stride(-1),
        d.stride(0), d.stride(1), d.stride(-2), d.stride(-1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_D,
        _TRITON_IEEE_PRECISION,
    )


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    alpha: float=1.0,
    beta: float=1.0,
    activation: ActivationType = ActivationType.Nop,
) -> torch.Tensor:
    assert len(a.shape) >= 2
    assert len(b.shape) >= 2
    assert len(d.shape) >= 2
    M, K_A = a.shape[-2:]
    K_B, N = b.shape[-2:]
    assert K_A == K_B, "Inner matrix dimension mismatch: {a.shape} vs {b.shape}"
    assert d.shape[-2:] == (M,N), f"Output shape mismatch: {d.shape[-2:]} vs {(M,N)}"
    feature_shape_c = c.shape[-2:]
    broadcast_feature_shape = torch.broadcast_shapes(feature_shape_c, (M,N))
    assert broadcast_feature_shape == (M,N), f"Broadcast shape mismatch: {broadcast_feature_shape} vs {(M,N)}"

    batch_a = a.shape[:-2]
    batch_b = b.shape[:-2]
    batch_c = c.shape[:-2]
    batch_d = d.shape[:-2]

    broadcast_batch = torch.broadcast_shapes(batch_a, batch_b, batch_c, batch_d)
    assert broadcast_batch == batch_d, f"Broadcast shape mismatch: {broadcast_batch} vs {batch_d}"

    a = a.expand(*broadcast_batch, M, K_A)
    b = b.expand(*broadcast_batch, K_B, N)
    c = c.expand(*broadcast_batch, M, N)
    # d不需要扩展

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_D = 32
    
    stride_ab = 0 if len(a.shape) == 2 else a.stride(-3)
    stride_bb = 0 if len(b.shape) == 2 else b.stride(-3)
    stride_cb = 0 if len(c.shape) == 2 else c.stride(-3)
    stride_db = 0 if len(d.shape) == 2 else d.stride(-3)

    batch_size = d.numel() // ( M * N)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N),  batch_size)
    gemm_kernel[grid](
        a,b,c,d,
        alpha, beta,
        M, N, K_A,
        stride_ab, a.stride(-2), a.stride(-1),
        stride_bb, b.stride(-2), b.stride(-1),
        stride_cb, c.stride(-2), c.stride(-1),
        stride_db, d.stride(-2), d.stride(-1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_D,
        _TRITON_IEEE_PRECISION,
    )

def gemm_bt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    alpha: float=1.0,
    beta: float=1.0,
    activation: ActivationType = ActivationType.Nop,
) -> torch.Tensor:
    assert len(a.shape) >= 2
    assert len(b.shape) >= 2
    assert len(d.shape) >= 2
    M, K_A = a.shape[-2:]
    N, K_B = b.shape[-2:]
    assert K_A == K_B, "Inner matrix dimension mismatch: {a.shape} vs {b.shape}"
    assert d.shape[-2:] == (M,N), f"Output shape mismatch: {d.shape[-2:]} vs {(M,N)}"
    feature_shape_c = c.shape[-2:]
    broadcast_feature_shape = torch.broadcast_shapes(feature_shape_c, (M,N))
    assert broadcast_feature_shape == (M,N), f"Broadcast shape mismatch: {broadcast_feature_shape} vs {(M,N)}"

    batch_a = a.shape[:-2]
    batch_b = b.shape[:-2]
    batch_c = c.shape[:-2]
    batch_d = d.shape[:-2]

    broadcast_batch = torch.broadcast_shapes(batch_a, batch_b, batch_c, batch_d)
    assert broadcast_batch == batch_d, f"Broadcast shape mismatch: {broadcast_batch} vs {batch_d}"

    a = a.expand(*broadcast_batch, M, K_A)
    b = b.expand(*broadcast_batch, N, K_A)
    c = c.expand(*broadcast_batch, M, N)
    # d不需要扩展

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_D = 32
    
    stride_ab = 0 if len(a.shape) == 2 else a.stride(-3)
    stride_bb = 0 if len(b.shape) == 2 else b.stride(-3)
    stride_cb = 0 if len(c.shape) == 2 else c.stride(-3)
    stride_db = 0 if len(d.shape) == 2 else d.stride(-3)

    batch_size = d.numel() // ( M * N)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N),  batch_size)
    gemm_kernel_bt[grid](
        a,b,c,d,
        alpha, beta,
        M, N, K_A,
        stride_ab, a.stride(-2), a.stride(-1),
        stride_bb, b.stride(-2), b.stride(-1),
        stride_cb, c.stride(-2), c.stride(-1),
        stride_db, d.stride(-2), d.stride(-1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_D,
        _TRITON_IEEE_PRECISION,
    )

if __name__ == "__main__":  
    M = 12
    N = 24
    K = 32
    B = 2
    dtype = torch.float16
    a = torch.randn(B, M, K).cuda().to(dtype)
    b = torch.randn(B, N, K).cuda().to(dtype).transpose(-2, -1)
    c = torch.randn(B, M, N).cuda().to(dtype)
    d = torch.randn(B, M, N).cuda().to(dtype)
    
    gemm_v2(a, b, c, d)
    ref_d = torch.matmul(a,b) + c
    diff = (ref_d - d).abs()
    print(diff[1].max())