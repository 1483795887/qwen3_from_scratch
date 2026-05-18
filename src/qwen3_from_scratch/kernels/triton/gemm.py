import triton
import triton.language as tl
import torch
import math
from typing import Optional
from enum import Enum
import triton

class ActivationType(Enum):
  Nop = 0
  GELU = 1
  SILU = 2

ACTIVATION_NOP = tl.constexpr(ActivationType.Nop.value)
ACTIVATION_GELU = tl.constexpr(ActivationType.GELU.value)
ACTIVATION_SILU = tl.constexpr(ActivationType.SILU.value)

ACTIVATION_MAP = {
  ActivationType.Nop: ACTIVATION_NOP,
  ActivationType.GELU: ACTIVATION_GELU,
  ActivationType.SILU: ACTIVATION_SILU
}

_TRITON_IEEE_PRECISION = False

@triton.jit
def activation_fc(items, activation:tl.constexpr):
  if activation == ACTIVATION_NOP:
    return items
  return items

@triton.jit
def gemm_kernel_core(
  a_ptr,
  b_ptr,
  acc,
  BLOCK_SIZE_K, K
):
  w_dtype = b_ptr.dtype.element_ty
  for k in tl.range(0, K, BLOCK_SIZE_K):
    block_a = tl.load(a_ptr, boundary_check=(0, 1))
    block_b = tl.load(b_ptr, boundary_check=(0,1))
    acc = tl.dot(block_a.to(w_dtype), block_b, acc)
    a_ptr = a_ptr.advance([0, BLOCK_SIZE_K])
    b_ptr = b_ptr.advance([BLOCK_SIZE_K, 0])
  return acc


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
    acc = gemm_kernel_core(a_ptr, b_ptr, acc, BLOCK_SIZE_K, K)
    if is_c_needed:
      block_c = tl.load(c_ptr, boundary_check=(0,1))
      acc = alpha * acc + beta * block_c
    else:
      # beta为0也不能解决nan*0=nan这个问题
      acc = alpha * acc
    if activation_fc == ACTIVATION_SILU:
      acc = acc * tl.sigmod(acc)
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
    BLOCK_SIZE_D = 32 if b.dtype == torch.float32 else 64
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N),  B * Ha)
    gemm_kernel[grid](
        a,b,c,d,
        alpha, beta,
        M, N, Ha, K, Ha // Hb,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        d.stride(0), d.stride(1), d.stride(2), d.stride(3),
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


def linear(
  x: torch.Tensor,
  w: torch.Tensor,
  output: torch.Tensor,
  bias: Optional[torch.Tensor] = None,
  activation_fc: ActivationType = ActivationType.Nop
):
  assert len(w.shape) == 2
  D1, D = w.shape
  assert D == x.shape[-1], f"x feature dim mismatch: expected {D}, got {x.shape[-1]}"
  # 普通的Linear不管分组，只要求dim-1连续，所以直接把所有全部打包到M维度
  assert x.stride(-1) == 1
  assert output.shape[-1] == D1
  M = x.numel() // D
  assert output.numel() // D1 == M
  x = x.view(1, 1, M, D)
  w = w.view(1, 1, D1, D)
  output = output.view(1, 1, M, D1)
  if bias is not None:
    assert bias.shape == (D1,)
    bias = bias.view(1, 1, 1, D1).expand(1, 1, M, D1)
    gemm(x, w.transpose(-1, -2), bias, output, True, activation=activation_fc)
  else:
    gemm(x, w.transpose(-1, -2), output, output, False, beta=0.0, activation=activation_fc)
  


@triton.jit
def grouped_gemm_kernel(
  a_ptrs, b_ptrs, c_ptrs, d_ptrs, alpha_ptrs, beta_ptrs,
  shapes, n_groups, lds,
  activation: int,
  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
  ab_dtype: tl.constexpr, cd_dtype: tl.constexpr
):
  num_ctas = tl.num_programs(0)
  tile_id = tl.program_id(0)
  start_tile = 0
  for g in tl.range(n_groups):
    m = tl.load(shapes + 3 * g)
    n = tl.load(shapes + 3 * g + 1)
    k = tl.load(shapes + 3 * g + 2)
    a_ptr = tl.load(a_ptrs + g).to(tl.pointer_type(ab_dtype))
    b_ptr = tl.load(b_ptrs + g).to(tl.pointer_type(ab_dtype))
    c_ptr = tl.load(c_ptrs + g).to(tl.pointer_type(cd_dtype))
    d_ptr = tl.load(d_ptrs + g).to(tl.pointer_type(cd_dtype))
    alpha = tl.load(alpha_ptrs + g)
    beta = tl.load(beta_ptrs + g)
    ldam = tl.load(lds + 8 * g)
    ldak = tl.load(lds + 8 * g + 1)
    ldbk = tl.load(lds + 8 * g + 2)
    ldbn = tl.load(lds + 8 * g + 3)
    ldcm = tl.load(lds + 8 * g + 4)
    ldcn = tl.load(lds + 8 * g + 5)
    lddm = tl.load(lds + 8 * g + 6)
    lddn = tl.load(lds + 8 * g + 7)

    num_tiles_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(n, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n
    while start_tile <= tile_id and tile_id < start_tile + num_tiles:
      tile_in_group = tile_id - start_tile
      tile_id_m = tile_in_group // num_tiles_n
      tile_id_n = tile_in_group % num_tiles_n

      offset_m = tile_id_m * BLOCK_SIZE_M
      offset_n = tile_id_n * BLOCK_SIZE_N

      desc_a = tl.make_block_ptr(a_ptr, (m,k), (ldam, ldak), [offset_m, 0], (BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1,0))
      desc_b = tl.make_block_ptr(b_ptr, (k,n), (ldbk, ldbn), [0, offset_n], (BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1,0))
      desc_c = tl.make_block_ptr(c_ptr, (m,n), (ldcm, ldcn), [offset_m, offset_n], (BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1,0))
      desc_d = tl.make_block_ptr(d_ptr, (m,n), (lddm, lddn), [offset_m, offset_n], (BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1,0))
      acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
      block_c = tl.load(desc_c, boundary_check=(0,1))
      acc = gemm_kernel_core(desc_a, desc_b, acc, BLOCK_SIZE_K, k)

      acc = acc * alpha + beta * block_c
      # acc = activation_fc(acc, activation)
      tl.store(desc_d, acc.to(cd_dtype), boundary_check=(0, 1))

      tile_id += num_ctas

    start_tile += num_tiles

def grouped_gemm(
  a_tensors: list[torch.Tensor],
  b_tensors: list[torch.Tensor],
  c_tensors: list[torch.Tensor],
  d_tensors: list[torch.Tensor],
  alpha: list[float],
  beta: list[float],
  activation: ActivationType = ActivationType.Nop
):
  assert len(a_tensors) == len(b_tensors) == len(c_tensors) == len(alpha) == len(beta), f"grouped size should be same: len(a) {len(a_tensors)} vs len(b) {len(b_tensors)} vs len(c) {len(c_tensors)} vs len(alpha) {len(alpha)} vs len(beta) {len(beta)}"
  assert len(a_tensors) > 0
  device = a_tensors[0].device
  ab_dtype = tl.float32 if  a_tensors[0].dtype == torch.float32 else tl.float16
  cd_dtype = tl.float32 if d_tensors[0].dtype == torch.float32 else tl.float16
  NUM_SMS = 84
  n_groups = len(a_tensors)
  shapes = []
  lds = []
  a_addrs =[]
  b_addrs = []
  c_addrs = []
  d_addrs = []
  for i in range(n_groups):
    a = a_tensors[i]
    b = b_tensors[i]
    c = c_tensors[i]
    d = d_tensors[i]
    a_addrs.append(a.data_ptr())
    b_addrs.append(b.data_ptr())
    c_addrs.append(c.data_ptr())
    d_addrs.append(d.data_ptr())

    m,k = a.shape
    assert b.shape[0] == k
    _, n = b.shape
    assert c.shape == (m,n) == d.shape
    shapes.append([m,n,k])
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    stride_dm, stride_dn = d.stride()

    lds.append([stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_dm, stride_dn])
  BLOCK_SIZE_M = 128
  BLOCK_SIZE_N = 128
  BLOCK_SIZE_K = 32
  grouped_gemm_kernel[[NUM_SMS]](
    torch.tensor(a_addrs, device=device),
    torch.tensor(b_addrs, device=device),
    torch.tensor(c_addrs, device=device),
    torch.tensor(d_addrs, device=device),
    torch.tensor(alpha, dtype=torch.float32, device=device),
    torch.tensor(beta, dtype=torch.float32, device=device),
    torch.tensor(shapes, dtype=torch.int32, device=device),
    n_groups = n_groups,
    lds = torch.tensor(lds, device=device),
    activation=ACTIVATION_MAP[activation],
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
    ab_dtype=ab_dtype,
    cd_dtype=cd_dtype
  )
    



if __name__ == "__main__":  
    M = 128
    N = 128
    K = 1024
    groups = 1
    device= 'cuda'
    a_tensors = [torch.rand(M, K, dtype=torch.float32, device=device) * 10 for _ in range(groups)]
    b_tensors = [torch.rand(K, N, dtype=torch.float32, device=device) * 10 for _ in range(groups)]
    d_tensors = [torch.empty(M,N,device=device) for _ in range(groups) ]
    alphas= [1.0] * groups
    betas = [0.0] * groups



    grouped_gemm(a_tensors, b_tensors , d_tensors, d_tensors, alphas, betas)
    for g in range(groups):
      ref_d = a_tensors[g] @ b_tensors[g]

      diff = (ref_d - d_tensors[g]).diff()
      print(diff.max())

    M = 128
    N = 128
    K = 1024
    B = 2
    dtype = torch.float32
    a = torch.randn(B, M, K).cuda().to(dtype) * 10
    b = torch.randn(B, N, K).cuda().to(dtype).transpose(-2, -1) * 10
    c = torch.randn(B, M, N).cuda().to(dtype) * 10
    d = torch.randn(B, M, N).cuda().to(dtype)
    
    gemm(a, b, c, d)
    ref_d = torch.matmul(a,b) + c
    diff = (ref_d - d).abs()
    print(diff.max())
    