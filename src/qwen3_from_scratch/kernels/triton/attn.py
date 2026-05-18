import triton
import triton.language as tl
import torch
import os
import math
from qwen3_from_scratch.kernels.triton.gemm import gemm, gemm_without_c

_TRITON_IEEE_PRECISION = os.environ.get("TRITON_IEEE_PRECISION", "0") == "1"
@triton.jit
def softmax(
    A, head_stride, row_stride, col_stride, M, N, is_causal, BLOCK_SIZE: tl.constexpr
):
    row_start = tl.program_id(0)
    pid_h = tl.program_id(1)
    A_ptr = A + pid_h * head_stride
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, M, row_step):
        row_start_ptr = A_ptr + row_idx * row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptr = row_start_ptr + col_offsets * col_stride
        mask = col_offsets < N
        row = tl.load(input_ptr, mask=mask & ((not is_causal) | (col_offsets <= row_idx)), other=-float("inf"))
        max_val = tl.max(row)
        row = row - max_val
        row = tl.math.exp2(row)
        sum_val = tl.sum(row)
        row = row / sum_val
        tl.store(input_ptr, row, mask=mask)

def sdpa_shape_assert(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    assert len(Q.shape) == 4
    assert len(K.shape) == 4
    assert len(V.shape) == 4
    assert K.shape == V.shape
    assert K.shape[-1] == Q.shape[-1]

    B, Hq, M, D = Q.shape
    _, Hk, N, _ = K.shape
    assert Hq % Hk == 0
    assert Q.stride(1) == D, f"Q stride mismatch: {Q.stride(1)} vs {D}"
    assert K.stride(1) == D, f"K stride mismatch: {K.stride(1)} vs {D}"
    assert V.stride(1) == D, f"V stride mismatch: {V.stride(1)} vs {D}"


def scaled_dot_production(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True
):
    sdpa_shape_assert(Q, K, V)

    B, Hq, M, D = Q.shape
    _, Hk, N, _ = K.shape
    assert Hq % Hk == 0
    groups = Hq // Hk

    scale = 1.0 / (D**0.5)* math.log2(math.e)

    attn = torch.empty((B, Hq, M, N), dtype=Q.dtype, device=Q.device)
    gemm_without_c(Q, K.transpose(-1,-2), attn, alpha=scale)

    BLOCK_SIZE = triton.next_power_of_2(N)
    ROW_PER_BLOCK = 32
    grid = (triton.cdiv(M, ROW_PER_BLOCK), B * Hq)
    softmax[grid](
        attn, attn.stride(1), attn.stride(2), attn.stride(3), M, N, is_causal and M > 1, BLOCK_SIZE
    )
    output = torch.empty_like(Q)
    gemm_without_c(attn, V, output,alpha= 1.0)
    return output

@triton.jit
def fused_attention_intr(
    data_q,
    K_ptr, V_ptr,
    result_o, max_val, dominator,
    N_QUERY, N_KEY, STRIDE_N_KV: tl.constexpr,
    start_m,
    scale,
    mask_d,
    offsets_d, offsets_n,
    offsets_m,
    STAGE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr,
    dtype: tl.constexpr
):
    if STAGE == 1:
        lo, hi = 0, min(start_m * BLOCK_SIZE_M, N_KEY)
    elif STAGE == 2:
        lo, hi = min(start_m * BLOCK_SIZE_M, N_KEY), min((start_m + 1) * BLOCK_SIZE_M, N_KEY)
    else:
        lo, hi = 0, N_KEY
    for k in tl.range(lo, hi, BLOCK_SIZE_N, warp_specialize=True):
        k = tl.multiple_of(k, BLOCK_SIZE_N)
        kv_mask = mask_d & (offsets_n[:, None] < N_KEY)
        data_k = tl.load(
            K_ptr + offsets_n[:, None] * STRIDE_N_KV + offsets_d[None, :],
            mask=kv_mask,
            other=0.0,
        )
        data_v = tl.load(
            V_ptr + offsets_n[:, None] * STRIDE_N_KV + offsets_d[None, :],
            mask=kv_mask,
            other=0.0,
        )
        if USE_FP32_ACCUM:
            attn = tl.dot(data_q, data_k.T, input_precision="ieee") * scale
        else:
            attn = tl.dot(data_q, data_k.T) * scale
        attn = tl.where(offsets_n[None, :] < N_KEY, attn, -float("inf"))
        if STAGE == 2:
            attn = tl.where(offsets_m[:, None] >= offsets_n[None, :], attn, -float("inf"))
        tmp_max = tl.max(attn, axis=-1, keep_dims=True)
        new_max_val = tl.maximum(max_val, tmp_max)
        attn = attn - new_max_val
        exp_attn = tl.math.exp2(attn)

        scale_factor = tl.math.exp2(max_val - new_max_val)
        dominator = dominator * scale_factor + tl.sum(exp_attn, axis=-1, keep_dims=True)
        max_val = new_max_val
        exp_attn = exp_attn.to(dtype)
        if USE_FP32_ACCUM:
            result_o = result_o * scale_factor + tl.dot(exp_attn, data_v, input_precision="ieee")
        else:
            result_o = result_o * scale_factor + tl.dot(exp_attn, data_v)

        offsets_n += BLOCK_SIZE_N
    return result_o, max_val, dominator, offsets_n

@triton.jit
def fused_attention(
    Q, K, V, output,
    N_QUERY, N_KEY, HEAD_QUERY:tl.constexpr, HEAD_DIM: tl.constexpr,
    scale,
    is_causal: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr,
    dtype: tl.constexpr,
):
    b_id = tl.program_id(2)
    h_id = tl.program_id(1)
    h_id_kv = h_id // groups
    n_q_id = tl.program_id(0)
    result_o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_DIM), dtype=tl.float32)
    offsets_qm = n_q_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_qd = tl.arange(0, BLOCK_SIZE_DIM)

    STRIDE_N_QO = HEAD_DIM * HEAD_QUERY
    STRIDE_N_KV = STRIDE_N_QO // groups
    STRIDE_B_QO = N_QUERY * STRIDE_N_QO
    STRIDE_B_KV = N_KEY * STRIDE_N_KV

    Q_ptr = Q + b_id * STRIDE_B_QO + h_id * HEAD_DIM
    K_ptr = K + b_id * STRIDE_B_KV + h_id_kv * HEAD_DIM
    V_ptr = V + b_id * STRIDE_B_KV + h_id_kv * HEAD_DIM
    O_ptr = output + b_id * STRIDE_B_QO + h_id * HEAD_DIM

    mask_m = offsets_qm[:, None] < N_QUERY
    mask_d = offsets_qd < HEAD_DIM

    data_q = tl.load(
        Q_ptr + offsets_qm[:, None] * STRIDE_N_QO + offsets_qd[None, :],
        mask=mask_m & mask_d,
        other=0.0,
    )
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    max_val = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32) - float("inf")
    dominator = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    if is_causal:
        result_o, max_val, dominator, offsets_n = fused_attention_intr(
            data_q, 
            K_ptr, V_ptr,
            result_o, max_val, dominator, 
            N_QUERY, N_KEY, STRIDE_N_KV,
            n_q_id,
            scale,
            mask_d,
            offsets_qd, offsets_n,
            offsets_qm,
            1,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            USE_FP32_ACCUM,
            dtype,
        )
        result_o, max_val, dominator, offsets_n = fused_attention_intr(
            data_q, 
            K_ptr, V_ptr,
            result_o, max_val, dominator,
            N_QUERY, N_KEY, STRIDE_N_KV,
            n_q_id,
            scale,
            mask_d,
            offsets_qd, offsets_n,
            offsets_qm,
            2,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            USE_FP32_ACCUM,
            dtype,
        )
    else:
        result_o, max_val, dominator, offsets_n = fused_attention_intr(
            data_q, 
            K_ptr, V_ptr,
            result_o, max_val, dominator,
            N_QUERY, N_KEY, STRIDE_N_KV,
            n_q_id,
            scale,
            mask_d,
            offsets_qd, offsets_n,
            offsets_qm,
            3,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            USE_FP32_ACCUM,
            dtype,
        )
    result_o = (result_o / dominator).to(dtype)

    tl.store(
        O_ptr + offsets_qm[:, None] * STRIDE_N_QO + offsets_qd[None, :],
        result_o.to(dtype),
        mask=mask_d & mask_m,
    )

# Q, K, V, output are tensors on the GPU


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal:bool=True):
    sdpa_shape_assert(Q, K, V)
    B, Hq, M, D = Q.shape
    _, Hk, N, _ = K.shape
    assert Hq % Hk == 0
    groups = Hq // Hk

    output = torch.empty_like(Q)

    scale = 1.0 / (D**0.5) * math.log2(math.e)
    BLOCK_SIZE_M = 32 if M > 1 else 1
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = triton.next_power_of_2(max(D, 16))
    grid = (triton.cdiv(M, BLOCK_SIZE_M), Hq , B)
    dtype = tl.float32 if Q.dtype == torch.float32 else (tl.float16 if Q.dtype == torch.float16 else tl.bfloat16)
    fused_attention[grid](
        Q, K, V, output,
        M, N, Hq, D,
        scale,
        is_causal and M > 1,
        groups,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        _TRITON_IEEE_PRECISION,
        dtype,
    )
    return output

def cpu_forward(q, k, v, head_dim, is_causal: bool = True):
    batch_size, head_q, seq_len_q, head_dim = q.shape
    head_kv = k.shape[1]
    
    # 计算缩放因子
    scale = head_dim ** -0.5
    
    # GQA: 将q的head分组，每组对应一个kv head
    assert head_q % head_kv == 0, f"head_q ({head_q}) must be divisible by head_kv ({head_kv})"
    n_groups = head_q // head_kv
    
    # 重塑q为 [batch, head_kv, n_groups, seq_len_q, head_dim]
    q_reshaped = q.reshape(batch_size, head_kv, n_groups, seq_len_q, head_dim)
    
    # 扩展k和v的维度以匹配q的分组 [batch, head_kv, 1, seq_len_k, head_dim]
    k_expanded = k.unsqueeze(2)
    v_expanded = v.unsqueeze(2)
    
    # 计算注意力分数: [batch, head_kv, n_groups, seq_len_q, seq_len_k]
    scores = torch.matmul(q_reshaped, k_expanded.transpose(-2, -1)) * scale
    
    # 应用因果掩码
    seq_len_k = k.shape[2]
    if seq_len_q > 1 and is_causal:
        # 创建因果掩码
        causal_mask = torch.tril(
            torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 计算输出: [batch, head_kv, n_groups, seq_len_q, head_dim]
    out = torch.matmul(attn_weights, v_expanded)
    
    # 重塑回原始形状 [batch, head_q, seq_len_q, head_dim]
    out = out.reshape(batch_size, head_q, seq_len_q, head_dim)
    
    return out


def main():
  print()
  for dtype in [
    torch.float32, 
    # torch.float16, 
    # torch.bfloat16
  ]:
      M = 1
      B = 2
      N = 1023
      D = 128
      H = 16
      groups = 2
      q = torch.rand(B, M, H, D, dtype=dtype, device='cuda').transpose(1,2)
      k = torch.rand(B, N, H//groups, D, dtype=dtype, device='cuda').transpose(1,2)
      v = torch.rand(B, N, H//groups, D, dtype=dtype, device='cuda').transpose(1,2)
      ref_o = cpu_forward(q, k, v, 4)
      o = flash_attention(q, k, v)
      sdpa_o = scaled_dot_production(q,k,v)
      # print(f"dtype={dtype}, ref_o={ref_o}, o={o}, sdpa={sdpa_o}")
      diff_fr = (o - ref_o).abs().max()
      print(diff_fr)
      diff_sr = (sdpa_o - ref_o).abs().max()
      print(diff_sr)


if __name__ == "__main__":
    main()
