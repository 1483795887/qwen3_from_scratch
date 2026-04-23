import triton
import triton.language as tl
import torch


@triton.jit
def matrix_multiplication_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_ch,
    stride_cm,
    stride_cn,
    scale,
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_h = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b_h = pid_h // groups

    a_ptr = a + pid_h * stride_ah
    b_ptr = b + pid_b_h * stride_bh
    c_ptr = c + pid_h * stride_ch

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offsets_m[:, None] < M
    mask_n = offsets_n[None, :] < N
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = (
        a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr + offsets_n[None, :] * stride_bn + offsets_k[:, None] * stride_bk
    )
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K):
        block_a = tl.load(a_ptrs, (offsets_k[None, :] < K - k) & mask_m, 0.0)
        block_b = tl.load(b_ptrs, (offsets_k[:, None] < K - k) & mask_n, 0.0)
        acc = tl.dot(block_a, block_b, acc, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    tl.store(
        c_ptr
        + offsets_m[:, None] * stride_cm
        + offsets_n[None, :] * stride_cn,
        acc * scale,
        mask_m & mask_n,
    )


# MxK, NxK=>MxN
@triton.jit
def matrix_multiplication_kernel_with_transpose(
    a,
    b,
    c,
    M,
    N,
    K,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bh,
    stride_bn,
    stride_bk,
    stride_ch,
    stride_cm,
    stride_cn,
    scale,
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_b_h = pid_h // groups
    a_ptr = a + pid_h * stride_ah
    b_ptr = b + pid_b_h * stride_bh
    c_ptr = c + pid_h * stride_ch

    offsets_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    ori_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_n = ori_offsets_n[:, None]
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offsets_m * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_n * stride_bn + offsets_k[None, :] * stride_bk
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K):
        block_a = tl.load(a_ptrs, (offsets_k[None, :] < K - k) & mask_m, 0.0)
        block_b = tl.load(b_ptrs, (offsets_k[None, :] < K - k) & mask_n, 0.0)
        acc = tl.dot(block_a, block_b.T, acc, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    tl.store(
        c_ptr + offsets_m * stride_cm + ori_offsets_n[None, :] * stride_cn,
        acc * scale,
        mask_m & (ori_offsets_n[None, :] < N),
    )


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
        row = tl.exp(row)
        sum_val = tl.sum(row)
        row = row / sum_val
        tl.store(input_ptr, row, mask=mask)


def scaled_dot_production(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True
):
    assert len(Q.shape) == 4
    assert len(K.shape) == 4
    assert len(V.shape) == 4
    assert K.shape == V.shape
    assert K.shape[-1] == Q.shape[-1]

    B, Hq, M, D = Q.shape
    _, Hk, N, _ = K.shape
    assert Hq % Hk == 0
    groups = Hq // Hk

    scale = 1.0 / (D**0.5)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_D = 32
    attn = torch.empty((B, Hq, M, N), dtype=torch.float32, device=Q.device)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), B * Hq)
    matrix_multiplication_kernel_with_transpose[grid](
        Q,
        K,
        attn,
        M,
        N,
        D,
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        attn.stride(1),
        attn.stride(2),
        attn.stride(3),
        scale,
        groups,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_D,
    )

    BLOCK_SIZE = triton.next_power_of_2(N)
    ROW_PER_BLOCK = 32
    grid = (triton.cdiv(M, ROW_PER_BLOCK), B * Hq)
    softmax[grid](
        attn, attn.stride(1), attn.stride(2), attn.stride(3), M, N, is_causal, BLOCK_SIZE
    )

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(D, BLOCK_SIZE_N), B * Hq)
    output = torch.empty_like(Q)
    matrix_multiplication_kernel[grid](
        attn,
        V,
        output,
        M,
        D,
        N,
        attn.stride(1),
        attn.stride(2),
        attn.stride(3),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        1.0,
        groups,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_D,
    )
    return output


@triton.jit
def fused_attention(
    Q,
    K,
    V,
    output,
    M,
    N,
    d,
    scale,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oh,
    stride_om,
    stride_od,
    is_causal: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_qh = tl.program_id(0)  # BxH ，放到一个编号中
    pid_kh = (pid_qh) // groups  # KV 的编号向下整除 组数，不用分离 B和H，自然和Q在同一个B中
    pid = tl.program_id(1)
    result_o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    offsets_qm = (pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offsets_qd = tl.arange(0, BLOCK_SIZE_K)

    Q_ptr = Q + pid_qh * stride_qh
    K_ptr = K + pid_kh * stride_kh
    V_ptr = V + pid_kh * stride_vh
    O_ptr = output + pid_qh * stride_oh

    data_q = tl.load(
        Q_ptr + offsets_qm[:, None] * stride_qm + offsets_qd[None, :] * stride_qd,
        mask=(offsets_qd[None, :] < d) & (offsets_qm[:, None] < M),
        other=0.0,
    )

    offsets_kk = tl.arange(0, BLOCK_SIZE_N)
    data_k = tl.load(
        K_ptr + offsets_kk[:, None] * stride_kn + offsets_qd[None, :] * stride_kd,
        mask=(offsets_qd[None, :] < d) & (offsets_kk[:, None] < N),
        other=0.0,
    )
    data_v = tl.load(
        V_ptr + offsets_kk[:, None] * stride_vn + offsets_qd[None, :] * stride_vd,
        mask=(offsets_qd[None, :] < d) & (offsets_kk[:, None] < N),
        other=0.0,
    )
    attn = tl.dot(data_q, data_k.T) * scale
    boundary_mask_q = (offsets_qm[:, None] < M) & (offsets_kk[None, :] < N)
    if is_causal:
      causual_mask = offsets_qm[:, None] >= offsets_kk[None, :]
      attn = tl.where(boundary_mask_q & causual_mask, attn, -float("inf"))
    else:
      attn = tl.where(boundary_mask_q, attn, -float("inf"))
    max_val = tl.max(attn, axis=-1, keep_dims=True)
    attn = attn - max_val
    exp_attn = tl.exp(attn)

    dominator = tl.sum(exp_attn, axis=-1, keep_dims=True)
    result_o = tl.dot(exp_attn, data_v)

    for k in tl.range(BLOCK_SIZE_N, N, BLOCK_SIZE_N):
        offsets_kk = k + tl.arange(0, BLOCK_SIZE_N)
        data_k = tl.load(
            K_ptr + offsets_kk[:, None] * stride_kn + offsets_qd[None, :] * stride_kd,
            mask=(offsets_qd[None, :] < d) & (offsets_kk[:, None] < N),
            other=0.0,
        )
        data_v = tl.load(
            V_ptr + offsets_kk[:, None] * stride_vn + offsets_qd[None, :] * stride_vd,
            mask=(offsets_qd[None, :] < d) & (offsets_kk[:, None] < N),
            other=0.0,
        )
        attn = tl.dot(data_q, data_k.T) * scale
        causual_mask = offsets_qm[:, None] >= offsets_kk[None, :]
        if is_causal:
          causual_mask = offsets_qm[:, None] >= offsets_kk[None, :]
          attn = tl.where(boundary_mask_q & causual_mask, attn, -float("inf"))
        else:
          attn = tl.where(boundary_mask_q, attn, -float("inf"))
        tmp_max = tl.max(attn, axis=-1, keep_dims=True)
        new_max_val = tl.maximum(max_val, tmp_max)
        attn = attn - new_max_val
        exp_attn = tl.exp(attn)

        scale_factor = tl.exp(max_val - new_max_val)
        dominator = dominator * scale_factor + tl.sum(exp_attn, axis=-1, keep_dims=True)
        max_val = new_max_val
        result_o = result_o * scale_factor + tl.dot(exp_attn, data_v)
    result_o /= dominator
    offsets_om = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_od = tl.arange(0, BLOCK_SIZE_K)

    tl.store(
        O_ptr + offsets_om[:, None] * stride_om + offsets_od[None, :] * stride_od,
        result_o,
        mask=(offsets_od[None, :] < d) & (offsets_om[:, None] < M),
    )

# Q, K, V, output are tensors on the GPU


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal:bool=True):
    assert len(Q.shape) == 4
    assert len(K.shape) == 4
    assert len(V.shape) == 4
    assert K.shape == V.shape
    assert K.shape[-1] == Q.shape[-1]

    B, Hq, M, D = Q.shape
    _, Hk, N, _ = K.shape
    assert Hq % Hk == 0
    groups = Hq // Hk

    output = torch.empty_like(Q)

    scale = 1.0 / (D**0.5)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = triton.next_power_of_2(max(D, 16))
    grid = (Hq * B, triton.cdiv(M, BLOCK_SIZE_M))
    fused_attention[grid](
        Q, K, V, output,
        M, N, D,
        scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        output.stride(1), output.stride(2), output.stride(3),
        is_causal,
        groups,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
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
    q = torch.range(0, 11, dtype=torch.float32, device="cuda").view(
        [1, 1, 3, 4]
    )
    k = torch.range(12, 19, dtype=torch.float32, device="cuda").view(
        [1, 1, 2, 4]
    )
    v = torch.range(20, 27, dtype=torch.float32, device="cuda").view(
        [1, 1, 2, 4]
    )
    ref_o = cpu_forward(q, k, v, 4)
    # o = scaled_dot_production(q, k, v)
    o = flash_attention(q,k,v)
    print(ref_o)
    print(o)


if __name__ == "__main__":
    main()
