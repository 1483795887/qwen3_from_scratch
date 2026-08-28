import math

import torch
import triton
import triton.language as tl


@triton.jit
def load_paged_memory(
    cache,
    block_tables,
    i_start,
    i_end,
    NUM_HEADS: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    cache: 是PagedAttention的K/V缓存, 总体形状为 (NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE_N),
        这里传入的时候已经加上了 head 的偏差
    block_tables: 是每个block的id, 总体形状为 (BATCH, cdiv(max_seq_len, PAGE_BLOCK_SIZE))
        这里传入的时候已经加上了 batch 的偏差
        长度不及 max 的会在最后填充 -1 , 但 i_end 不会加载到那里
    """
    HIDDEN_DIM = NUM_HEADS * HEAD_DIM
    result = tl.zeros((BLOCK_SIZE_N, HEAD_DIM), dtype=cache.dtype.element_ty)
    dim_offsets = tl.arange(0, HEAD_DIM)
    row_offsets = tl.arange(0, BLOCK_SIZE_N)

    i_size = i_end - i_start
    # 页内起始偏移: 只有 i_start 是 PAGE_BLOCK_SIZE 整倍数时才为 0
    # (decode 时 causal STAGE 2 的 i_start = N_KEY - 1, 不是整倍数)
    block_offset = i_start % PAGE_BLOCK_SIZE
    for i in tl.range(i_start, i_end, PAGE_BLOCK_SIZE):
        block_idx = i // PAGE_BLOCK_SIZE
        block_id = tl.load(block_tables + block_idx)
        # 通过 i_end 可以保证 block_id > 0, 而且 triton 中无法写 break 和 continue 就不写了
        loaded_rows = i - i_start
        # 从 -loaded_heads 开始加载 PAGE_BLOCK_SIZE'
        # BLOCK_SIZE_N 中加载 [loaded_heads, loaded_heads + PAGE_BLOCK_SIZE)
        # 所以 mask 要把前后的给遮掉
        # 页内偏移 = block_offset, 页内可加载量 = PAGE_BLOCK_SIZE - block_offset
        global_row_offsets = (
            block_id * PAGE_BLOCK_SIZE
            - loaded_rows
            + row_offsets[:, None]
            + block_offset
        )
        block_data = tl.load(
            cache + global_row_offsets * HIDDEN_DIM + dim_offsets[None, :],
            mask=(
                (row_offsets[:, None])
                < tl.minimum(
                    i_size, loaded_rows + PAGE_BLOCK_SIZE - block_offset
                )
            )
            & (row_offsets[:, None] >= loaded_rows),
            other=0.0,
        )
        result += block_data
    return result


@triton.jit
def load_contiguous_memory(
    ptr,
    i_start,
    i_end,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    ptr: 指向 (N, NUM_HEADS, HEAD_DIM) 的数据，传入时已经加上了 head 的偏差
    """
    HIDDEN_DIM = NUM_HEADS * HEAD_DIM
    row_offsets = tl.arange(0, BLOCK_SIZE_N)
    dim_offsets = tl.arange(0, HEAD_DIM)
    mask = row_offsets[:, None] < tl.minimum(i_end - i_start, BLOCK_SIZE_N)
    global_row_offsets = i_start + row_offsets[:, None]
    return tl.load(
        ptr + global_row_offsets * HIDDEN_DIM + dim_offsets[None, :],
        mask=mask,
        other=0.0,
    )


@triton.jit
def flash_attention_intr(
    data_q,
    K_ptr,
    V_ptr,
    result_o,
    max_val,
    dominator,
    N_KEY,
    scale,
    offsets_m,
    block_tables,
    STAGE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    cache_type: tl.constexpr,
):
    dtype = data_q.dtype
    # query 块在请求内的全局起始位置: already_cached + start_m * BLOCK_SIZE_M
    m_start = tl.min(offsets_m)
    if STAGE == 1:
        lo, hi = 0, tl.minimum(m_start, N_KEY)
    elif STAGE == 2:
        lo, hi = (
            tl.minimum(m_start, N_KEY),
            tl.minimum(m_start + BLOCK_SIZE_M, N_KEY),
        )
    else:
        lo, hi = 0, N_KEY
    for k in tl.range(lo, hi, BLOCK_SIZE_N, warp_specialize=True):
        k = tl.multiple_of(k, BLOCK_SIZE_N)
        i_end = tl.minimum(hi, k + BLOCK_SIZE_N)
        offsets_n = k + tl.arange(0, BLOCK_SIZE_N)
        if cache_type == 0:
            data_k = load_contiguous_memory(
                K_ptr,
                k,
                i_end,
                NUM_HEADS,
                HEAD_DIM,
                BLOCK_SIZE_N,
            )
            data_v = load_contiguous_memory(
                V_ptr,
                k,
                i_end,
                NUM_HEADS,
                HEAD_DIM,
                BLOCK_SIZE_N,
            )
        else:
            data_k = load_paged_memory(
                K_ptr,
                block_tables,
                k,
                i_end,
                NUM_HEADS,
                PAGE_BLOCK_SIZE,
                HEAD_DIM,
                BLOCK_SIZE_N,
            )
            data_v = load_paged_memory(
                V_ptr,
                block_tables,
                k,
                i_end,
                NUM_HEADS,
                PAGE_BLOCK_SIZE,
                HEAD_DIM,
                BLOCK_SIZE_N,
            )
        attn = tl.dot(data_q, data_k.T) * scale
        attn = tl.where(offsets_n[None, :] < i_end, attn, -float("inf"))
        if STAGE == 2:
            attn = tl.where(
                offsets_m[:, None] >= offsets_n[None, :], attn, -float("inf")
            )
        tmp_max = tl.max(attn, axis=-1, keep_dims=True)
        new_max_val = tl.maximum(max_val, tmp_max)
        attn = attn - new_max_val
        exp_attn = tl.math.exp2(attn)

        scale_factor = tl.math.exp2(max_val - new_max_val)
        dominator = dominator * scale_factor + tl.sum(
            exp_attn, axis=-1, keep_dims=True
        )
        max_val = new_max_val
        exp_attn = exp_attn.to(dtype)
        result_o = result_o * scale_factor + tl.dot(exp_attn, data_v)
    return result_o, max_val, dominator


@triton.jit
def flash_attention_decode_intr(
    data_q,
    K_ptr,
    V_ptr,
    result_o,
    max_val,
    dominator,
    N_KEY,
    scale,
    block_tables,
    BLOCK_SIZE_N: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    dtype = data_q.dtype
    for k in tl.range(0, N_KEY, BLOCK_SIZE_N, warp_specialize=True):
        k = tl.multiple_of(k, BLOCK_SIZE_N)
        i_end = tl.minimum(N_KEY, k + BLOCK_SIZE_N)
        offsets_n = k + tl.arange(0, BLOCK_SIZE_N)
        data_k = load_paged_memory(
            K_ptr,
            block_tables,
            k,
            i_end,
            NUM_HEADS,
            PAGE_BLOCK_SIZE,
            HEAD_DIM,
            BLOCK_SIZE_N,
        )
        data_v = load_paged_memory(
            V_ptr,
            block_tables,
            k,
            i_end,
            NUM_HEADS,
            PAGE_BLOCK_SIZE,
            HEAD_DIM,
            BLOCK_SIZE_N,
        )
        attn = tl.dot(data_q, data_k.T) * scale
        attn = tl.where(offsets_n[None, :] < i_end, attn, -float("inf"))
        tmp_max = tl.max(attn, axis=-1, keep_dims=True)
        new_max_val = tl.maximum(max_val, tmp_max)
        attn = attn - new_max_val
        exp_attn = tl.math.exp2(attn)

        scale_factor = tl.math.exp2(max_val - new_max_val)
        dominator = dominator * scale_factor + tl.sum(
            exp_attn, axis=-1, keep_dims=True
        )
        max_val = new_max_val
        exp_attn = exp_attn.to(dtype)
        result_o = result_o * scale_factor + tl.dot(exp_attn, data_v)
    return result_o, max_val, dominator


@triton.jit
def flash_attn_varlen_kernel(
    Q,
    K,
    V,
    output,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seq_len_k,
    scale,
    block_tables,
    NUM_HEADS_KV: tl.constexpr,
    groups: tl.constexpr,
    cache_type: tl.constexpr,
    causal: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
):
    b_id = tl.program_id(2)
    h_id = tl.program_id(1)
    h_id_kv = h_id // groups
    n_q_id = tl.program_id(0)
    result_o = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    cu_seqlen_q_start = tl.load(cu_seqlens_q + b_id)
    cu_seqlen_q_end = tl.load(cu_seqlens_q + b_id + 1)
    cu_seqlen_k_start = tl.load(cu_seqlens_k + b_id)
    cu_seqlen_k_end = tl.load(cu_seqlens_k + b_id + 1)
    N_KEY = cu_seqlen_k_end - cu_seqlen_k_start
    N_Q = cu_seqlen_q_end - cu_seqlen_q_start
    already_cached = N_KEY - N_Q

    HIDDEN_DIM = NUM_HEADS_KV * HEAD_DIM
    HIDDEN_DIM_Q = groups * HIDDEN_DIM
    Q_ptr = Q + cu_seqlen_q_start * HIDDEN_DIM_Q + h_id * HEAD_DIM
    if cache_type == 0:
        K_ptr = K + cu_seqlen_k_start * HIDDEN_DIM + h_id_kv * HEAD_DIM
        V_ptr = V + cu_seqlen_k_start * HIDDEN_DIM + h_id_kv * HEAD_DIM
    else:
        K_ptr = K + h_id_kv * HEAD_DIM
        V_ptr = V + h_id_kv * HEAD_DIM
    O_ptr = output + cu_seqlen_q_start * HIDDEN_DIM_Q + h_id * HEAD_DIM
    if block_tables is not None:
        num_blocks_per_seq = (
            max_seq_len_k + PAGE_BLOCK_SIZE - 1
        ) // PAGE_BLOCK_SIZE
        block_tables = block_tables + b_id * num_blocks_per_seq

    offsets_qm = n_q_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    global_qm = offsets_qm + already_cached
    offsets_qd = tl.arange(0, HEAD_DIM)
    mask_m = offsets_qm[:, None] < N_Q
    mask_d = offsets_qd < HEAD_DIM

    data_q = tl.load(
        Q_ptr + offsets_qm[:, None] * HIDDEN_DIM_Q + offsets_qd[None, :],
        mask=mask_m & mask_d,
        other=0.0,
    )
    max_val = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32) - float("inf")
    dominator = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    if causal:
        result_o, max_val, dominator = flash_attention_intr(
            data_q,
            K_ptr,
            V_ptr,
            result_o,
            max_val,
            dominator,
            N_KEY,
            scale,
            global_qm,
            block_tables,
            1,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            PAGE_BLOCK_SIZE,
            NUM_HEADS_KV,
            HEAD_DIM,
            cache_type,
        )
        result_o, max_val, dominator = flash_attention_intr(
            data_q,
            K_ptr,
            V_ptr,
            result_o,
            max_val,
            dominator,
            N_KEY,
            scale,
            global_qm,
            block_tables,
            2,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            PAGE_BLOCK_SIZE,
            NUM_HEADS_KV,
            HEAD_DIM,
            cache_type,
        )
    else:
        result_o, max_val, dominator = flash_attention_intr(
            data_q,
            K_ptr,
            V_ptr,
            result_o,
            max_val,
            dominator,
            N_KEY,
            scale,
            global_qm,
            block_tables,
            3,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            PAGE_BLOCK_SIZE,
            NUM_HEADS_KV,
            HEAD_DIM,
            cache_type,
        )
    dtype = Q.dtype.element_ty
    result_o = (result_o / dominator).to(dtype)
    tl.store(
        O_ptr + offsets_qm[:, None] * HIDDEN_DIM_Q + offsets_qd[None, :],
        result_o,
        mask=mask_d & mask_m,
    )


@triton.jit
def flash_attn_varlen_decode_kernel(
    Q,
    K,
    V,
    output,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seq_len_k,
    scale,
    block_tables,
    NUM_HEADS_KV: tl.constexpr,
    groups: tl.constexpr,
    cache_type: tl.constexpr,
    causal: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
):
    """decode 特化（N_Q==1 且 groups>1）：GQA 折叠。

    原 kernel 按 q-head 展开 grid（每个 q-head 独立加载自己 kv-head 的 KV 页），
    GQA groups=2 时同一 KV 页被两个 q-head 重复读 → 2× 冗余内存流量。
    本 kernel 按 kv-head 展开，把 groups 个 q-head 折进 BLOCK_SIZE_M 行
    （行 → token=row//groups、head=kv_id*groups+row%groups），KV 页只加载一次。
    """
    b_id = tl.program_id(2)
    h_id_kv = tl.program_id(1)  # 按 kv-head 展开（原 kernel 按 q-head）
    n_q_id = tl.program_id(0)  # cdiv(N_Q*groups, BLOCK_SIZE_M)
    result_o = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    cu_seqlen_q_start = tl.load(cu_seqlens_q + b_id)
    cu_seqlen_q_end = tl.load(cu_seqlens_q + b_id + 1)
    cu_seqlen_k_start = tl.load(cu_seqlens_k + b_id)
    cu_seqlen_k_end = tl.load(cu_seqlens_k + b_id + 1)
    N_KEY = cu_seqlen_k_end - cu_seqlen_k_start
    N_Q = cu_seqlen_q_end - cu_seqlen_q_start  # decode=1，prefill 分段=chunk
    N_ROWS = N_Q * groups  # 折叠后的 M 行数
    # 无实际行的程序：N_KEY 0 → k 循环为空 → 不读 KV、store 被 mask 全遮。
    # （不用早退 return：会破坏 warp_specialize 的控制流约束。）
    # 混合 batch 中 grid[0] 按最大 N_Q 展开，N_Q 小的请求（decode）只有 n_q_id=0
    # 有真实行，其余程序空转。
    has_rows = n_q_id * BLOCK_SIZE_M < N_ROWS
    N_KEY = tl.where(has_rows, N_KEY, 0)
    already_cached = N_KEY - N_Q

    HIDDEN_DIM = NUM_HEADS_KV * HEAD_DIM
    HIDDEN_DIM_Q = groups * HIDDEN_DIM
    if cache_type == 0:
        K_ptr = K + cu_seqlen_k_start * HIDDEN_DIM + h_id_kv * HEAD_DIM
        V_ptr = V + cu_seqlen_k_start * HIDDEN_DIM + h_id_kv * HEAD_DIM
    else:
        K_ptr = K + h_id_kv * HEAD_DIM
        V_ptr = V + h_id_kv * HEAD_DIM
    if block_tables is not None:
        num_blocks_per_seq = (
            max_seq_len_k + PAGE_BLOCK_SIZE - 1
        ) // PAGE_BLOCK_SIZE
        block_tables = block_tables + b_id * num_blocks_per_seq

    offsets_qm = n_q_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_qd = tl.arange(0, HEAD_DIM)
    # 行 → (token, q-head)：token = row//groups、head = kv_id*groups + row%groups。
    # head 偏移 = kv_id*groups*HEAD_DIM + (row%groups)*HEAD_DIM。
    head_off = (offsets_qm % groups) * HEAD_DIM
    tok_off = (offsets_qm // groups) * HIDDEN_DIM_Q
    group_head_stride = groups * HEAD_DIM
    global_qm = already_cached + (offsets_qm // groups)  # 同组两行同一位置
    mask_m = offsets_qm[:, None] < N_ROWS
    mask_d = offsets_qd < HEAD_DIM

    data_q = tl.load(
        Q
        + cu_seqlen_q_start * HIDDEN_DIM_Q
        + h_id_kv * group_head_stride
        + (tok_off + head_off)[:, None]
        + offsets_qd[None, :],
        mask=mask_m & mask_d,
        other=0.0,
    )
    max_val = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32) - float("inf")
    dominator = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    if causal:
        result_o, max_val, dominator = flash_attention_intr(
            data_q,
            K_ptr,
            V_ptr,
            result_o,
            max_val,
            dominator,
            N_KEY,
            scale,
            global_qm,
            block_tables,
            1,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            PAGE_BLOCK_SIZE,
            NUM_HEADS_KV,
            HEAD_DIM,
            cache_type,
        )
        result_o, max_val, dominator = flash_attention_intr(
            data_q,
            K_ptr,
            V_ptr,
            result_o,
            max_val,
            dominator,
            N_KEY,
            scale,
            global_qm,
            block_tables,
            2,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            PAGE_BLOCK_SIZE,
            NUM_HEADS_KV,
            HEAD_DIM,
            cache_type,
        )
    else:
        result_o, max_val, dominator = flash_attention_intr(
            data_q,
            K_ptr,
            V_ptr,
            result_o,
            max_val,
            dominator,
            N_KEY,
            scale,
            global_qm,
            block_tables,
            3,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            PAGE_BLOCK_SIZE,
            NUM_HEADS_KV,
            HEAD_DIM,
            cache_type,
        )
    dtype = Q.dtype.element_ty
    result_o = (result_o / dominator).to(dtype)
    tl.store(
        output
        + cu_seqlen_q_start * HIDDEN_DIM_Q
        + h_id_kv * group_head_stride
        + (tok_off + head_off)[:, None]
        + offsets_qd[None, :],
        result_o,
        mask=mask_d & mask_m,
    )


@triton.jit
def flash_attn_decode_kernel(
    Q,
    K,
    V,
    output,
    context_lens,
    scale,
    block_tables,
    block_tables_stride,
    NUM_HEADS_KV: tl.constexpr,
    NUM_HEADS_Q: tl.constexpr,
    groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
):
    """decode 特化：GQA 折叠。
    N长度为1，因果没有效果
    decode 时所有的Q长度都是1, 于是可以把 Groups 个 头部一起加载, 和同一个组的 KV 相乘
    """
    b_id = tl.program_id(1)
    h_id_kv = tl.program_id(0)  # 按 kv-head 展开
    result_o = tl.zeros((NUM_HEADS_Q, HEAD_DIM), dtype=tl.float32)

    N_KEY = tl.load(context_lens + b_id)

    HIDDEN_DIM = NUM_HEADS_KV * HEAD_DIM
    HIDDEN_DIM_Q = groups * HIDDEN_DIM
    K_ptr = K + h_id_kv * HEAD_DIM
    V_ptr = V + h_id_kv * HEAD_DIM
    block_tables = block_tables + b_id * block_tables_stride
    offsets_qm = tl.arange(0, NUM_HEADS_Q)
    offsets_qd = tl.arange(0, HEAD_DIM)
    # 行 → (token, q-head)：token = row//groups、head = kv_id*groups + row%groups。
    # head 偏移 = kv_id*groups*HEAD_DIM + (row%groups)*HEAD_DIM。
    head_off = (offsets_qm % groups) * HEAD_DIM
    tok_off = (offsets_qm // groups) * HIDDEN_DIM_Q
    group_head_stride = groups * HEAD_DIM
    # decode 每条序列只有 1 个 token，有效行只有 groups 行（token 0 的 groups 个 q-head）。
    # 不遮住的话，行 ≥ groups 会读 q[b + row//groups]（越界/跨 batch 元素）并写回
    # output[b + row//groups]（跨 batch 行写竞态 + 越界写，图重放时偶发乱码 / IMA）。
    N_ROWS = groups
    mask_m = offsets_qm[:, None] < N_ROWS
    mask_d = offsets_qd < HEAD_DIM

    data_q = tl.load(
        Q
        + b_id * HIDDEN_DIM_Q
        + h_id_kv * group_head_stride
        + (tok_off + head_off)[:, None]
        + offsets_qd[None, :],
        mask=mask_m & mask_d,
        other=0.0,
    )
    max_val = tl.zeros((NUM_HEADS_Q, 1), dtype=tl.float32) - float("inf")
    dominator = tl.zeros((NUM_HEADS_Q, 1), dtype=tl.float32)
    result_o, max_val, dominator = flash_attention_decode_intr(
        data_q,
        K_ptr,
        V_ptr,
        result_o,
        max_val,
        dominator,
        N_KEY,
        scale,
        block_tables,
        BLOCK_SIZE_N,
        PAGE_BLOCK_SIZE,
        NUM_HEADS_KV,
        HEAD_DIM,
    )
    dtype = Q.dtype.element_ty
    result_o = (result_o / dominator).to(dtype)
    tl.store(
        output
        + b_id * HIDDEN_DIM_Q
        + h_id_kv * group_head_stride
        + (tok_off + head_off)[:, None]
        + offsets_qd[None, :],
        result_o,
        mask=mask_d & mask_m,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    block_table: torch.Tensor | None = None,
):
    # 内部使用 exp2 减少乘法运算
    softmax_scale *= math.log2(math.e)
    H_q, D = q.shape[1:]
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    if block_table is None:
        # 连续内存: k, v 为 (total_kv, num_heads_kv, head_dim)
        PAGE_BLOCK_SIZE = 0
        H_k = k.shape[1]
        cache_type = 0
    else:
        # 分页内存: k, v 为缓存 (num_pages, block_size, num_heads_kv, head_dim)
        # block_table 为 (B, blocks_per_seq)
        PAGE_BLOCK_SIZE = k.shape[1]
        H_k = k.shape[2]
        cache_type = 1
    output = torch.empty_like(q)
    groups = H_q // H_k
    if groups > 1 and causal:
        # GQA 折叠：decode（N_Q=1）与 chunked prefill（N_Q=chunk）都适用——
        # 每 kv-head 的 KV 页只加载一次，内存流量减半；混合 batch 中无实际行的
        # 程序在 kernel 内早退。
        # T9（ncu 引导调参，2026-08-14）：纯 decode 批（max_seqlen_q==1，
        # 折叠后 N_ROWS=2）把 BLOCK_SIZE_M 缩到 16——regs 116→72、achieved
        # occupancy 31→55%、DRAM 72→92%，kernel 378→298µs（-21%，逼近 DRAM
        # 地板）。混合/prefill 批保持 32：M 缩小会让 causal 前缀被更多 M-tile
        # 重读，KV 流量翻倍（TTFT 风险），且 decode 程序的 dummy 程序数翻倍。
        if max_seqlen_q == 1:
            BLOCK_SIZE_M = 16
            launch_kwargs = {"num_warps": 4, "num_stages": 3}
        else:
            # 混合/prefill 路径保持 triton 默认启动参数，零改动
            launch_kwargs = {}
        grid = [
            triton.cdiv(max_seqlen_q * groups, BLOCK_SIZE_M),
            H_k,
            cu_seqlens_q.shape[0] - 1,
        ]
        flash_attn_varlen_decode_kernel[grid](
            q,
            k,
            v,
            output,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_k,
            softmax_scale,
            block_table,
            H_k,
            groups,
            cache_type,
            causal,
            D,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            PAGE_BLOCK_SIZE,
            **launch_kwargs,
        )
        return output
    grid = [
        triton.cdiv(max_seqlen_q, BLOCK_SIZE_M),
        H_q,
        cu_seqlens_q.shape[0] - 1,
    ]
    flash_attn_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_k,
        softmax_scale,
        block_table,
        H_k,
        groups,
        cache_type,
        causal,
        D,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        PAGE_BLOCK_SIZE,
    )

    return output


def flash_attn_decode_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
):
    # 内部使用 exp2 减少乘法运算
    softmax_scale *= math.log2(math.e)
    bs, H_q, D = q.shape
    BLOCK_SIZE_N = 32
    PAGE_BLOCK_SIZE = k_cache.shape[1]
    H_k = k_cache.shape[2]
    output = torch.empty_like(q)
    groups = H_q // H_k
    grid = [H_k, bs]
    flash_attn_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        output,
        context_lens,
        softmax_scale,
        block_tables,
        block_tables.shape[1],
        H_k,
        H_q,
        groups,
        D,
        BLOCK_SIZE_N,
        PAGE_BLOCK_SIZE,
    )
    return output


@triton.jit
def _update_paged_kv_cache_kernel(
    k_cache, v_cache, k, v, slot_mapping, HIDDEN_DIM: tl.constexpr
):
    n_id = tl.program_id(0)

    slot = tl.load(slot_mapping + n_id)
    if slot < 0:
        return
    offsets = tl.arange(0, HIDDEN_DIM)
    k_cache_ptr = k_cache + (slot * HIDDEN_DIM + offsets)
    v_cache_ptr = v_cache + (slot * HIDDEN_DIM + offsets)
    k_ptr = k + (n_id * HIDDEN_DIM + offsets)
    v_ptr = v + (n_id * HIDDEN_DIM + offsets)

    item_k = tl.load(k_ptr)
    item_v = tl.load(v_ptr)

    target_dtype = k_cache.dtype.element_ty
    tl.store(k_cache_ptr, item_k.to(target_dtype))
    tl.store(v_cache_ptr, item_v.to(target_dtype))


def update_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N = k.shape[0]
    HIDDEN_DIM = k.shape[1] * k.shape[2]
    _update_paged_kv_cache_kernel[(N,)](
        k_cache, v_cache, k, v, slot_mapping, HIDDEN_DIM
    )
