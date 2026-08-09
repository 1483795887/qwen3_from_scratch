import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.inference.context import (
    ModelContext,
    get_forward_context,
    set_forward_context,
)
from qwen3_from_scratch.inference.kv_cache.paged_cache import PagedKVCache


class FakeModule(torch.nn.Module):
    def __init__(self, n_kv_groups: int = 2):
        super().__init__()
        self.training = False
        self.num_key_value_groups = n_kv_groups


def _make_paged_attn(model_config, layer_idx=0, component_impl="paged_attn_torch"):
    return ComponentFactory.create(
        "attn", model_config, component_impl=component_impl, layer_idx=layer_idx
    )


def _create_kv_cache(model_config, n_batch, n_seq, block_size, layer_idx, device):
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    num_blocks_per_batch = (n_seq + block_size - 1) // block_size
    num_pages_needed = n_batch * num_blocks_per_batch
    itemsize = torch.tensor(0, dtype=torch.float32).element_size()
    block_size_in_bytes = 1 * num_heads_kv * head_dim * itemsize * block_size
    mem_size = (num_pages_needed + 4) * 2 * block_size_in_bytes
    num_blocks = PagedKVCache.get_block_num(
        mem_size=mem_size, layers=1, num_heads=num_heads_kv, head_dim=head_dim,
        dtype=torch.float32, block_size=block_size, device=device,
    )
    kv_cache = PagedKVCache(
        num_blocks=num_blocks, layers=1, num_heads=num_heads_kv, head_dim=head_dim,
        dtype=torch.float32, block_size=block_size, device=device,
    )
    block_tables = torch.arange(
        n_batch * num_blocks_per_batch, dtype=torch.int32, device=device
    ).reshape(n_batch, num_blocks_per_batch)
    return kv_cache, block_tables


def _build_slot_mapping(block_tables, seq_lens, max_seq_len, block_size, device):
    """根据 block_tables 构造 slot_mapping。

    对于 batch b 的第 i 个 token (i < seq_lens[b]):
        slot = block_tables[b, i // block_size] * block_size + (i % block_size)
    对于 padding (i >= seq_lens[b]):
        slot = -1 (跳过)

    返回扁平张量, 形状 (n_batch * max_seq_len,)。
    """
    n_batch = block_tables.shape[0]
    token_indices = torch.arange(max_seq_len, device=device)
    block_indices = token_indices // block_size          # (max_seq_len,)
    offsets = token_indices % block_size                 # (max_seq_len,)
    block_ids = block_tables[:, block_indices]           # (n_batch, max_seq_len)
    slots = block_ids * block_size + offsets.unsqueeze(0)  # (n_batch, max_seq_len)
    slot_mapping = slots.reshape(-1)                     # (n_batch * max_seq_len,)

    if isinstance(seq_lens, int):
        seq_lens = [seq_lens] * n_batch
    for b in range(n_batch):
        n = seq_lens[b]
        if n < max_seq_len:
            slot_mapping[b * max_seq_len + n: (b + 1) * max_seq_len] = -1

    return slot_mapping.to(torch.int32)


def _run_paged_attn(paged_attn, q, k, v, kv_cache, block_tables, block_size,
                    slot_mapping, cache_position):
    """在同一个 context 中完成 update + forward。"""
    context = ModelContext(
        use_cache=True,
        kv_cache=kv_cache,
        cache_position=cache_position,
        block_tables=block_tables,
        block_size=block_size,
        slot_mapping=slot_mapping,
    )
    old_context = get_forward_context()
    try:
        set_forward_context(context)
        kv_cache.update(k.transpose(1, 2), v.transpose(1, 2), paged_attn.layer_idx)
        return paged_attn(q, k, v)
    finally:
        set_forward_context(old_context)


def test_torch_paged_attn(model_config, qwen3_config, device):
    """batch=2 decode 场景: 对比 sdpa 参考实现与 paged_attn_torch 的输出。

    流程:
      1. 生成 q(1 token) / k,v(256 tokens), 格式 BHSD
      2. 参考: sdpa 无 mask, q attend 全部 256 个 KV
      3. paged_attn_torch: 将 k,v 写入 PagedKVCache, 通过 context 传递 block_tables 等,
         query 位于最后位置(cache_position=255), 因果掩码不遮蔽任何 KV
      4. 两者 shape 与数值一致
    """
    n_batch = 2
    n_seq = 256
    block_size = 16
    layer_idx = 0

    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    groups = num_heads_q // num_heads_kv

    # --- 创建 paged_attn_torch 模块 ---
    new_gqa = ComponentFactory.create(
        "attn", model_config, component_impl="paged_attn_torch", layer_idx=layer_idx
    ).to(device)

    # --- 参考: transformers sdpa ---
    qwen3_config._attn_implementation = "sdpa"
    transformers_attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        qwen3_config._attn_implementation, eager_attention_forward
    )
    scale = qwen3_config.head_dim ** -0.5
    fake_module = FakeModule(groups)

    with torch.no_grad():
        # q: decode (1 token), k/v: 256 tokens — BHSD 格式
        q = torch.rand(n_batch, 1, num_heads_q, head_dim, device=device).transpose(1, 2)
        k = torch.rand(n_batch, n_seq, num_heads_kv, head_dim, device=device).transpose(1, 2)
        v = torch.rand(n_batch, n_seq, num_heads_kv, head_dim, device=device).transpose(1, 2)

        # --- 参考输出: sdpa (无 mask) ---
        attn_output, _ = transformers_attention_interface(
            fake_module, q, k, v, None, dropout=0.0, scaling=scale,
        )

        # --- PagedKVCache 设置 ---
        kv_cache, block_tables = _create_kv_cache(
            model_config, n_batch, n_seq, block_size, layer_idx, device,
        )
        slot_mapping = _build_slot_mapping(
            block_tables, n_seq, n_seq, block_size, device,
        )

        new_o = _run_paged_attn(
            new_gqa, q, k, v, kv_cache, block_tables, block_size,
            slot_mapping, cache_position=n_seq - 1,
        ).transpose(1, 2)

        # --- 对比 ---
        assert attn_output.shape == new_o.shape
        assert torch.allclose(attn_output, new_o, atol=1e-5)


def test_torch_paged_attn_prefill(model_config, device):
    """batch=2 prefill 场景: 4 query tokens, 4 KV tokens, 因果掩码生效

    generated_len = 0 (没有已有的 KV), 每个 q token 只能 attend 到 <= 自身位置的 kv.
    参考: sdpa with is_causal=True
    """
    n_batch = 2
    n_seq = 4
    block_size = 16
    layer_idx = 0

    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    paged_attn = _make_paged_attn(model_config, layer_idx).to(device)
    kv_cache, block_tables = _create_kv_cache(
        model_config, n_batch, n_seq, block_size, layer_idx, device
    )

    with torch.no_grad():
        q = torch.rand(n_batch, n_seq, num_heads_q, head_dim, device=device).transpose(1, 2)
        k = torch.rand(n_batch, n_seq, num_heads_kv, head_dim, device=device).transpose(1, 2)
        v = torch.rand(n_batch, n_seq, num_heads_kv, head_dim, device=device).transpose(1, 2)

        scale = model_config.head_dim ** -0.5
        ref = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale, enable_gqa=True,
        )

        slot_mapping = _build_slot_mapping(
            block_tables, n_seq, n_seq, block_size, device,
        )
        new_o = _run_paged_attn(
            paged_attn, q, k, v, kv_cache, block_tables, block_size,
            slot_mapping, cache_position=n_seq - 1,
        )

        assert ref.shape == new_o.shape
        assert torch.allclose(ref, new_o, atol=1e-5)


def _per_seq_sdpa_ref(q_shd, k_shd, v_shd, cum_seq_lens_q, cum_seq_lens_kv,
                      num_heads_q, num_heads_kv, head_dim, device, scale, is_causal=True):
    n_seqs = len(cum_seq_lens_q) - 1
    ref_parts = []
    for i in range(n_seqs):
        q_s = cum_seq_lens_q[i]
        q_e = cum_seq_lens_q[i + 1]
        kv_s = cum_seq_lens_kv[i]
        kv_e = cum_seq_lens_kv[i + 1]

        q_i = q_shd[q_s:q_e]
        k_i = k_shd[kv_s:kv_e]
        v_i = v_shd[kv_s:kv_e]

        q_bhsd = q_i.unsqueeze(0).transpose(1, 2)
        k_bhsd = k_i.unsqueeze(0).transpose(1, 2)
        v_bhsd = v_i.unsqueeze(0).transpose(1, 2)

        if is_causal and q_i.shape[0] > 1 and q_i.shape[0] == k_i.shape[0]:
            ref_i = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd,
                is_causal=True, scale=scale, enable_gqa=True,
            )
        elif q_e - q_s == 1:
            ref_i = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd,
                is_causal=False, scale=scale, enable_gqa=True,
            )
        else:
            # 分段 prefill: 新 query token 占据 kv 区间的末尾 q_len 个绝对位置
            # (前面 generated_len 个是已缓存的 KV), 只 attend 绝对位置 < 自身的 kv。
            q_pos = torch.arange(kv_e - (q_e - q_s), kv_e, device=device)
            kv_pos = torch.arange(kv_s, kv_e, device=device)
            causal_mask = q_pos[:, None] < kv_pos[None, :]
            attn_mask = torch.zeros(q_e - q_s, kv_e - kv_s, device=device, dtype=q_shd.dtype)
            attn_mask = attn_mask.masked_fill(causal_mask, float('-inf'))
            ref_i = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd,
                attn_mask=attn_mask, scale=scale, enable_gqa=True,
            )

        ref_parts.append(ref_i.squeeze(0).transpose(0, 1))

    return torch.cat(ref_parts, dim=0)


def _run_var_len_paged_attn(paged_attn, q, k_bshd, v_bshd, kv_cache, block_tables,
                            block_size, cum_seq_lens_q, cum_seq_lens_kv, seq_lens_kv,
                            max_kv_per_seq):
    """var_len 场景: 在同一个 context 中完成 update + forward。

    k_bshd, v_bshd: (n_seqs, max_kv_per_seq, num_heads, head_dim) — BSHD, 含 padding。
    """
    slot_mapping = _build_slot_mapping(
        block_tables, seq_lens_kv, max_kv_per_seq, block_size, q.device,
    )
    context = ModelContext(
        use_cache=True,
        kv_cache=kv_cache,
        block_tables=block_tables,
        block_size=block_size,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        slot_mapping=slot_mapping,
    )
    old_context = get_forward_context()
    try:
        set_forward_context(context)
        kv_cache.update(k_bshd, v_bshd, paged_attn.layer_idx)
        # forward 不使用 k/v 参数 (从缓存读取), 传 dummy
        k_dummy = torch.zeros_like(k_bshd)
        v_dummy = torch.zeros_like(v_bshd)
        return paged_attn(q, k_dummy.reshape(-1, *k_dummy.shape[2:]),
                          v_dummy.reshape(-1, *v_dummy.shape[2:]))
    finally:
        set_forward_context(old_context)


def test_var_len_paged_attn_prefill(model_config, device):
    """var_len prefill: 2 different-length seqs (4, 8 tokens), causal masking.

    Seq 0: q=4, kv=4, generated_len=0
    Seq 1: q=8, kv=8, generated_len=0
    """
    block_size = 16
    layer_idx = 0

    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    scale = head_dim ** -0.5

    n_seqs = 2
    seq_lens = [4, 8]
    cum_seq_lens_q = torch.tensor([0, 4, 12], dtype=torch.int32, device=device)
    cum_seq_lens_kv = torch.tensor([0, 4, 12], dtype=torch.int32, device=device)
    total_q = cum_seq_lens_q[-1].item()
    total_kv = cum_seq_lens_kv[-1].item()
    max_kv_per_seq = max(seq_lens)

    paged_attn = _make_paged_attn(
        model_config, component_impl="var_len_paged_attn", layer_idx=layer_idx
    ).to(device)
    kv_cache, block_tables = _create_kv_cache(
        model_config, n_seqs, max_kv_per_seq, block_size, layer_idx, device
    )

    with torch.no_grad():
        q_shd = torch.rand(total_q, num_heads_q, head_dim, device=device)
        kv_shd = torch.rand(total_kv, num_heads_kv, head_dim, device=device)

        # build BSHD from SHD so cache data matches reference
        k_bshd = torch.zeros(n_seqs, max_kv_per_seq, num_heads_kv, head_dim, device=device)
        v_bshd = torch.zeros(n_seqs, max_kv_per_seq, num_heads_kv, head_dim, device=device)
        offset = 0
        for i in range(n_seqs):
            n = seq_lens[i]
            k_bshd[i, :n] = kv_shd[offset:offset + n]
            v_bshd[i, :n] = kv_shd[offset:offset + n]
            offset += n

        ref = _per_seq_sdpa_ref(
            q_shd, kv_shd, kv_shd, cum_seq_lens_q.tolist(), cum_seq_lens_kv.tolist(),
            num_heads_q, num_heads_kv, head_dim, device, scale, is_causal=True,
        )

        new_o = _run_var_len_paged_attn(
            paged_attn, q_shd, k_bshd, v_bshd, kv_cache, block_tables,
            block_size, cum_seq_lens_q, cum_seq_lens_kv, seq_lens, max_kv_per_seq,
        )

        assert ref.shape == new_o.shape
        assert torch.allclose(ref, new_o, atol=1e-5)


def test_var_len_paged_attn_decode(model_config, device):
    """var_len decode: 2 different-length seqs (kv=16, 32), 1 query token each.

    Seq 0: q=1, kv=16, generated_len=15
    Seq 1: q=1, kv=32, generated_len=31
    """
    block_size = 16
    layer_idx = 0

    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    scale = head_dim ** -0.5

    n_seqs = 2
    seq_lens_kv = [16, 32]
    max_kv_per_seq = max(seq_lens_kv)
    cum_seq_lens_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    cum_seq_lens_kv = torch.tensor([0, 16, 48], dtype=torch.int32, device=device)
    total_q = cum_seq_lens_q[-1].item()
    total_kv = cum_seq_lens_kv[-1].item()

    paged_attn = _make_paged_attn(
        model_config, component_impl="var_len_paged_attn", layer_idx=layer_idx
    ).to(device)
    kv_cache, block_tables = _create_kv_cache(
        model_config, n_seqs, max_kv_per_seq, block_size, layer_idx, device
    )

    with torch.no_grad():
        q_shd = torch.rand(total_q, num_heads_q, head_dim, device=device)

        # shared KV data: one flat SHD source, copied to BSHD for cache
        kv_shd = torch.rand(total_kv, num_heads_kv, head_dim, device=device)
        k_bshd = torch.zeros(n_seqs, max_kv_per_seq, num_heads_kv, head_dim, device=device)
        v_bshd = torch.zeros(n_seqs, max_kv_per_seq, num_heads_kv, head_dim, device=device)
        offset = 0
        for i, n in enumerate(seq_lens_kv):
            k_bshd[i, :n] = kv_shd[offset:offset + n]
            v_bshd[i, :n] = kv_shd[offset:offset + n]
            offset += n

        ref = _per_seq_sdpa_ref(
            q_shd, kv_shd, kv_shd, cum_seq_lens_q.tolist(), cum_seq_lens_kv.tolist(),
            num_heads_q, num_heads_kv, head_dim, device, scale, is_causal=False,
        )

        new_o = _run_var_len_paged_attn(
            paged_attn, q_shd, k_bshd, v_bshd, kv_cache, block_tables,
            block_size, cum_seq_lens_q, cum_seq_lens_kv, seq_lens_kv, max_kv_per_seq,
        )

        assert ref.shape == new_o.shape
        assert torch.allclose(ref, new_o, atol=1e-5)


def test_torch_paged_attn_prefill_with_existing_kv(model_config, device):
    """batch=2 prefill 场景: 4 new tokens + 16 existing KV 已缓存

    Total KV = 20 (16 已有 + 4 新增). generated_len = 16.
    因果掩码: new token at position gen_len + i attends to KV[:gen_len + i + 1].
    参考: sdpa with custom casual attention mask.
    """
    n_batch = 2
    n_existing = 16
    n_prefill = 4
    n_seq = n_existing + n_prefill
    block_size = 16
    layer_idx = 0

    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    paged_attn = _make_paged_attn(model_config, layer_idx).to(device)
    kv_cache, block_tables = _create_kv_cache(
        model_config, n_batch, n_seq, block_size, layer_idx, device
    )

    with torch.no_grad():
        existing_k = torch.rand(n_batch, n_existing, num_heads_kv, head_dim, device=device)
        existing_v = torch.rand(n_batch, n_existing, num_heads_kv, head_dim, device=device)
        new_k = torch.rand(n_batch, n_prefill, num_heads_kv, head_dim, device=device)
        new_v = torch.rand(n_batch, n_prefill, num_heads_kv, head_dim, device=device)

        k_full_bshd = torch.cat([existing_k, new_k], dim=1)
        v_full_bshd = torch.cat([existing_v, new_v], dim=1)
        q = torch.rand(n_batch, n_prefill, num_heads_q, head_dim, device=device).transpose(1, 2)

        # 构造因果掩码: 新 token 在绝对位置 [16, 17, 18, 19]
        q_pos = torch.arange(n_existing, n_seq, device=device)
        kv_pos = torch.arange(n_seq, device=device)
        causal_mask_bool = q_pos[:, None] < kv_pos[None, :]
        attn_mask = torch.zeros(n_prefill, n_seq, device=device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(causal_mask_bool, float('-inf'))

        scale = model_config.head_dim ** -0.5
        ref = F.scaled_dot_product_attention(
            q, k_full_bshd.transpose(1, 2), v_full_bshd.transpose(1, 2),
            attn_mask=attn_mask, scale=scale, enable_gqa=True,
        )

        slot_mapping = _build_slot_mapping(
            block_tables, n_seq, n_seq, block_size, device,
        )
        new_o = _run_paged_attn(
            paged_attn, q,
            k_full_bshd.transpose(1, 2), v_full_bshd.transpose(1, 2),
            kv_cache, block_tables, block_size,
            slot_mapping, cache_position=n_seq - 1,
        )

        assert ref.shape == new_o.shape
        assert torch.allclose(ref, new_o, atol=1e-5)


# ---------------------------------------------------------------------------
# flash_attn_varlen_func (Triton) 包装测试
#
# 红灯阶段: flash_attn_varlen_func 尚未实现, 以下用例因 import 失败而红。
# kernel 落地后自动转绿。参考实现全部用 torch SDPA, 与实现无关。
# ---------------------------------------------------------------------------


def _build_var_len_slot_mapping(block_tables, seq_lens, block_size, device):
    """为变长 SHD 输入构建 slot_mapping, 形状 (sum(seq_lens),)。

    与 _build_slot_mapping 不同: 不做 BSHD padding, 直接对应扁平拼接的 kv_shd。
    """
    slots = []
    for b, n in enumerate(seq_lens):
        for i in range(n):
            block_idx = i // block_size
            offset = i % block_size
            slot = block_tables[b, block_idx].item() * block_size + offset
            slots.append(slot)
    return torch.tensor(slots, dtype=torch.int32, device=device)


def _create_scattered_block_tables(n_seqs, blocks_per_seq, device):
    """打散的 block_tables: 页号逆序, 证明 kernel 走间接寻址而非顺序假设。"""
    ids = torch.arange(
        n_seqs * blocks_per_seq, device=device
    ).reshape(n_seqs, blocks_per_seq)
    return ids.flip(0).flip(1)


def _run_flash_attn_varlen_continuous(q_shd, k_shd, v_shd, cum_seq_lens_q,
                                      cum_seq_lens_kv, max_seqlen_q, max_seqlen_k,
                                      scale, causal=True):
    """连续内存路径: block_table=None, k/v 直接传连续 SHD。"""
    from qwen3_from_scratch.kernels.triton.paged_attn import (
        flash_attn_varlen_func,
    )
    return flash_attn_varlen_func(
        q_shd, k_shd, v_shd,
        max_seqlen_q=max_seqlen_q, cu_seqlens_q=cum_seq_lens_q,
        max_seqlen_k=max_seqlen_k, cu_seqlens_k=cum_seq_lens_kv,
        softmax_scale=scale, causal=causal, block_table=None,
    )


def _run_flash_attn_varlen_paged(model_config, q_shd, kv_shd, seq_lens_kv,
                                 cum_seq_lens_q, cum_seq_lens_kv,
                                 max_seqlen_q, max_seqlen_k, scale, layer_idx=0):
    """分页路径: kv 写入 PagedKVCache, flash_attn_varlen_func 从缓存基地址 + block_table 读。"""
    from qwen3_from_scratch.kernels.triton.paged_attn import (
        flash_attn_varlen_func,
    )

    block_size = 16
    n_seqs = len(seq_lens_kv)
    max_kv_per_seq = max(seq_lens_kv)
    kv_cache, _ = _create_kv_cache(
        model_config, n_seqs, max_kv_per_seq, block_size, layer_idx, q_shd.device,
    )
    blocks_per_seq = (max_kv_per_seq + block_size - 1) // block_size
    block_tables = _create_scattered_block_tables(n_seqs, blocks_per_seq, q_shd.device)
    slot_mapping = _build_var_len_slot_mapping(
        block_tables, seq_lens_kv, block_size, q_shd.device,
    )

    context = ModelContext(
        use_cache=True,
        kv_cache=kv_cache,
        block_tables=block_tables,
        block_size=block_size,
        slot_mapping=slot_mapping,
    )
    old_context = get_forward_context()
    try:
        set_forward_context(context)
        kv_cache.update(kv_shd, kv_shd, layer_idx)
        k_cache, v_cache = kv_cache.get(layer_idx)
        return flash_attn_varlen_func(
            q_shd, k_cache, v_cache,
            max_seqlen_q=max_seqlen_q, cu_seqlens_q=cum_seq_lens_q,
            max_seqlen_k=max_seqlen_k, cu_seqlens_k=cum_seq_lens_kv,
            softmax_scale=scale, causal=True, block_table=block_tables,
        )
    finally:
        set_forward_context(old_context)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attn_varlen_continuous_prefill(model_config):
    """连续内存首 prefill: 2 个不同长度请求 (4, 8), gen=0, 因果掩码。"""
    device = "cuda"
    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    scale = head_dim ** -0.5

    seq_lens = [4, 8]
    cum = torch.tensor([0, 4, 12], dtype=torch.int32, device=device)
    total = 12

    q_shd = torch.rand(total, num_heads_q, head_dim, device=device)
    kv_shd = torch.rand(total, num_heads_kv, head_dim, device=device)

    ref = _per_seq_sdpa_ref(
        q_shd, kv_shd, kv_shd, cum.tolist(), cum.tolist(),
        num_heads_q, num_heads_kv, head_dim, device, scale, is_causal=True,
    )
    o = _run_flash_attn_varlen_continuous(
        q_shd, kv_shd, kv_shd, cum, cum,
        max(seq_lens), max(seq_lens), scale,
    )

    assert o.shape == ref.shape
    assert torch.allclose(o, ref, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attn_varlen_paged_prefill(model_config):
    """分页内存首 prefill: 2 个不同长度请求 (4, 8), 页号打散。"""
    device = "cuda"
    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    scale = head_dim ** -0.5

    seq_lens_kv = [4, 8]
    cum = torch.tensor([0, 4, 12], dtype=torch.int32, device=device)
    total = 12

    q_shd = torch.rand(total, num_heads_q, head_dim, device=device)
    kv_shd = torch.rand(total, num_heads_kv, head_dim, device=device)

    ref = _per_seq_sdpa_ref(
        q_shd, kv_shd, kv_shd, cum.tolist(), cum.tolist(),
        num_heads_q, num_heads_kv, head_dim, device, scale, is_causal=True,
    )
    o = _run_flash_attn_varlen_paged(
        model_config, q_shd, kv_shd, seq_lens_kv,
        cum, cum, max(seq_lens_kv), max(seq_lens_kv), scale,
    )

    assert o.shape == ref.shape
    assert torch.allclose(o, ref, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attn_varlen_continuous_segmented_prefill(model_config):
    """连续内存分段 prefill: 已有 KV (16, 16) + 新 token (4, 8), 绝对位置因果掩码。

    generated_len > 0: cu_seqlens_q != cu_seqlens_kv, query 只 attend 绝对位置 < 自身的 kv。
    """
    device = "cuda"
    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    scale = head_dim ** -0.5

    seq_lens_q = [4, 8]
    seq_lens_kv = [20, 24]  # 16 已缓存 + 新 4/8
    cum_q = torch.tensor([0, 4, 12], dtype=torch.int32, device=device)
    cum_kv = torch.tensor([0, 20, 44], dtype=torch.int32, device=device)
    total_q = 12
    total_kv = 44

    q_shd = torch.rand(total_q, num_heads_q, head_dim, device=device)
    kv_shd = torch.rand(total_kv, num_heads_kv, head_dim, device=device)

    ref = _per_seq_sdpa_ref(
        q_shd, kv_shd, kv_shd, cum_q.tolist(), cum_kv.tolist(),
        num_heads_q, num_heads_kv, head_dim, device, scale, is_causal=True,
    )
    o = _run_flash_attn_varlen_continuous(
        q_shd, kv_shd, kv_shd, cum_q, cum_kv,
        max(seq_lens_q), max(seq_lens_kv), scale,
    )

    assert o.shape == ref.shape
    assert torch.allclose(o, ref, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attn_varlen_paged_segmented_prefill(model_config):
    """分页内存分段 prefill: 已有缓存页 + 追加 prefill, 绝对位置因果掩码。"""
    device = "cuda"
    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    scale = head_dim ** -0.5

    seq_lens_q = [4, 8]
    seq_lens_kv = [20, 24]  # 16 已缓存 + 新 4/8
    cum_q = torch.tensor([0, 4, 12], dtype=torch.int32, device=device)
    cum_kv = torch.tensor([0, 20, 44], dtype=torch.int32, device=device)
    total_q = 12
    total_kv = 44

    q_shd = torch.rand(total_q, num_heads_q, head_dim, device=device)
    kv_shd = torch.rand(total_kv, num_heads_kv, head_dim, device=device)

    ref = _per_seq_sdpa_ref(
        q_shd, kv_shd, kv_shd, cum_q.tolist(), cum_kv.tolist(),
        num_heads_q, num_heads_kv, head_dim, device, scale, is_causal=True,
    )
    o = _run_flash_attn_varlen_paged(
        model_config, q_shd, kv_shd, seq_lens_kv,
        cum_q, cum_kv, max(seq_lens_q), max(seq_lens_kv), scale,
    )

    assert o.shape == ref.shape
    assert torch.allclose(o, ref, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash_attn_varlen_single_request_smoke(model_config):
    """单请求冒烟: 分页, gen=0。"""
    device = "cuda"
    num_heads_q = model_config.num_attention_heads
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    scale = head_dim ** -0.5

    seq_lens_kv = [8]
    cum = torch.tensor([0, 8], dtype=torch.int32, device=device)
    total = 8

    q_shd = torch.rand(total, num_heads_q, head_dim, device=device)
    kv_shd = torch.rand(total, num_heads_kv, head_dim, device=device)

    ref = _per_seq_sdpa_ref(
        q_shd, kv_shd, kv_shd, cum.tolist(), cum.tolist(),
        num_heads_q, num_heads_kv, head_dim, device, scale, is_causal=True,
    )
    o = _run_flash_attn_varlen_paged(
        model_config, q_shd, kv_shd, seq_lens_kv,
        cum, cum, max(seq_lens_kv), max(seq_lens_kv), scale,
    )

    assert o.shape == ref.shape
    assert torch.allclose(o, ref, atol=1e-5)
# ---------------------------------------------------------------------------
# PagedKVCache._update_var_len (分页写入) 测试
#
# 被测接口: 只走 PagedKVCache.update()/get() 公开接口。
# 分派策略: 由 PagedKVCache 内置分派决定(device 参数化 cpu/cuda, 见 conftest)。
#   - 当前 triton 版本未实现, cpu/cuda 均走 torch 参考实现 → 全部绿。
#   - triton 版本落地后, cuda 用例会切换到 triton 分派, 但参考计算
#     (独立 vectorized scatter) 不变, 因此不改用例即可验证 triton 正确性。
#
# 参考实现: _scatter_reference 用 torch 向量化 scatter 独立算出期望缓存,
#   与实现内部循环结构不同, 避免"照抄实现"的 tautology。
# ---------------------------------------------------------------------------


def _scatter_reference(kv_cache, k_shd, v_shd, slot_mapping, layer_idx, block_size):
    """独立参考: 基于未写入前的缓存克隆, 向量化 scatter 期望值。

    未写槽位保持原样(与 torch.empty 初始内容一致), 只改写有效 slot 对应位置,
    从而支持 torch.equal 全量比较且不依赖未初始化内存的具体值。
    """
    ref_k = kv_cache.k_cache[layer_idx].clone()
    ref_v = kv_cache.v_cache[layer_idx].clone()
    valid = slot_mapping != -1
    slots = slot_mapping[valid].long()
    blocks = slots // block_size
    inner = slots % block_size
    ref_k[blocks, inner] = k_shd[valid]
    ref_v[blocks, inner] = v_shd[valid]
    return ref_k, ref_v


def _run_update_var_len(kv_cache, k_shd, v_shd, slot_mapping, layer_idx, block_size):
    """在同一 context 中执行 PagedKVCache.update()。k_shd/v_shd 为 SHD (3D)。"""
    context = ModelContext(
        use_cache=True, kv_cache=kv_cache, block_size=block_size, slot_mapping=slot_mapping,
    )
    old_context = get_forward_context()
    try:
        set_forward_context(context)
        kv_cache.update(k_shd, v_shd, layer_idx)
        return kv_cache.get(layer_idx)
    finally:
        set_forward_context(old_context)


def _make_paged_cache(model_config, num_pages, block_size, layer_idx, device):
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    return PagedKVCache(
        num_blocks=num_pages, layers=1, num_heads=num_heads_kv, head_dim=head_dim,
        dtype=torch.float32, block_size=block_size, device=device,
    )


def _scattered_slot_mapping(block_ids, seq_len, block_size, device):
    """按 (block_id 打散) 顺序填充 seq_len 个连续 token 的 slot, 生成 slot_mapping。

    block_ids: 该序列用到的页号列表(顺序即物理顺序)。token i 落在
    block_ids[i // block_size] 页的 (i % block_size) 槽位。
    """
    slots = []
    for i in range(seq_len):
        block = block_ids[i // block_size]
        slots.append(block * block_size + (i % block_size))
    return torch.tensor(slots, dtype=torch.int32, device=device)


def test_update_var_len_scatter(model_config, device):
    """跨多页打散写入: SHD 输入经 update 后 get 到正确分页缓存。

    block_ids 故意乱序(非 0,1,2 顺序), 证明写入走间接寻址而非顺序假设。
    """
    layer_idx = 0
    block_size = 16
    seq_len = 40
    block_ids = [3, 1, 5]
    num_pages = 8
    num_heads = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    slot_mapping = _scattered_slot_mapping(block_ids, seq_len, block_size, device)
    k_shd = torch.rand(seq_len, num_heads, head_dim, device=device)
    v_shd = torch.rand(seq_len, num_heads, head_dim, device=device)

    kv_cache = _make_paged_cache(model_config, num_pages, block_size, layer_idx, device)
    ref_k, ref_v = _scatter_reference(kv_cache, k_shd, v_shd, slot_mapping, layer_idx, block_size)

    got_k, got_v = _run_update_var_len(kv_cache, k_shd, v_shd, slot_mapping, layer_idx, block_size)

    assert torch.equal(got_k, ref_k)
    assert torch.equal(got_v, ref_v)


def test_update_var_len_skips_invalid_slots(model_config, device):
    """slot_mapping 含 -1 的 padding 槽位必须被跳过, 不写入缓存。"""
    layer_idx = 0
    block_size = 16
    seq_len = 20  # 前 4 个 padding(-1), 后 16 个有效, 落入同一页
    block_ids = [2]
    num_pages = 5
    num_heads = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    # slot_mapping: 前 4 个 -1, 后 16 个有效
    valid_slots = _scattered_slot_mapping(block_ids, 16, block_size, device)
    slot_mapping = torch.cat(
        [torch.full((4,), -1, dtype=torch.int32, device=device), valid_slots]
    )
    k_shd = torch.rand(seq_len, num_heads, head_dim, device=device)
    v_shd = torch.rand(seq_len, num_heads, head_dim, device=device)

    kv_cache = _make_paged_cache(model_config, num_pages, block_size, layer_idx, device)
    ref_k, ref_v = _scatter_reference(kv_cache, k_shd, v_shd, slot_mapping, layer_idx, block_size)

    got_k, got_v = _run_update_var_len(kv_cache, k_shd, v_shd, slot_mapping, layer_idx, block_size)

    assert torch.equal(got_k, ref_k)
    assert torch.equal(got_v, ref_v)