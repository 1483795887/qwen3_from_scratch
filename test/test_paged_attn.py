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
    kv_cache = PagedKVCache(
        mem_size=mem_size, layers=1, num_heads=num_heads_kv, head_dim=head_dim,
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
            q_pos = torch.arange(kv_s, kv_e, device=device)[:q_e - q_s]
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
