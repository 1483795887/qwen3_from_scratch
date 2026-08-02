
import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.inference.context import (
    ModelContext,
    set_forward_context,
)
from qwen3_from_scratch.inference.kv_cache.paged_cache import PagedKVCache
from qwen3_from_scratch.inference.kv_cache.pre_allocated_kv_cache import (
    PreAllocatedKVCache,
)
from qwen3_from_scratch.models.attn import create_causal_attention_mask
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.rotary import get_rope


def _get_cos_sin_for_hf(model_config, position_ids, device, dtype):
    """从 get_rope 获取 cos/sin，返回 HF apply_rotary_pos_emb 所需的 (cos, sin) 元组。

    HF 期望 (1, S, D) 格式。
    """
    rotary = get_rope(
        model_config.head_dim,
        model_config.head_dim,
        model_config.max_position_embeddings,
        model_config.pos_embed_params["rope_theta"],
    )
    pos = position_ids.reshape(-1).cpu()
    cos_sin = rotary.cos_sin_cache[pos].to(device, dtype)
    half = cos_sin.shape[-1] // 2
    cos = torch.cat([cos_sin[..., :half], cos_sin[..., :half]], dim=-1)
    sin = torch.cat([cos_sin[..., half:], cos_sin[..., half:]], dim=-1)
    cos = cos.squeeze(1)  # (N, D)
    sin = sin.squeeze(1)
    # HF 期望 (1, S, D)
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return cos, sin


@pytest.mark.parametrize("component_type", ["base", "my_op"])
def test_self_attn_shape_correct(
    model_config, model_path, component_type, device
):
    loader = ParameterLoader()
    loader.load(model_path)
    self_attn = ComponentFactory.create(
        "self_attn",
        model_config,
        name="model.layers.8.self_attn",
        component_impl=component_type,
    ).to(device)
    self_attn.load_state(loader)
    n_seq = 256
    x = torch.randn(
        2, n_seq, model_config.hidden_size, dtype=torch.bfloat16
    ).to(device)
    context = ModelContext()
    context.position_ids = torch.arange(0, n_seq).view(1, -1).to(device)
    set_forward_context(context)
    with torch.no_grad():
        out = self_attn(x)
        assert out.shape == x.shape


@pytest.mark.parametrize("component_type", ["base", "my_op"])
def test_self_attn_shape_correct_with_kv_cache(
    model_config, component_type, device
):
    self_attn = ComponentFactory.create(
        "self_attn",
        model_config,
        name="",
        layer_idx=3,
        component_impl=component_type,
    ).to(device)
    context = ModelContext()
    context.position_ids = torch.arange(100, 101).view(1, -1).to(device)
    cache_k = torch.randn(
        2, 100, model_config.num_key_value_heads, model_config.head_dim
    ).to(device)
    cache_v = torch.randn(
        2, 100, model_config.num_key_value_heads, model_config.head_dim
    ).to(device)
    context.kv_cache.update(cache_k, cache_v, 3)
    context.use_cache = True

    set_forward_context(context)
    with torch.no_grad():
        x = torch.randn(2, 1, model_config.hidden_size).to(device)
        out = self_attn(x)
        assert out.shape == (2, 1, model_config.hidden_size)


@pytest.mark.parametrize("component_type", ["base", "my_op"])
def test_self_attn_output_close_to_transformers(
    model_config, model_path, qwen3_config, component_type, device
):
    self_attn = ComponentFactory.create(
        "self_attn", model_config, name="", component_impl=component_type
    ).to(device)
    off_self_attn = Qwen3Attention(qwen3_config, layer_idx=3).to(device)
    off_self_attn.load_state_dict(self_attn.state_dict())

    with torch.no_grad():
        torch.manual_seed(42)
        x = torch.randn(2, 256, model_config.hidden_size).to(device)
        context = ModelContext()
        context.position_ids = torch.arange(0, 256).view(1, -1).to(device)
        set_forward_context(context)
        output = self_attn(x)
        cos_hf, sin_hf = _get_cos_sin_for_hf(
            model_config, context.position_ids, device, x.dtype
        )
        attn_mask = create_causal_attention_mask(x.shape[1], x.device, x.dtype)
        off_output, _ = off_self_attn(
            x,
            position_ids=context.position_ids,
            position_embeddings=(cos_hf, sin_hf),
            attention_mask=attn_mask,
        )
        assert torch.allclose(output, off_output, atol=1e-2)


@pytest.mark.parametrize("component_type", ["base", "my_op"])
def test_self_attn_output_close_to_transformers_with_kv_cache(
    model_config, model_path, qwen3_config, component_type, device
):
    self_attn = ComponentFactory.create(
        "self_attn",
        model_config,
        name="",
        layer_idx=3,
        component_impl=component_type,
    ).to(device)
    off_self_attn = Qwen3Attention(qwen3_config, layer_idx=3).to(device)
    off_self_attn.load_state_dict(self_attn.state_dict())
    past_key_values = DynamicCache(config=qwen3_config)
    context = ModelContext()
    context.position_ids = torch.arange(100, 101).view(1, -1).to(device)
    cache_k = torch.randn(
        2, 100, model_config.num_key_value_heads, model_config.head_dim
    ).to(device)
    cache_v = torch.randn(
        2, 100, model_config.num_key_value_heads, model_config.head_dim
    ).to(device)
    context.kv_cache = PreAllocatedKVCache(1024, 3)
    context.kv_cache.update(cache_k, cache_v, 3)
    context.use_cache = True
    context.cache_position = 100

    past_key_values.update(
        cache_k.transpose(1, 2), cache_v.transpose(1, 2), 3
    )

    with torch.no_grad():
        torch.manual_seed(42)
        x = torch.randn(2, 1, model_config.hidden_size).to(device)
        set_forward_context(context)
        output = self_attn(x)
        cos_hf, sin_hf = _get_cos_sin_for_hf(
            model_config, context.position_ids, device, x.dtype
        )
        off_output, _ = off_self_attn(
            x,
            position_ids=context.position_ids,
            position_embeddings=(cos_hf, sin_hf),
            attention_mask=None,
            past_key_values=past_key_values,
        )
        assert torch.allclose(output, off_output, atol=1e-2)


# ---------------------------------------------------------------------------
# PagedSelfAttention (component_impl="paged_attn") 变长输入测试
# ---------------------------------------------------------------------------


def _build_rope_table(max_pos, head_dim, rope_theta, device, dtype):
    """构建 RoPE cos/sin 表，形状 (max_pos, head_dim)。"""
    positions = torch.arange(max_pos, device=device).float()
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (max_pos, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_pos, D)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _create_paged_kv_cache(model_config, n_seqs, max_seq_len, block_size, device):
    """创建 PagedKVCache 和 block_tables。"""
    num_heads_kv = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_pages_needed = n_seqs * num_blocks_per_seq
    itemsize = torch.tensor(0, dtype=torch.float32).element_size()
    block_size_in_bytes = num_heads_kv * head_dim * itemsize * block_size
    mem_size = (num_pages_needed + 4) * 2 * block_size_in_bytes
    kv_cache = PagedKVCache(
        mem_size=mem_size,
        layers=1,
        num_heads=num_heads_kv,
        head_dim=head_dim,
        dtype=torch.float32,
        block_size=block_size,
        device=device,
    )
    block_tables = torch.arange(
        n_seqs * num_blocks_per_seq, dtype=torch.int32, device=device
    ).reshape(n_seqs, num_blocks_per_seq)
    return kv_cache, block_tables


def _build_var_len_slot_mapping(block_tables, seq_lens, block_size, device):
    """为变长输入构建 slot_mapping，返回扁平张量 (sum(seq_lens),)。"""
    slots = []
    for b, n in enumerate(seq_lens):
        for i in range(n):
            block_idx = i // block_size
            offset = i % block_size
            slot = block_tables[b, block_idx].item() * block_size + offset
            slots.append(slot)
    return torch.tensor(slots, dtype=torch.int32, device=device)


def _build_slot_mapping_for_positions(block_tables, positions, block_size, device):
    """为每条序列的特定位置构建 slot_mapping，返回 (n_seqs,)。"""
    slots = []
    for b, pos in enumerate(positions):
        block_idx = pos // block_size
        offset = pos % block_size
        slot = block_tables[b, block_idx].item() * block_size + offset
        slots.append(slot)
    return torch.tensor(slots, dtype=torch.int32, device=device)


def _build_cum_seq_lens(seq_lens, device):
    """构建累积序列长度张量 [0, s1, s1+s2, ...]。"""
    cum = [0]
    for n in seq_lens:
        cum.append(cum[-1] + n)
    return torch.tensor(cum, dtype=torch.int32, device=device)


def test_paged_self_attn_var_len_shape(model_config, device):
    """PagedSelfAttention 变长输入 shape 测试: 2 条不同长度序列 [4, 8]。"""
    if device == "cuda":
        pytest.skip("PagedSelfAttention CUDA 路径尚未实现")

    layer_idx = 0
    block_size = 16

    paged_attn = ComponentFactory.create(
        "self_attn",
        model_config,
        name="model.layers.8.self_attn",
        layer_idx=layer_idx,
        component_impl="paged_attn",
    ).to(device)

    seq_lens = [4, 8]
    total_seq_len = sum(seq_lens)
    max_seq_len = max(seq_lens)
    n_seqs = len(seq_lens)

    position_ids = torch.tensor(
        [i for n in seq_lens for i in range(n)], dtype=torch.long, device=device
    )

    rope_theta = model_config.pos_embed_params["rope_theta"]
    cos_table, sin_table = _build_rope_table(
        max_seq_len, model_config.head_dim, rope_theta, device, torch.float32
    )

    kv_cache, block_tables = _create_paged_kv_cache(
        model_config, n_seqs, max_seq_len, block_size, device
    )
    slot_mapping = _build_var_len_slot_mapping(
        block_tables, seq_lens, block_size, device
    )
    cum_seq_lens = _build_cum_seq_lens(seq_lens, device)

    x = torch.randn(
        total_seq_len, model_config.hidden_size, dtype=torch.float32, device=device
    )

    context = ModelContext(
        use_cache=True,
        kv_cache=kv_cache,
        position_ids=position_ids,
        block_tables=block_tables,
        block_size=block_size,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens.clone(),
        slot_mapping=slot_mapping,
    )
    set_forward_context(context)
    with torch.no_grad():
        out = paged_attn(x)
        assert out.shape == (total_seq_len, model_config.hidden_size)


def test_paged_self_attn_var_len_prefill(
    model_config, model_path, qwen3_config, device
):
    """PagedSelfAttention 变长 prefill: 2 条不同长度序列 [4, 8]，对比 HF 参考实现。"""
    if device == "cuda":
        pytest.skip("PagedSelfAttention CUDA 路径尚未实现")

    layer_idx = 0
    block_size = 16

    loader = ParameterLoader()
    loader.load(model_path)
    paged_attn = ComponentFactory.create(
        "self_attn",
        model_config,
        name="model.layers.8.self_attn",
        layer_idx=layer_idx,
        component_impl="paged_attn",
    ).to(device)
    paged_attn.load_state(loader)
    paged_attn = paged_attn.float()

    off_self_attn = Qwen3Attention(qwen3_config, layer_idx=layer_idx).to(device)
    off_self_attn.load_state_dict(paged_attn.state_dict())

    seq_lens = [4, 8]
    total_seq_len = sum(seq_lens)
    max_seq_len = max(seq_lens)
    n_seqs = len(seq_lens)

    position_ids = torch.tensor(
        [i for n in seq_lens for i in range(n)], dtype=torch.long, device=device
    )

    rope_theta = model_config.pos_embed_params["rope_theta"]
    cos_table, sin_table = _build_rope_table(
        max_seq_len, model_config.head_dim, rope_theta, device, torch.float32
    )

    kv_cache, block_tables = _create_paged_kv_cache(
        model_config, n_seqs, max_seq_len, block_size, device
    )
    slot_mapping = _build_var_len_slot_mapping(
        block_tables, seq_lens, block_size, device
    )
    cum_seq_lens = _build_cum_seq_lens(seq_lens, device)

    torch.manual_seed(42)
    x = torch.randn(
        total_seq_len, model_config.hidden_size, dtype=torch.float32, device=device
    )

    context = ModelContext(
        use_cache=True,
        kv_cache=kv_cache,
        position_ids=position_ids,
        block_tables=block_tables,
        block_size=block_size,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens.clone(),
        slot_mapping=slot_mapping,
    )
    set_forward_context(context)
    with torch.no_grad():
        output = paged_attn(x)

    # 参考实现: 逐序列运行 HF Qwen3Attention
    ref_parts = []
    offset = 0
    for n in seq_lens:
        x_i = x[offset : offset + n].unsqueeze(0)  # (1, n, hidden)
        pos_ids_i = torch.arange(n, device=device).unsqueeze(0)
        cos_i = cos_table[:n].unsqueeze(0)
        sin_i = sin_table[:n].unsqueeze(0)
        attn_mask = create_causal_attention_mask(n, device, x.dtype)
        off_output, _ = off_self_attn(
            x_i,
            position_ids=pos_ids_i,
            position_embeddings=(cos_i, sin_i),
            attention_mask=attn_mask,
        )
        ref_parts.append(off_output.squeeze(0))
        offset += n
    ref_output = torch.cat(ref_parts, dim=0)

    assert output.shape == ref_output.shape
    assert torch.allclose(output, ref_output, atol=1e-2)


def test_paged_self_attn_var_len_decode(
    model_config, model_path, qwen3_config, device
):
    """PagedSelfAttention 变长 decode: 2 条序列 (已有 KV 16/32)，各生成 1 个新 token。

    流程:
      1. Prefill 阶段: 将已有 token 的 KV 写入 PagedKVCache
      2. Decode 阶段: 用新 token 调用 PagedSelfAttention，从缓存读取全部 KV
      3. 参考: HF Qwen3Attention 全量 prefill 后取最后一个 token 的输出
    """
    if device == "cuda":
        pytest.skip("PagedSelfAttention CUDA 路径尚未实现")

    layer_idx = 0
    block_size = 16

    loader = ParameterLoader()
    loader.load(model_path)
    paged_attn = ComponentFactory.create(
        "self_attn",
        model_config,
        name="model.layers.8.self_attn",
        layer_idx=layer_idx,
        component_impl="paged_attn",
    ).to(device)
    paged_attn.load_state(loader)
    paged_attn = paged_attn.float()

    off_self_attn = Qwen3Attention(qwen3_config, layer_idx=layer_idx).to(device)
    off_self_attn.load_state_dict(paged_attn.state_dict())

    existing_lens = [16, 32]
    n_seqs = len(existing_lens)
    total_existing = sum(existing_lens)
    total_kv_lens = [l + 1 for l in existing_lens]  # [17, 33]
    max_total_kv = max(total_kv_lens)

    rope_theta = model_config.pos_embed_params["rope_theta"]
    cos_table, sin_table = _build_rope_table(
        max_total_kv, model_config.head_dim, rope_theta, device, torch.float32
    )

    kv_cache, block_tables = _create_paged_kv_cache(
        model_config, n_seqs, max_total_kv, block_size, device
    )

    torch.manual_seed(42)
    x_existing = torch.randn(
        total_existing, model_config.hidden_size, dtype=torch.float32, device=device
    )
    x_new = torch.randn(
        n_seqs, model_config.hidden_size, dtype=torch.float32, device=device
    )

    # --- Prefill: 将已有 KV 写入缓存 ---
    existing_pos_ids = torch.tensor(
        [i for n in existing_lens for i in range(n)], dtype=torch.long, device=device
    )
    existing_slot_mapping = _build_var_len_slot_mapping(
        block_tables, existing_lens, block_size, device
    )
    existing_cum = _build_cum_seq_lens(existing_lens, device)

    prefill_ctx = ModelContext(
        use_cache=True,
        kv_cache=kv_cache,
        position_ids=existing_pos_ids,
        block_tables=block_tables,
        block_size=block_size,
        cum_seq_lens_q=existing_cum,
        cum_seq_lens_kv=existing_cum.clone(),
        slot_mapping=existing_slot_mapping,
    )
    set_forward_context(prefill_ctx)
    with torch.no_grad():
        paged_attn(x_existing)  # 输出不需要，只为填充缓存

    # --- Decode: 新 token ---
    new_pos_ids = torch.tensor(
        existing_lens, dtype=torch.long, device=device
    )  # [16, 32]
    new_slot_mapping = _build_slot_mapping_for_positions(
        block_tables, existing_lens, block_size, device
    )
    cum_seq_lens_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    cum_seq_lens_kv = _build_cum_seq_lens(total_kv_lens, device)

    decode_ctx = ModelContext(
        use_cache=True,
        kv_cache=kv_cache,
        position_ids=new_pos_ids,
        block_tables=block_tables,
        block_size=block_size,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        slot_mapping=new_slot_mapping,
    )
    set_forward_context(decode_ctx)
    with torch.no_grad():
        output = paged_attn(x_new)

    # 参考实现: 逐序列全量 prefill，取最后一个 token 输出
    ref_parts = []
    offset = 0
    for i, n in enumerate(existing_lens):
        x_full = torch.cat(
            [x_existing[offset : offset + n], x_new[i : i + 1]]
        ).unsqueeze(0)  # (1, n+1, hidden)
        pos_ids = torch.arange(n + 1, device=device).unsqueeze(0)
        cos_i = cos_table[: n + 1].unsqueeze(0)
        sin_i = sin_table[: n + 1].unsqueeze(0)
        attn_mask = create_causal_attention_mask(n + 1, device, x_full.dtype)
        off_output, _ = off_self_attn(
            x_full,
            position_ids=pos_ids,
            position_embeddings=(cos_i, sin_i),
            attention_mask=attn_mask,
        )
        ref_parts.append(off_output[:, -1].squeeze(0))
        offset += n
    ref_output = torch.stack(ref_parts, dim=0)

    assert output.shape == ref_output.shape
    assert torch.allclose(output, ref_output, atol=1e-2)
