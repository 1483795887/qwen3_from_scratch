import torch
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
    """满足 transformers attention interface 的最小桩模块"""

    def __init__(self, n_kv_groups: int = 2):
        super().__init__()
        self.training = False
        self.num_key_value_groups = n_kv_groups


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
        num_blocks_per_batch = (n_seq + block_size - 1) // block_size  # 16
        num_pages_needed = n_batch * num_blocks_per_batch  # 32
        # 计算足够的 mem_size (只需 1 层)
        itemsize = torch.tensor(0, dtype=torch.float32).element_size()
        block_size_in_bytes = 1 * num_heads_kv * head_dim * itemsize * block_size
        mem_size = (num_pages_needed + 4) * 2 * block_size_in_bytes

        kv_cache = PagedKVCache(
            mem_size=mem_size, layers=1, num_heads=num_heads_kv, head_dim=head_dim,
            dtype=torch.float32, block_size=block_size, device=device,
        )

        # block_tables: 顺序映射 batch 0 -> pages[0..15], batch 1 -> pages[16..31]
        block_tables = torch.arange(
            n_batch * num_blocks_per_batch, dtype=torch.int32, device=device
        ).reshape(n_batch, num_blocks_per_batch)
        kv_cache.block_tables = block_tables

        # 将 k, v 写入分页缓存 (update 接收 BSHD 格式)
        kv_cache.update(k.transpose(1, 2), v.transpose(1, 2), layer_idx, cache_pos=0)

        # --- 设置全局上下文 ---
        context = ModelContext(
            use_cache=True,
            kv_cache=kv_cache,
            cache_position=n_seq - 1,  # query 位于最后一个位置, 因果掩码不遮蔽任何 KV
            block_tables=block_tables,
            block_size=block_size,
        )
        old_context = get_forward_context()
        try:
            set_forward_context(context)
            new_o = new_gqa(q, k, v).transpose(1, 2)
        finally:
            set_forward_context(old_context)

        # --- 对比 ---
        assert attn_output.shape == new_o.shape
        assert torch.allclose(attn_output, new_o, atol=1e-5)
