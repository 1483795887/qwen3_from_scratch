import pytest
import torch

from qwen3_from_scratch.factory import ComponentFactory


@pytest.mark.parametrize("component_type", ["base"])
def test_gqa_attn_shape_correct(model_config, component_type, device):
    n_batch = 2
    n_seq = 256
    n_head_q = model_config.num_attention_heads
    n_head_kv = model_config.num_key_value_heads
    n_head_dim = model_config.head_dim

    q = torch.rand(n_batch, n_head_q, n_seq, n_head_dim, device=device)
    k = torch.rand(n_batch, n_head_kv, n_seq, n_head_dim, device=device)
    v = torch.rand(n_batch, n_head_kv, n_seq, n_head_dim, device=device)
    attn = ComponentFactory.create(
        "attn", model_config, component_impl=component_type
    ).to(device)
    with torch.no_grad():
        result = attn(q, k, v)
        assert result.shape == q.shape
