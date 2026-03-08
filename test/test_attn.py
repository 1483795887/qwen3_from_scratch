import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.models.attn import (
    create_causal_attention_mask,
)


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


class FakeModule(torch.nn.Module):
    def __init__(self, n_kv_groups: int = 2):
        super().__init__()
        self.training = False
        self.num_key_value_groups = n_kv_groups


@pytest.mark.parametrize("component_type", ["base"])
def test_gqa_against_transformers(
    model_config, qwen3_config, component_type, device
):
    new_gqa = ComponentFactory.create(
        "attn", model_config, component_impl=component_type
    ).to(device)
    n_batch = 2
    n_seq = 256
    transformers_attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        qwen3_config._attn_implementation, eager_attention_forward
    )
    scale = qwen3_config.head_dim**-0.5
    fake_module = FakeModule(2)
    head_dim = qwen3_config.head_dim
    with torch.no_grad():
        q = torch.rand(
            n_batch,
            qwen3_config.num_attention_heads,
            n_seq,
            head_dim,
            device=device,
        )
        k = torch.rand(
            n_batch,
            qwen3_config.num_key_value_heads,
            n_seq,
            head_dim,
            device=device,
        )
        v = torch.rand(
            n_batch,
            qwen3_config.num_key_value_heads,
            n_seq,
            head_dim,
            device=device,
        )
        attn_output, _ = transformers_attention_interface(
            fake_module,
            q,
            k,
            v,
            create_causal_attention_mask(n_seq, q.device, q.dtype),
            dropout=0.0,
            scaling=scale,
        )
        new_o = new_gqa(q, k, v).transpose(1, 2)
        assert attn_output.shape == new_o.shape
        assert torch.allclose(attn_output, new_o, atol=1e-5)


@pytest.mark.parametrize("component_type", ["base"])
def test_gqa_against_transformers_with_cache(
    model_config, component_type, qwen3_config, device
):
    new_gqa = ComponentFactory.create("attn", model_config, component_impl=component_type).to(device)
    n_batch = 2
    n_seq = 256
    qwen3_config._attn_implementation = "sdpa"
    transformers_attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        qwen3_config._attn_implementation, eager_attention_forward
    )
    scale = qwen3_config.head_dim**-0.5
    groups = model_config.num_attention_heads // model_config.num_key_value_heads
    fake_module = FakeModule(groups)
    head_dim = model_config.head_dim
    with torch.no_grad():
        q = torch.rand(
            n_batch, model_config.num_attention_heads, 1, head_dim, device=device
        )
        k = torch.rand(
            n_batch, model_config.num_key_value_heads, n_seq, head_dim, device=device
        )
        v = torch.rand(
            n_batch, model_config.num_key_value_heads, n_seq, head_dim, device=device
        )
        attn_output, _ = transformers_attention_interface(
            fake_module,
            q,
            k,
            v,
            None,
            dropout=0.0,
            scaling=scale,
        )
        new_o = new_gqa(q, k, v).transpose(1, 2)
        assert attn_output.shape == new_o.shape
        assert torch.allclose(attn_output, new_o, atol=1e-5)
