import copy

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.inference.context import ModelContext, set_forward_context


@pytest.mark.parametrize("component_type", ["base", "my_op"])
def test_rope(model_config, component_type, device):
    """PythonRope/MyRope 应用 RoPE 后 shape 不变。"""
    shape = (2, 16, 1024, 128)
    context = ModelContext()
    context.position_ids = torch.arange(0, 1024).view(1, -1).to(device)
    for rope_type in ["normal", "neox"]:
        config = copy.deepcopy(model_config)
        config.pos_embed_type = rope_type
        rope = ComponentFactory.create(
            "rope", config, component_impl=component_type
        ).to(device)
        x = torch.randn(shape).to(device)
        set_forward_context(context)
        with torch.no_grad():
            x = rope(x)
            assert x.shape == shape


@pytest.mark.parametrize("component_type", ["base", "my_op"])
def test_rope_with_position_inputs(model_config, component_type, device):
    """设 position_ids 后 RoPE 正常工作。"""
    shape = (2, 16, 1024, 128)
    context = ModelContext()
    context.position_ids = torch.arange(0, 1024).view(1, -1).to(device)
    for rope_type in ["normal", "neox"]:
        config = copy.deepcopy(model_config)
        config.pos_embed_type = rope_type
        rope = ComponentFactory.create(
            "rope", config, component_impl=component_type
        ).to(device)
        x = torch.randn(shape).to(device)
        set_forward_context(context)
        with torch.no_grad():
            x = rope(x)
            assert x.shape == shape


@pytest.mark.parametrize("component_type", ["base", "my_op"])
def test_rope_against_transformers(
    model_config, qwen3_config, component_type, device
):
    """PythonRope/MyRope 输出与 HF Qwen3RotaryEmbedding 对比一致。"""
    new_rope = ComponentFactory.create(
        "rope", model_config, component_impl=component_type
    ).to(device)
    official_rope = Qwen3RotaryEmbedding(config=qwen3_config).to(device)

    with torch.no_grad():
        torch.manual_seed(42)
        n_seq = 256
        k = torch.randn(2, n_seq, model_config.hidden_size).to(device)
        v = torch.randn(2, n_seq, model_config.hidden_size).to(device)
        context = ModelContext()
        context.position_ids = torch.arange(0, n_seq).view(1, -1).to(device)

        position_ids = torch.arange(0, n_seq).view(1, -1).to(device)
        position_embeddings = official_rope(v, position_ids=position_ids)

        hidden_shape = (*k.shape[:-1], -1, model_config.head_dim)
        k = k.view(hidden_shape).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)
        official_k, official_v = apply_rotary_pos_emb(
            k, v, *position_embeddings
        )
        set_forward_context(context)
        new_k = new_rope(k)
        new_v = new_rope(v)
        assert torch.allclose(official_k, new_k, atol=1e-5)
        assert torch.allclose(official_v, new_v, atol=1e-5)


def test_rope_assert_position_ids_required(model_config, device):
    """不设 position_ids 时 PythonRope.forward 应 assert 失败。"""
    rope = ComponentFactory.create(
        "rope", model_config, component_impl="base"
    ).to(device)
    x = torch.randn(2, 16, 1024, 128).to(device)
    context = ModelContext()
    set_forward_context(context)
    with pytest.raises(AssertionError):
        rope(x)
