import copy

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.models.context import ModelContext


@pytest.mark.parametrize("component_type", ["base"])
def test_rope(model_config, component_type, device):
    shape = (2, 16, 1024, 128)
    context = ModelContext()
    for rope_type in ["normal", "neox"]:
        config = copy.deepcopy(model_config)
        config.pos_embed_type = rope_type
        rope = ComponentFactory.create(
            "rope", config, component_impl=component_type
        ).to(device)
        x = torch.randn(shape).to(device)
        with torch.no_grad():
            x = rope(x, context)
            assert x.shape == shape


@pytest.mark.parametrize("component_type", ["base"])
def test_rope_with_position_inputs(model_config, component_type, device):
    shape = (2, 16, 1024, 128)
    context = ModelContext()
    context.position_ids = torch.arange(0, 1024).view(1, -1)
    for rope_type in ["normal", "neox"]:
        config = copy.deepcopy(model_config)
        config.pos_embed_type = rope_type
        rope = ComponentFactory.create(
            "rope", config, component_impl=component_type
        ).to(device)
        x = torch.randn(shape).to(device)
        with torch.no_grad():
            x = rope(x, context)
            assert x.shape == shape


@pytest.mark.parametrize("component_type", ["base"])
def test_rope_against_transformers(
    model_config, qwen3_config, component_type, device
):
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
        context.position_embeddings = new_rope.build_cos_sin_embed(
            k.dtype, context.position_ids
        )
        position_ids = torch.arange(0, n_seq).view(1, -1).to(device)
        position_embeddings = official_rope(v, position_ids=position_ids)

        hidden_shape = (*k.shape[:-1], -1, model_config.head_dim)
        k = k.view(hidden_shape).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)
        official_k, official_v = apply_rotary_pos_emb(
            k, v, *position_embeddings
        )
        new_k = new_rope(k, context)
        new_v = new_rope(v, context)
        assert torch.allclose(official_k, new_k, atol=1e-5)
        assert torch.allclose(official_v, new_v, atol=1e-5)
