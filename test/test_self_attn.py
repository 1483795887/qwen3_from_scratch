import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.models.attn import create_causal_attention_mask
from qwen3_from_scratch.models.context import ModelContext
from qwen3_from_scratch.models.parameter_loader import ParameterLoader


@pytest.mark.parametrize("component_type", ["base"])
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
    with torch.no_grad():
        out = self_attn(x, context)
        assert out.shape == x.shape

@pytest.mark.parametrize("component_type", ["base"])
def test_self_attn_output_close_to_transformers(
    model_config, qwen3_config, component_type, device
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
        output = self_attn(x, context)
        position_embeddings = context.position_embeddings
        attn_mask = create_causal_attention_mask(x.shape[1], x.device, x.dtype)
        off_output, _ = off_self_attn(
            x,
            position_ids=context.position_ids,
            position_embeddings=(
                position_embeddings.cos_embed,
                position_embeddings.sin_embed,
            ),
            attention_mask=attn_mask,
        )
        assert torch.allclose(output, off_output, atol=1e-2)