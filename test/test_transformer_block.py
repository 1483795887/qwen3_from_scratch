import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.models.attn import create_causal_attention_mask
from qwen3_from_scratch.models.parameter_loader import ParameterLoader


@pytest.mark.parametrize("component_type", ["base"])
def test_transformer_block_shape_correct(
    model_config, model_path, component_type, device
):
    loader = ParameterLoader()
    loader.load(model_path)
    transformer_block = ComponentFactory.create(
        "decoder_layer",
        model_config,
        name="model.layers.8",
        layer_idx=8,
        component_impl=component_type,
    ).to(device)
    transformer_block.load_state(loader)
    n_seq = 256
    x = torch.randn(
        2, n_seq, model_config.hidden_size, dtype=torch.bfloat16
    ).to(device)
    context = ModelContext()
    context.position_ids = torch.arange(0, n_seq).view(1, -1).to(device)
    with torch.no_grad():
        output = transformer_block(x, context)
        assert output.shape == x.shape


@pytest.mark.parametrize("component_type", ["base"])
def test_transformer_block_output_close_to_transformers(
    model_config, model_path, component_type, device
):
    transformer_block = ComponentFactory.create(
        "decoder_layer",
        model_config,
        name="model.layers.8",
        layer_idx=8,
        component_impl=component_type,
    ).to(device)
    qwen3_config = Qwen3Config.from_pretrained(model_path)
    hf_decoder_layer = Qwen3DecoderLayer(qwen3_config, layer_idx=8).to(device)
    hf_rotary_embed = Qwen3RotaryEmbedding(config=qwen3_config).to(device)

    transformer_block_states = transformer_block.state_dict()

    hf_decoder_layer.load_state_dict(transformer_block_states)
    with torch.no_grad():
        torch.manual_seed(42)
        x = torch.randn(2, 256, model_config.hidden_size).to(device)
        context = ModelContext()
        position_ids = torch.arange(0, x.shape[1]).view(1, -1).to(device)

        context.position_ids = position_ids
        attention_mask = create_causal_attention_mask(
            x.shape[1], x.device, x.dtype
        )
        position_embeddings = hf_rotary_embed(x, position_ids=position_ids)

        output = transformer_block(x, context)
        hf_output = hf_decoder_layer(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        assert torch.allclose(output, hf_output, atol=1e-5, rtol=1e-5)
