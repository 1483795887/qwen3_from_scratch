
import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.inference.kv_cache.simple_kv_cache import SimpleKVCache
from qwen3_from_scratch.models.attn import create_causal_attention_mask
from qwen3_from_scratch.inference.context import KVCache, ModelContext
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
        2, model_config.num_key_value_heads, 100, model_config.head_dim
    ).to(device)
    cache_v = torch.randn(
        2, model_config.num_key_value_heads, 100, model_config.head_dim
    ).to(device)
    context.kv_cache = {3: KVCache(cache_k, cache_v)}
    context.use_cache = True

    with torch.no_grad():
        x = torch.randn(2, 1, model_config.hidden_size).to(device)
        out = self_attn(x, context)
        assert out.shape == (2, 1, model_config.hidden_size)


@pytest.mark.parametrize("component_type", ["base"])
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


@pytest.mark.parametrize("component_type", ["base"])
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
        2, model_config.num_key_value_heads, 100, model_config.head_dim
    ).to(device)
    cache_v = torch.randn(
        2, model_config.num_key_value_heads, 100, model_config.head_dim
    ).to(device)
    context.kv_cache = {3: KVCache(cache_k, cache_v)}
    context.use_cache = True

    past_key_values.update(cache_k, cache_v, 3)

    with torch.no_grad():
        torch.manual_seed(42)
        x = torch.randn(2, 1, model_config.hidden_size).to(device)
        output = self_attn(x, context)
        position_embeddings = context.position_embeddings
        off_output, _ = off_self_attn(
            x,
            position_ids=context.position_ids,
            position_embeddings=(
                position_embeddings.cos_embed,
                position_embeddings.sin_embed,
            ),
            attention_mask=None,
            past_key_values=past_key_values,
        )
        assert torch.allclose(output, off_output, atol=1e-2)
