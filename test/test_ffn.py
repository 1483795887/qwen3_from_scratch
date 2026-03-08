import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.models.parameter_loader import ParameterLoader


@pytest.mark.parametrize("component_type", ["base"])
def test_ffn_load(model_path, model_config, component_type, device):
    loader = ParameterLoader()
    loader.load(model_path)
    ffn = ComponentFactory.create(
        "mlp",
        model_config,
        name="model.layers.7.mlp",
        component_impl=component_type,
    ).to(device)
    ffn.load_state(loader)
    with torch.no_grad():
        x = torch.rand(2, 128, 1024, dtype=torch.bfloat16).to(device)
        y = ffn(x)
        assert x.shape == y.shape


@pytest.mark.parametrize("component_type", ["base"])
def test_ffn_compare_with_transformers(
    model_config, qwen3_config, component_type, device
):
    ffn = ComponentFactory.create(
        "mlp",
        model_config,
        name="model.layers.7.mlp",
        component_impl=component_type,
    ).to(device)
    transformers_ffn = Qwen3MLP(qwen3_config).to(device)
    ffn_states = ffn.state_dict()
    transformers_ffn.load_state_dict(ffn_states)
    with torch.no_grad():
        x = torch.rand(2, 128, 1024).to(device)
        y1 = ffn(x)
        y2 = transformers_ffn(x)
        assert torch.allclose(y1, y2, atol=1e-5)
