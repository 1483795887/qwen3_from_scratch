import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from qwen3_from_scratch import ComponentFactory


@pytest.mark.parametrize("component_type", ["base"])
def test_torch_rms_norm(model_config, component_type, device):
    shape = (2, 128, model_config.hidden_size)
    x = torch.randn(*shape, dtype=torch.float32).to(device)
    rms_norm = ComponentFactory.create(
        "norm",
        model_config,
        name="",
        dim=model_config.hidden_size,
        component_impl=component_type,
    ).to(device)
    with torch.no_grad():
        y = rms_norm(x)
        assert y.shape == shape


@pytest.mark.parametrize("component_type", ["base"])
def test_torch_rms_norm_vs_transformers(model_config, component_type, device):
    shape = (2, 128, model_config.hidden_size)
    x = torch.randn(*shape, dtype=torch.float32).to(device)
    torch_rms_norm = ComponentFactory.create(
        "norm",
        model_config,
        name="",
        dim=model_config.hidden_size,
        component_impl=component_type,
    ).to(device)
    transformers_rms_norm = Qwen3RMSNorm(shape[-1]).to(device)
    torch_rms_norm.load_state_dict(transformers_rms_norm.state_dict())
    with torch.no_grad():
        y_torch = torch_rms_norm(x)
        y_transformers = transformers_rms_norm(x)
        assert torch.allclose(y_torch, y_transformers, atol=1e-5)
