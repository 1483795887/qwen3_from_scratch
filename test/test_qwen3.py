import pytest
import torch

from qwen3_from_scratch.inference.context import ModelContext, set_forward_context
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()


def test_parameter_loading(model_config, model_path, device):
    pytest.skip("这个操作比较耗时，暂且跳过")
    loader = ParameterLoader()
    loader.load(model_path)
    model = Qwen3(model_config).to(device)
    model.load_state(loader)
    x = torch.tensor([1, 2, 3, 4]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.bfloat16
    context.position_ids = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
    set_forward_context(context)
    with torch.no_grad():
        y = model(x)
        assert y.shape == (1, 4, model_config.vocab_size)


def test_qwen3_forward_requires_position_ids(model_config, device):
    """Qwen3.forward 不设 position_ids 时应 assert 失败（RoPE 内部 assert）。"""
    config = model_config
    model = Qwen3(config).to(device)
    x = torch.tensor([1, 2, 3]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.float32
    set_forward_context(context)
    with pytest.raises(AssertionError):
        model(x)


def test_qwen3_forward_with_position_ids(model_config, device):
    """Qwen3.forward 设 position_ids 后正常输出。"""
    config = model_config
    model = Qwen3(config).to(device)
    x = torch.tensor([1, 2, 3]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.float32
    context.position_ids = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
    set_forward_context(context)
    with torch.no_grad():
        y = model(x)
        assert y.shape == (1, 3, model_config.vocab_size)
