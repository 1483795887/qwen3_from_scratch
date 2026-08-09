import pytest
import torch

from qwen3_from_scratch.factory.config import ModelConfig
from qwen3_from_scratch.inference.context import (
    ModelContext,
    set_forward_context,
)
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()

# 完整模型专用的缩小配置，跑完整 Qwen3 前向更快
_SMALL_MODEL_CONFIG = ModelConfig(
    vocab_size=512,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=512,
    max_position_embeddings=1024,
)


def test_parameter_loading(real_model_config, real_model_path, device):
    pytest.skip("这个操作比较耗时，暂且跳过")
    loader = ParameterLoader()
    loader.load(real_model_path)
    model = Qwen3(real_model_config).to(device)
    model.load_state(loader)
    x = torch.tensor([1, 2, 3, 4]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.bfloat16
    context.position_ids = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
    set_forward_context(context)
    with torch.no_grad():
        y = model(x)
        assert y.shape == (1, 4, real_model_config.vocab_size)


def test_qwen3_forward_requires_position_ids(device):
    """Qwen3.forward 不设 position_ids 时应 assert 失败（RoPE 内部 assert）。"""
    config = _SMALL_MODEL_CONFIG
    model = Qwen3(config).to(device)
    x = torch.tensor([1, 2, 3]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.float32
    set_forward_context(context)
    with pytest.raises(AssertionError):
        model(x)


def test_qwen3_forward_with_position_ids(device):
    """Qwen3.forward 设 position_ids 后正常输出。"""
    config = _SMALL_MODEL_CONFIG
    model = Qwen3(config).to(device)
    x = torch.tensor([1, 2, 3]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.float32
    context.position_ids = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
    set_forward_context(context)
    with torch.no_grad():
        y = model(x)
        assert y.shape == (1, 3, config.vocab_size)
