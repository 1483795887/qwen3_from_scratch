import pytest
import torch

from qwen3_from_scratch.factory.config import ModelConfig
from qwen3_from_scratch.inference.context import (
    ModelContext,
    set_forward_context,
)
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.models.rotary import build_cos_sin_table
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


def _fill_cos_sin(
    context: ModelContext,
    config: ModelConfig,
    position_ids: torch.Tensor,
    device,
    dtype,
):
    """按 position_ids 切片最大长度表填到 context.cos / context.sin。"""
    cos_t, sin_t = build_cos_sin_table(
        config.head_dim,
        config.max_position_embeddings,
        config.pos_embed_params["rope_theta"],
        device,
        dtype,
    )
    context.cos = cos_t[position_ids.reshape(-1).long()]
    context.sin = sin_t[position_ids.reshape(-1).long()]


def test_parameter_loading(real_model_config, real_model_path, device):
    pytest.skip("这个操作比较耗时，暂且跳过")
    loader = ParameterLoader()
    loader.load(real_model_path)
    model = Qwen3(real_model_config).to(device)
    model.load_state(loader)
    x = torch.tensor([1, 2, 3, 4]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.bfloat16
    position_ids = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
    context.position_ids = position_ids
    _fill_cos_sin(context, real_model_config, position_ids, device, x.dtype)
    set_forward_context(context)
    with torch.no_grad():
        y = model(x)
        assert y.shape == (1, 4, real_model_config.vocab_size)


def test_qwen3_forward_requires_cos_sin(device):
    """Qwen3.forward 不设 context.cos / context.sin 时应 assert 失败。"""
    config = _SMALL_MODEL_CONFIG
    model = Qwen3(config).to(device)
    x = torch.tensor([1, 2, 3]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.float32
    set_forward_context(context)
    with pytest.raises(AssertionError):
        model(x)


def test_qwen3_forward_with_position_ids(device):
    """Qwen3.forward 设 position_ids + cos/sin 后正常输出。"""
    config = _SMALL_MODEL_CONFIG
    model = Qwen3(config).to(device)
    x = torch.tensor([1, 2, 3]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.float32
    position_ids = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
    context.position_ids = position_ids
    _fill_cos_sin(context, config, position_ids, device, x.dtype)
    set_forward_context(context)
    with torch.no_grad():
        y = model(x)
        assert y.shape == (1, 3, config.vocab_size)
