import torch

from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.utils.env import load_env_file

load_env_file()


def test_parameter_loading(model_config, model_path, device):
    loader = ParameterLoader()
    loader.load(model_path)
    model = Qwen3(model_config).to(device)
    model.load_state(loader)
    x = torch.tensor([1, 2, 3, 4]).unsqueeze(0).to(device)
    context = ModelContext()
    context.dtype = torch.bfloat16
    with torch.no_grad():
        y = model(x, context)
        assert y.shape == (1, 4, model_config.vocab_size)
