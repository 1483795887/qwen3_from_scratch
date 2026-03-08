import os

import pytest
from transformers import Qwen3Config

from qwen3_from_scratch.factory import load_from_file
from qwen3_from_scratch.utils.env import load_env_file


@pytest.fixture()
def model_path():
    load_env_file()
    return os.environ.get("MODEL_PATH")


@pytest.fixture()
def model_config(model_path):
    model_path = os.environ.get("MODEL_PATH")
    return load_from_file(model_path + "/config.json")


@pytest.fixture()
def qwen3_config(model_path):
    return Qwen3Config.from_pretrained(model_path)


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        metafunc.parametrize("device", ["cpu", "cuda"])


def pytest_runtest_setup(item):
    device_param = item.funcargs.get("device")
    if device_param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
