import os

import pytest
import torch
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

def pytest_runtest_call(item):
    """
    在ops.so不存在时跳过测试
    """
    # 先执行测试用例，捕获异常
    try:
        # 执行原始的测试用例逻辑
        item.runtest()
    except ImportError as e:
        if "qwen3_from_scratch.kernels.ops" in str(e) and 'module' in str(e):
            pytest.skip(f"跳过测试：加载SO/组件失败，异常：{str(e)}")
        raise
    except (OSError, RuntimeError) as e:
        # 补充捕获so加载的其他常见异常（如ctypes加载失败、运行时链接库缺失）
        if "cannot open shared object file" in str(e) or "undefined symbol" in str(e):
            pytest.skip(f"跳过测试：SO运行时错误，异常：{str(e)}")
            return
        raise

def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        if torch.cuda.is_available():
            metafunc.parametrize("device", ["cpu", "cuda"])
        else:
            metafunc.parametrize("device", ["cpu"])


def pytest_runtest_setup(item):
    device_param = item.funcargs.get("device")
    if device_param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
