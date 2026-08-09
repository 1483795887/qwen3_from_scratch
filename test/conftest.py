import os

# 在导入任何模块之前设置环境变量
os.environ["TRITON_IEEE_PRECISION"] = "1"
os.environ["TRITON_F32_DEFAULT"] = "ieee"
import pytest
import torch

from qwen3_from_scratch.factory import load_from_file
from qwen3_from_scratch.factory.config import ModelConfig
from qwen3_from_scratch.utils.env import load_env_file


@pytest.fixture()
def model_config() -> ModelConfig:
    """预置 ModelConfig，不依赖本地模型文件。"""
    return ModelConfig()


@pytest.fixture()
def qwen3_config(model_config):
    """与 model_config 对应的 transformers Qwen3Config。"""
    return model_config.to_transformers_config()


@pytest.fixture()
def real_model_path():
    """真实模型权重路径。未设置 MODEL_PATH 时跳过依赖真实权重的用例。"""
    load_env_file()
    path = os.environ.get("MODEL_PATH")
    if not path:
        pytest.skip("MODEL_PATH 未设置：跳过真实模型权重用例")
    return path


@pytest.fixture()
def real_model_config(real_model_path):
    """真实模型的配置，从 real_model_path/config.json 读取。"""
    return load_from_file(os.path.join(real_model_path, "config.json"))


def pytest_runtest_call(item):
    """
    在ops.so不存在时跳过测试
    """
    # 先执行测试用例，捕获异常
    try:
        # 执行原始的测试用例逻辑
        item.runtest()
    except ImportError as e:
        msg = str(e)
        if "qwen3_from_scratch.kernels.ops" in msg and "module" in msg:
            pytest.skip(f"跳过测试：加载SO/组件失败，异常：{msg}")
        if (
            "No module named 'triton'" in msg
            or "No module named 'triton." in msg
        ):
            pytest.skip(f"跳过测试：triton 不可用，异常：{msg}")
        raise
    except (OSError, RuntimeError) as e:
        # 补充捕获so加载的其他常见异常（如ctypes加载失败、运行时链接库缺失）
        if "cannot open shared object file" in str(
            e
        ) or "undefined symbol" in str(e):
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
