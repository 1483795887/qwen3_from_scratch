import triton
import torch

import torch
from typing import Literal

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.factory.config import (
    ModelConfig,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

configs = []

PROVIDERS = Literal["base", "my_op", "cpp"]

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["D"],
            x_vals=[128 * i for i in range(1, 64)],
            line_arg="provider",
            line_vals=["base", "my_op", "cpp"],
            line_names=["Torch", "triton", "CUDA"],
            ylabel="TFLOPS",
            plot_name=f"rms-norm-{dtype}",
            args={"dtype": dtype},
        )
    )


@triton.testing.perf_report(configs)
def benchmark(D: int, provider: PROVIDERS, dtype: torch.dtype):
    config = ModelConfig(
        hidden_size=D,
        norm_params={"eps": 1e-5},
    )
    a = torch.randn((2, 512, D), dtype=dtype, device=DEVICE)
    norm_op = ComponentFactory.create(
        "norm",
        config=config,
        name="test_norm",
        dim=D,
        component_impl=provider,
    ).to(DEVICE)
    norm_op.weight.data = norm_op.weight.data.to(dtype)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: norm_op(a), quantiles=quantiles
    )
    # 平方，计算量D，求和，计算量D，乘以gamma，计算量系数，计算量系数，计算量 D，共 5D
    perf = lambda ms: 5 * D * 1e-12 / (ms * 1e-3) * dtype.itemsize
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
