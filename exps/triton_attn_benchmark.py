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

PROVIDERS = Literal["base", "my_op", "my_op_flash"]

for dtype in [torch.float32]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[64 * i for i in range(1, 32)],
            line_arg="provider",
            line_vals=["base", "my_op", "my_op_flash"],
            line_names=["Torch", "MyOp", "MyOpFlash"],
            ylabel="TFLOPS",
            plot_name=f"attn-{dtype}",
            args={"dtype": dtype, "D": 1024, "head_dim": 128, "num_q_heads": 16, "num_kv_heads": 8},
        )
    )


@triton.testing.perf_report(configs)
def benchmark(seq_len: int, provider: PROVIDERS, dtype: torch.dtype, D: int, head_dim: int, num_q_heads: int, num_kv_heads: int):
    config = ModelConfig(
        hidden_size=D,
        head_dim=head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        norm_params={"eps": 1e-5},
    )
    batch_size = 2
    q = torch.randn((batch_size, num_q_heads, seq_len, head_dim), dtype=dtype, device=DEVICE)
    k = torch.randn((batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype, device=DEVICE)
    v = torch.randn((batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype, device=DEVICE)
    attn_op = ComponentFactory.create(
        "attn",
        config=config,
        component_impl=provider,
    ).to(DEVICE)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: attn_op(q, k, v), quantiles=quantiles
    )
    # QK^T: batch * num_q_heads * seq_q * seq_k * head_dim
    # softmax: batch * num_q_heads * seq_q * seq_k
    # attn * V: batch * num_q_heads * seq_q * seq_k * head_dim
    # 总计算量: 2 * batch * num_q_heads * seq_q * seq_k * head_dim
    perf = lambda ms: 2 * batch_size * num_q_heads * seq_len * seq_len * head_dim * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
