import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.factory.config import load_from_file, ModelConfig
from qwen3_from_scratch.utils.env import load_env_file


@dataclass
class BenchmarkResult:
    dim: int
    dtype: str
    torch_time_ms: float
    my_time_ms: float
    speedup: float


def create_model_config(dim: int, base_config: ModelConfig) -> ModelConfig:
    config = ModelConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=dim,
        hidden_act=base_config.hidden_act,
        num_hidden_layers=base_config.num_hidden_layers,
        max_position_embeddings=base_config.max_position_embeddings,
        eos_token_id=base_config.eos_token_id,
        num_key_value_heads=base_config.num_key_value_heads,
        num_attention_heads=base_config.num_attention_heads,
        head_dim=dim,
        intermediate_size=dim * 4,
        norm_type=base_config.norm_type,
        norm_params=base_config.norm_params.copy(),
        pos_embed_type=base_config.pos_embed_type,
        pos_embed_params=base_config.pos_embed_params.copy(),
    )
    return config


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def benchmark_rms_norm(
    base_config: ModelConfig,
    dim: int,
    dtype: str,
    batch_size: int = 16,
    seq_len: int = 512,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda",
) -> BenchmarkResult:
    config = create_model_config(dim, base_config)
    torch_dtype = get_torch_dtype(dtype)

    torch_norm = ComponentFactory.create(
        "norm",
        config=config,
        name="test_norm",
        dim=dim,
        component_impl="base",
    ).to(device)
    torch_norm.weight.data = torch_norm.weight.data.to(torch_dtype)

    my_norm = ComponentFactory.create(
        "norm",
        config=config,
        name="test_norm",
        dim=dim,
        component_impl="my_op",
    ).to(device)
    my_norm.weight.data = my_norm.weight.data.to(torch_dtype)

    x = torch.randn(batch_size, seq_len, dim, dtype=torch_dtype, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = torch_norm(x)
            _ = my_norm(x)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = torch_norm(x)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / num_iterations * 1000

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = my_norm(x)
        torch.cuda.synchronize()
        my_time = (time.perf_counter() - start) / num_iterations * 1000

    speedup = torch_time / my_time if my_time > 0 else 0

    return BenchmarkResult(dim=dim, dtype=dtype, torch_time_ms=torch_time, my_time_ms=my_time, speedup=speedup)


def run_benchmarks(
    dims: List[int],
    dtypes: List[str],
    base_config: ModelConfig,
    device: str = "cuda"
) -> List[BenchmarkResult]:
    if not torch.cuda.is_available():
        print("CUDA is not available, cannot run benchmark")
        return []

    print(f"Running benchmarks on {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("-" * 60)

    results = []
    for dtype in dtypes:
        print(f"\n=== Testing dtype: {dtype} ===")
        for dim in dims:
            print(f"Testing dim={dim}...", end=" ")
            result = benchmark_rms_norm(base_config, dim, dtype)
            results.append(result)
            print(f"Torch: {result.torch_time_ms:.4f}ms, My: {result.my_time_ms:.4f}ms, Speedup: {result.speedup:.2f}x")

    return results


def generate_report(results: List[BenchmarkResult], output_path: str, pics_path: str):
    dtypes = sorted(set(r.dtype for r in results))
    dims = sorted(set(r.dim for r in results))

    df_data = []
    for r in results:
        df_data.append({
            "Dim": r.dim,
            "Dtype": r.dtype,
            "TorchRmsNorm (ms)": r.torch_time_ms,
            "MyRmsNorm (ms)": r.my_time_ms,
            "Speedup": r.speedup,
        })
    df = pd.DataFrame(df_data)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, dtype in enumerate(dtypes):
        dtype_results = [r for r in results if r.dtype == dtype]
        dtype_results.sort(key=lambda x: x.dim)

        x = [r.dim for r in dtype_results]
        torch_times = [r.torch_time_ms for r in dtype_results]
        my_times = [r.my_time_ms for r in dtype_results]

        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        ax.plot(x, torch_times, marker='o', label='TorchRmsNorm', linewidth=2)
        ax.plot(x, my_times, marker='s', label='MyRmsNorm', linewidth=2)
        ax.set_xlabel('Dim')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'RMSNorm Performance - {dtype}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    speedup_ax = axes[1, 1]
    speedup_ax.axis('off')
    speedup_table_data = []
    for dim in dims:
        row = [dim]
        for dtype in dtypes:
            r = next((res for res in results if res.dim == dim and res.dtype == dtype), None)
            if r:
                row.append(f"{r.speedup:.2f}x")
            else:
                row.append("N/A")
        speedup_table_data.append(row)

    table = speedup_ax.table(
        cellText=speedup_table_data,
        colLabels=["Dim"] + dtypes,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    speedup_ax.set_title('Speedup Summary (My/Torch)', y=0.85)

    plt.tight_layout()
    chart_path = f"{pics_path}/rms_norm_benchmark.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to: {chart_path}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    for idx, dtype in enumerate(dtypes):
        dtype_results = [r for r in results if r.dtype == dtype]
        dtype_results.sort(key=lambda x: x.dim)

        x = [r.dim for r in dtype_results]
        speedups = [r.speedup for r in dtype_results]

        axes2[idx].bar(range(len(x)), speedups)
        axes2[idx].set_xticks(range(len(x)))
        axes2[idx].set_xticklabels(x, rotation=45)
        axes2[idx].set_xlabel('Dim')
        axes2[idx].set_ylabel('Speedup (x)')
        axes2[idx].set_title(f'Speedup by Dim - {dtype}')
        axes2[idx].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        axes2[idx].grid(True, alpha=0.3)

    chart_path2 = f"{pics_path}/rms_norm_benchmark_speedup.png"
    plt.savefig(chart_path2, dpi=150)
    plt.close()
    print(f"Speedup chart saved to: {chart_path2}")

    report = f"""# RMSNorm CUDA 性能对比报告

## 测试环境

- **设备**: {torch.cuda.get_device_name(0)}
- **PyTorch 版本**: {torch.__version__}
- **CUDA 版本**: {torch.version.cuda}
- **数据类型**: float32, float16, bfloat16

## 测试配置

- Batch Size: 16
- Seq Len: 512
- Warmup: 10 iterations
- 测试迭代: 100 iterations

## 测试结果

"""

    for dtype in dtypes:
        report += f"\n### {dtype}\n\n"
        report += "| Dim | TorchRmsNorm (ms) | MyRmsNorm (ms) | Speedup |\n"
        report += "|-----|-------------------|----------------|---------|\n"

        dtype_results = [r for r in results if r.dtype == dtype]
        dtype_results.sort(key=lambda x: x.dim)

        for r in dtype_results:
            report += f"| {r.dim} | {r.torch_time_ms:.4f} | {r.my_time_ms:.4f} | {r.speedup:.2f}x |\n"

    report += "\n## 分析\n\n"

    for dtype in dtypes:
        dtype_results = [r for r in results if r.dtype == dtype]
        avg_speedup = sum(r.speedup for r in dtype_results) / len(dtype_results)
        best = max(dtype_results, key=lambda r: r.speedup)
        worst = min(dtype_results, key=lambda r: r.speedup)

        report += f"""### {dtype}
- **平均 Speedup**: {avg_speedup:.2f}x
- **最佳 Dim**: {best.dim} (Speedup: {best.speedup:.2f}x)
- **最差 Dim**: {worst.dim} (Speedup: {worst.speedup:.2f}x)

"""

    report += """## 结论

"""

    for dtype in dtypes:
        dtype_results = [r for r in results if r.dtype == dtype]
        avg_speedup = sum(r.speedup for r in dtype_results) / len(dtype_results)
        if avg_speedup > 1:
            report += f"- **{dtype}**: MyRmsNorm 平均快 {avg_speedup:.2f}x\n"
        else:
            report += f"- **{dtype}**: TorchRmsNorm 平均快 {1/avg_speedup:.2f}x\n"

    report += f"""
## 性能图表

![RMSNorm Performance]({chart_path})

![Speedup by Dim]({chart_path2})
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
    return report


def main():
    import os

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires CUDA.")
        return

    load_env_file()
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        print("Error: MODEL_PATH environment variable not set")
        return

    base_config = load_from_file(model_path + "/config.json")

    dims = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    dtypes = ["float32", "float16", "bfloat16"]

    print("Starting RMSNorm CUDA Performance Benchmark")
    print("=" * 60)

    results = run_benchmarks(dims, dtypes, base_config, device)

    output_path = "exps/reports/rms_norm_benchmark_report.md"
    pics_path = "pics"

    report = generate_report(results, output_path, pics_path)
    print("\n" + "=" * 60)
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
