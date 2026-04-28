import triton
import torch
import matplotlib.pyplot as plt
from typing import Literal

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.factory.config import ModelConfig

DEVICE = triton.runtime.driver.active.get_active_torch_device()

PROVIDERS = Literal["base", "my_op", "my_op_flash"]


def measure_memory(provider: PROVIDERS, seq_len: int, dtype: torch.dtype, D: int, head_dim: int, num_q_heads: int, num_kv_heads: int):
    """测量指定配置下的最大显存占用"""
    config = ModelConfig(
        hidden_size=D,
        head_dim=head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        norm_params={"eps": 1e-5},
    )
    batch_size = 2

    # 创建输入张量（不计算梯度）
    with torch.no_grad():
        q = torch.randn((batch_size, num_q_heads, seq_len, head_dim), dtype=dtype, device=DEVICE)
        k = torch.randn((batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype, device=DEVICE)
        v = torch.randn((batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype, device=DEVICE)

    # 创建算子
    attn_op = ComponentFactory.create(
        "attn",
        config=config,
        component_impl=provider,
    ).to(DEVICE)
    attn_op.eval()

    # 清空缓存并同步
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 记录初始显存
    torch.cuda.reset_peak_memory_stats(DEVICE)
    initial_memory = torch.cuda.memory_allocated(DEVICE)

    # 前向传播
    with torch.no_grad():
        _ = attn_op(q, k, v)

    # 同步并记录峰值显存
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(DEVICE)

    # 计算实际占用（峰值减去初始）
    memory_used = (peak_memory - initial_memory) / (1024 ** 2)  # 转换为 MB

    # 清理
    del q, k, v, attn_op
    torch.cuda.empty_cache()

    return memory_used


def run_benchmark():
    """运行显存基准测试"""
    # 配置参数
    D = 1024
    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 8
    dtype = torch.float32

    # 序列长度范围
    seq_lengths = [64 * i for i in range(1, 33)]  # 64, 128, ..., 2048

    # 存储结果
    results = {provider: [] for provider in ["base", "my_op", "my_op_flash"]}

    print(f"Running memory benchmark (D={D}, head_dim={head_dim}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, dtype={dtype})")
    print("-" * 80)
    print(f"{'Seq Len':>10} | {'base (MB)':>12} | {'my_op (MB)':>12} | {'my_op_flash (MB)':>18}")
    print("-" * 80)

    for seq_len in seq_lengths:
        row_data = [seq_len]
        for provider in ["base", "my_op", "my_op_flash"]:
            try:
                memory = measure_memory(provider, seq_len, dtype, D, head_dim, num_q_heads, num_kv_heads)
                results[provider].append(memory)
                row_data.append(f"{memory:.2f}")
            except Exception as e:
                results[provider].append(float('nan'))
                row_data.append("N/A")
                print(f"Error with {provider} at seq_len={seq_len}: {e}")

        print(f"{row_data[0]:>10} | {row_data[1]:>12} | {row_data[2]:>12} | {row_data[3]:>18}")

    print("-" * 80)

    # 绘制图表
    plt.figure(figsize=(12, 6))

    for provider in ["base", "my_op", "my_op_flash"]:
        plt.plot(seq_lengths, results[provider], marker='o', label=provider, linewidth=2)

    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Memory Usage (MB)", fontsize=12)
    plt.title(f"Attention Memory Usage vs Sequence Length\n(D={D}, head_dim={head_dim}, {num_q_heads}Q/{num_kv_heads}KV heads, {dtype})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图表
    output_path = "/home/hego/exercise/tech/dl/qwen3_from_scratch/exps/attn_memory_benchmark.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    # 显示图表
    plt.show()

    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        run_benchmark()
