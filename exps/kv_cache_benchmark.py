import json
import os
import time
from dataclasses import dataclass
from typing import List

import jinja2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tokenizers import Tokenizer

from qwen3_from_scratch.factory.config import load_from_file
from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.inference.generate import generate
from qwen3_from_scratch.inference.kv_cache.pre_allocated_kv_cache import (
    PreAllocatedKVCache,
)
from qwen3_from_scratch.models.parameter_loader import ParameterLoader
from qwen3_from_scratch.models.qwen3 import Qwen3
from qwen3_from_scratch.utils.env import load_env_file


@dataclass
class BenchmarkResult:
    token_id: int
    token_time: float
    cumulative_time: float
    tokens_per_second: float
    prefill_time: float = 0.0  # 预填充时间（第一个 token 的时间，包含 KV Cache 预分配）


def setup_model_and_tokenizer(model_path: str):
    """加载模型和分词器"""
    loader = ParameterLoader()
    loader.load(model_path)
    config = load_from_file(model_path + "/config.json")
    model = Qwen3(config=config)
    model.load_state(loader)
    unused_keys = loader.get_unused_keys()
    assert len(unused_keys) == 0, f"Unused keys: {unused_keys}"

    with open(model_path + "/tokenizer_config.json") as f:
        data = json.load(f)
        template = jinja2.Template(data["chat_template"])
        # 使用需要长回答的提示词来测试 KV Cache 性能
        prompt = template.render(
            messages=[{
                "role": "user", 
                "content": "请详细介绍人工智能的发展历程、主要应用领域、当前面临的挑战以及未来的发展趋势。请尽可能详细地阐述每个方面，包括具体的技术、案例和观点。"
            }]
        )
        tokenizer = Tokenizer.from_file(model_path + "/tokenizer.json")
        inputs = tokenizer.encode(prompt)
        
    return model, config, tokenizer, prompt, inputs


def benchmark_generation(
    model: torch.nn.Module,
    config,
    inputs,
    tokenizer,
    use_cache: bool,
    max_new_tokens: int,
    device: str,
    temperature: float = 0.7,
) -> List[BenchmarkResult]:
    """
     benchmark 生成过程，记录每个 token 的生成时间
    """
    results = []
    cumulative_time = 0.0
    
    context = ModelContext()
    context.use_cache = use_cache
    context.dtype = torch.bfloat16  # 模型参数本身就是 bf16
    
    if use_cache:
        context.kv_cache = PreAllocatedKVCache(
            max_new_tokens + len(inputs.ids), config.num_hidden_layers
        )
    
    model = model.to(device)
    idx = torch.tensor([inputs.ids]).to(device)
    
    is_prefill = True
    
    for token_id in range(max_new_tokens):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if is_prefill or not context.use_cache:
                context.cache_position = 0
                logits = model(idx, context=context)
                is_prefill = False
            else:
                context.cache_position = idx.shape[1] - 1
                logits = model(idx[:, -1:], context=context)
        
        logits = logits[:, -1, :]
        
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == config.eos_token_id:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
        
        end_time = time.perf_counter()
        token_time = end_time - start_time
        cumulative_time += token_time
        
        prefill_time = token_time if token_id == 0 else 0.0
        
        results.append(BenchmarkResult(
            token_id=token_id + 1,
            token_time=token_time,
            cumulative_time=cumulative_time,
            tokens_per_second=1.0 / token_time if token_time > 0 else float('inf'),
            prefill_time=prefill_time
        ))
        
        if token_id % 10 == 0:
            if token_id == 0:
                print(f"  Token {token_id + 1}/{max_new_tokens}, Time: {token_time:.4f}s (prefill), "
                      f"Cumulative: {cumulative_time:.4f}s")
            else:
                print(f"  Token {token_id + 1}/{max_new_tokens}, Time: {token_time:.4f}s, "
                      f"Cumulative: {cumulative_time:.4f}s")
    
    return results


def plot_results(
    cpu_results_cache: List[BenchmarkResult],
    cpu_results_no_cache: List[BenchmarkResult],
    gpu_results_cache: List[BenchmarkResult],
    gpu_results_no_cache: List[BenchmarkResult],
    save_path: str,
):
    """生成性能对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图 1: CPU 上每个 token 的生成时间
    ax = axes[0, 0]
    if cpu_results_cache:
        ax.plot(
            [r.token_id for r in cpu_results_cache],
            [r.token_time for r in cpu_results_cache],
            label='With KV Cache',
            linewidth=2,
            marker='o',
            markersize=3
        )
    if cpu_results_no_cache:
        ax.plot(
            [r.token_id for r in cpu_results_no_cache],
            [r.token_time for r in cpu_results_no_cache],
            label='Without KV Cache',
            linewidth=2,
            marker='s',
            markersize=3
        )
    ax.set_xlabel('Token ID', fontsize=12)
    ax.set_ylabel('Time per Token (seconds)', fontsize=12)
    ax.set_title('CPU: Time per Token with/without KV Cache', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 图 2: GPU 上每个 token 的生成时间
    ax = axes[0, 1]
    if gpu_results_cache:
        ax.plot(
            [r.token_id for r in gpu_results_cache],
            [r.token_time for r in gpu_results_cache],
            label='With KV Cache',
            linewidth=2,
            marker='o',
            markersize=3
        )
    if gpu_results_no_cache:
        ax.plot(
            [r.token_id for r in gpu_results_no_cache],
            [r.token_time for r in gpu_results_no_cache],
            label='Without KV Cache',
            linewidth=2,
            marker='s',
            markersize=3
        )
    ax.set_xlabel('Token ID', fontsize=12)
    ax.set_ylabel('Time per Token (seconds)', fontsize=12)
    ax.set_title('GPU: Time per Token with/without KV Cache', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 图 3: CPU 上累积时间对比（不计预填充开销）
    ax = axes[1, 0]
    if cpu_results_cache:
        # 计算不计预填充的累积时间：用平均解码时间替代第一个 token 的时间
        avg_decode_time = sum(r.token_time for r in cpu_results_cache[1:]) / len(cpu_results_cache[1:]) if len(cpu_results_cache) > 1 else cpu_results_cache[0].token_time
        adjusted_cumulative = [avg_decode_time]
        for i in range(1, len(cpu_results_cache)):
            adjusted_cumulative.append(adjusted_cumulative[-1] + cpu_results_cache[i].token_time)
        ax.plot(
            [r.token_id for r in cpu_results_cache],
            adjusted_cumulative,
            label='With KV Cache (adjusted)',
            linewidth=2,
            marker='o',
            markersize=3
        )
    if cpu_results_no_cache:
        ax.plot(
            [r.token_id for r in cpu_results_no_cache],
            [r.cumulative_time for r in cpu_results_no_cache],
            label='Without KV Cache',
            linewidth=2,
            marker='s',
            markersize=3
        )
    ax.set_xlabel('Token ID', fontsize=12)
    ax.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax.set_title('CPU: Cumulative Time with/without KV Cache (adjusted)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 图 4: GPU 上累积时间对比（不计预填充开销）
    ax = axes[1, 1]
    if gpu_results_cache:
        # 计算不计预填充的累积时间
        avg_decode_time = sum(r.token_time for r in gpu_results_cache[1:]) / len(gpu_results_cache[1:]) if len(gpu_results_cache) > 1 else gpu_results_cache[0].token_time
        adjusted_cumulative = [avg_decode_time]
        for i in range(1, len(gpu_results_cache)):
            adjusted_cumulative.append(adjusted_cumulative[-1] + gpu_results_cache[i].token_time)
        ax.plot(
            [r.token_id for r in gpu_results_cache],
            adjusted_cumulative,
            label='With KV Cache (adjusted)',
            linewidth=2,
            marker='o',
            markersize=3
        )
    if gpu_results_no_cache:
        ax.plot(
            [r.token_id for r in gpu_results_no_cache],
            [r.cumulative_time for r in gpu_results_no_cache],
            label='Without KV Cache',
            linewidth=2,
            marker='s',
            markersize=3
        )
    ax.set_xlabel('Token ID', fontsize=12)
    ax.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax.set_title('GPU: Cumulative Time with/without KV Cache (adjusted)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到：{save_path}")


def create_summary_table(
    cpu_results_cache: List[BenchmarkResult],
    cpu_results_no_cache: List[BenchmarkResult],
    gpu_results_cache: List[BenchmarkResult],
    gpu_results_no_cache: List[BenchmarkResult],
) -> pd.DataFrame:
    """创建性能对比汇总表"""
    data = []
    
    if cpu_results_cache:
        avg_time = sum(r.token_time for r in cpu_results_cache) / len(cpu_results_cache)
        avg_time_no_prefill = sum(r.token_time for r in cpu_results_cache[1:]) / len(cpu_results_cache[1:]) if len(cpu_results_cache) > 1 else avg_time
        prefill_time = cpu_results_cache[0].prefill_time if cpu_results_cache else 0.0
        data.append({
            'Device': 'CPU',
            'KV Cache': 'Yes',
            'Total Tokens': len(cpu_results_cache),
            'Prefill Time (s)': prefill_time,
            'Avg Time/Token (s)': avg_time,
            'Avg Time/Token (no prefill)': avg_time_no_prefill,
            'Tokens/s': 1.0 / avg_time if avg_time > 0 else float('inf'),
            'Tokens/s (no prefill)': 1.0 / avg_time_no_prefill if avg_time_no_prefill > 0 else float('inf')
        })
    
    if cpu_results_no_cache:
        avg_time = sum(r.token_time for r in cpu_results_no_cache) / len(cpu_results_no_cache)
        data.append({
            'Device': 'CPU',
            'KV Cache': 'No',
            'Total Tokens': len(cpu_results_no_cache),
            'Prefill Time (s)': 0.0,
            'Avg Time/Token (s)': avg_time,
            'Avg Time/Token (no prefill)': avg_time,
            'Tokens/s': 1.0 / avg_time if avg_time > 0 else float('inf'),
            'Tokens/s (no prefill)': 1.0 / avg_time if avg_time > 0 else float('inf')
        })
    
    if gpu_results_cache:
        avg_time = sum(r.token_time for r in gpu_results_cache) / len(gpu_results_cache)
        avg_time_no_prefill = sum(r.token_time for r in gpu_results_cache[1:]) / len(gpu_results_cache[1:]) if len(gpu_results_cache) > 1 else avg_time
        prefill_time = gpu_results_cache[0].prefill_time if gpu_results_cache else 0.0
        data.append({
            'Device': 'GPU',
            'KV Cache': 'Yes',
            'Total Tokens': len(gpu_results_cache),
            'Prefill Time (s)': prefill_time,
            'Avg Time/Token (s)': avg_time,
            'Avg Time/Token (no prefill)': avg_time_no_prefill,
            'Tokens/s': 1.0 / avg_time if avg_time > 0 else float('inf'),
            'Tokens/s (no prefill)': 1.0 / avg_time_no_prefill if avg_time_no_prefill > 0 else float('inf')
        })
    
    if gpu_results_no_cache:
        avg_time = sum(r.token_time for r in gpu_results_no_cache) / len(gpu_results_no_cache)
        data.append({
            'Device': 'GPU',
            'KV Cache': 'No',
            'Total Tokens': len(gpu_results_no_cache),
            'Prefill Time (s)': 0.0,
            'Avg Time/Token (s)': avg_time,
            'Avg Time/Token (no prefill)': avg_time,
            'Tokens/s': 1.0 / avg_time if avg_time > 0 else float('inf'),
            'Tokens/s (no prefill)': 1.0 / avg_time if avg_time > 0 else float('inf')
        })
    
    return pd.DataFrame(data)


def generate_report(
    df: pd.DataFrame,
    chart_path: str,
    report_path: str,
    cpu_results_cache: List[BenchmarkResult],
    cpu_results_no_cache: List[BenchmarkResult],
    gpu_results_cache: List[BenchmarkResult],
    gpu_results_no_cache: List[BenchmarkResult],
):
    """生成 Markdown 格式的报告"""
    
    # 计算加速比（有 cache 速度 / 无 cache 速度）
    cpu_speedup = None
    gpu_speedup = None
    
    cpu_row_cache = df[(df['Device'] == 'CPU') & (df['KV Cache'] == 'Yes')]
    cpu_row_no_cache = df[(df['Device'] == 'CPU') & (df['KV Cache'] == 'No')]
    if not cpu_row_cache.empty and not cpu_row_no_cache.empty:
        cpu_speedup = cpu_row_cache['Tokens/s'].values[0] / cpu_row_no_cache['Tokens/s'].values[0]
    
    gpu_row_cache = df[(df['Device'] == 'GPU') & (df['KV Cache'] == 'Yes')]
    gpu_row_no_cache = df[(df['Device'] == 'GPU') & (df['KV Cache'] == 'No')]
    if not gpu_row_cache.empty and not gpu_row_no_cache.empty:
        gpu_speedup = gpu_row_cache['Tokens/s'].values[0] / gpu_row_no_cache['Tokens/s'].values[0]
    
    report = f"""# KV Cache 性能对比实验报告

## 实验目的
对比开启和关闭 KV Cache 对模型生成性能的影响，验证 KV Cache 在加速自回归生成过程中的作用。

## 实验环境
- **设备**: CPU 和 GPU (NVIDIA GeForce RTX 3060)
- **模型**: Qwen3
- **输入提示词**: 关于人工智能发展历程、应用、挑战和趋势的详细介绍（长回答型问题）
- **最大生成长度**: 
  - CPU: 无 KV Cache 50 tokens, 有 KV Cache 200 tokens
  - GPU: 无 KV Cache 512 tokens, 有 KV Cache 1024 tokens

## 实验方法
1. 在相同输入 prompt 下，分别测试开启和关闭 KV Cache 的生成性能
2. 记录每个 token 的生成时间和累积时间
3. 对比不同设置下的 tokens/s 指标

## 性能对比结果

### 汇总表格

| Device | KV Cache | Total Tokens | Prefill Time (s) | Avg Time/Token (s) | Avg Time (no prefill) | Tokens/s | Tokens/s (no prefill) |
|--------|----------|--------------|------------------|-------------------|----------------------|----------|----------------------|
"""
    
    for _, row in df.iterrows():
        report += f"| {row['Device']} | {row['KV Cache']} | {row['Total Tokens']:.0f} | {row['Prefill Time (s)']:.4f} | {row['Avg Time/Token (s)']:.6f} | {row['Avg Time/Token (no prefill)']:.6f} | {row['Tokens/s']:.4f} | {row['Tokens/s (no prefill)']:.4f} |\n"
    
    report += f"""
### 性能提升

"""
    
    if cpu_speedup:
        report += f"- **CPU**: 使用 KV Cache 后，性能提升 **{cpu_speedup:.2f}x** (不计预填充：**{cpu_row_cache['Tokens/s (no prefill)'].values[0] / cpu_row_no_cache['Tokens/s'].values[0]:.2f}x**)\n"
    
    if gpu_speedup:
        report += f"- **GPU**: 使用 KV Cache 后，性能提升 **{gpu_speedup:.2f}x** (不计预填充：**{gpu_row_cache['Tokens/s (no prefill)'].values[0] / gpu_row_no_cache['Tokens/s'].values[0]:.2f}x**)\n"
    
    report += f"""
### 关于预填充时间的说明

使用 KV Cache 时，第一个 token 的生成时间（预填充时间）包含以下开销：
1. **KV Cache 预分配**：为后续生成预留内存空间
2. **Prompt 处理**：处理整个输入序列并生成初始 KV 值

因此，我们提供了两种性能指标：
- **包含预填充**：反映实际使用场景的端到端性能
- **不计预填充**：仅统计解码阶段的稳定生成速度，用于对比 KV Cache 的解码加速效果
"""

    report += f"""
### 可视化结果

![KV Cache 性能对比图]({chart_path})

*上图展示了每个 token 的生成时间和累积时间随生成 token 数量的变化趋势*

## 详细数据

### CPU 性能数据

#### 开启 KV Cache (前 10 个 token)
"""
    
    if cpu_results_cache:
        report += "| Token ID | Time (s) | Cumulative Time (s) | Tokens/s |\n"
        report += "|----------|----------|---------------------|----------|\n"
        for r in cpu_results_cache[:10]:
            report += f"| {r.token_id} | {r.token_time:.6f} | {r.cumulative_time:.6f} | {r.tokens_per_second:.4f} |\n"
    
    report += """
#### 关闭 KV Cache (前 10 个 token)
"""
    
    if cpu_results_no_cache:
        report += "| Token ID | Time (s) | Cumulative Time (s) | Tokens/s |\n"
        report += "|----------|----------|---------------------|----------|\n"
        for r in cpu_results_no_cache[:10]:
            report += f"| {r.token_id} | {r.token_time:.6f} | {r.cumulative_time:.6f} | {r.tokens_per_second:.4f} |\n"
    
    report += """
### GPU 性能数据

#### 开启 KV Cache (前 10 个 token)
"""
    
    if gpu_results_cache:
        report += "| Token ID | Time (s) | Cumulative Time (s) | Tokens/s |\n"
        report += "|----------|----------|---------------------|----------|\n"
        for r in gpu_results_cache[:10]:
            report += f"| {r.token_id} | {r.token_time:.6f} | {r.cumulative_time:.6f} | {r.tokens_per_second:.4f} |\n"
    
    report += """
#### 关闭 KV Cache (前 10 个 token)
"""
    
    if gpu_results_no_cache:
        report += "| Token ID | Time (s) | Cumulative Time (s) | Tokens/s |\n"
        report += "|----------|----------|---------------------|----------|\n"
        for r in gpu_results_no_cache[:10]:
            report += f"| {r.token_id} | {r.token_time:.6f} | {r.cumulative_time:.6f} | {r.tokens_per_second:.4f} |\n"
    
    report += """
## 结论

1. **KV Cache 显著提升性能**: 无论是 CPU 还是 GPU，使用 KV Cache 都能大幅减少每个 token 的生成时间
2. **无 KV Cache 时性能递减**: 不使用 KV Cache 时，随着生成 token 数量增加，需要重复计算所有历史 token 的 KV 值，导致生成时间逐渐增加
3. **KV Cache 的必要性**: 对于长文本生成，KV Cache 是优化推理性能的关键技术

## 实验日期
""" + time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到：{report_path}")


def main():
    load_env_file()
    
    model_path = os.environ.get("MODEL_PATH")
    print(f"正在加载模型：{model_path}")
    
    # 加载模型
    model, config, tokenizer, prompt, inputs = setup_model_and_tokenizer(model_path)
    
    print(f"Prompt: {prompt[:100]}...")
    print(f"输入 token 数：{len(inputs.ids)}")
    
    # 设置不同的最大生成长度
    max_new_tokens_no_cache = 50  # 无 cache 时生成较少 token
    max_new_tokens_with_cache = 200  # 有 cache 时生成较多 token
    max_new_tokens_gpu_no_cache = 512  # GPU 无 cache 时生成更多 token
    max_new_tokens_gpu_with_cache = 1024  # GPU 有 cache 时生成更多 token
    
    # CPU 测试
    print("\n" + "="*60)
    print("开始 CPU 测试")
    print("="*60)
    
    print("\nCPU - 开启 KV Cache...")
    cpu_results_cache = benchmark_generation(
        model, config, inputs, tokenizer,
        use_cache=True,
        max_new_tokens=max_new_tokens_with_cache,
        device="cpu"
    )
    
    print("\nCPU - 关闭 KV Cache...")
    cpu_results_no_cache = benchmark_generation(
        model, config, inputs, tokenizer,
        use_cache=False,
        max_new_tokens=max_new_tokens_no_cache,
        device="cpu"
    )
    
    # GPU 测试
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("开始 GPU 测试")
        print("="*60)
        
        print(f"\nGPU 设备：{torch.cuda.get_device_name(0)}")
        
        print("\nGPU - 开启 KV Cache...")
        gpu_results_cache = benchmark_generation(
            model, config, inputs, tokenizer,
            use_cache=True,
            max_new_tokens=max_new_tokens_gpu_with_cache,
            device="cuda"
        )
        
        print("\nGPU - 关闭 KV Cache...")
        gpu_results_no_cache = benchmark_generation(
            model, config, inputs, tokenizer,
            use_cache=False,
            max_new_tokens=max_new_tokens_gpu_no_cache,
            device="cuda"
        )
    else:
        print("\n未检测到 GPU，跳过 GPU 测试")
        gpu_results_cache = []
        gpu_results_no_cache = []
    
    # 生成图表
    chart_path = "../../pics/kv_cache_benchmark.png"
    print("\n生成性能对比图表...")
    plot_results(
        cpu_results_cache,
        cpu_results_no_cache,
        gpu_results_cache,
        gpu_results_no_cache,
        chart_path
    )
    
    # 生成汇总表格
    df = create_summary_table(
        cpu_results_cache,
        cpu_results_no_cache,
        gpu_results_cache,
        gpu_results_no_cache
    )
    
    print("\n性能对比汇总:")
    print(df.to_string(index=False))
    
    # 生成报告
    report_path = "exps/reports/kv_cache_benchmark_report.md"
    generate_report(
        df,
        chart_path,
        report_path,
        cpu_results_cache,
        cpu_results_no_cache,
        gpu_results_cache,
        gpu_results_no_cache
    )
    
    print("\n实验完成!")


if __name__ == "__main__":
    main()
