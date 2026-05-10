import triton
import torch
import time

from qwen3_from_scratch.factory import ComponentFactory
from qwen3_from_scratch.factory.config import ModelConfig

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def benchmark():
    D = 1024
    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 8
    dtype = torch.float16
    seq_len = 1024
    batch_size = 2
    q = torch.randn((batch_size, seq_len, num_q_heads, head_dim), dtype=dtype, device=DEVICE).transpose(1,2)
    k = torch.randn((batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype, device=DEVICE).transpose(1,2)
    v = torch.randn((batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype, device=DEVICE).transpose(1,2)

    config = ModelConfig(
        hidden_size=D,
        head_dim=head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        norm_params={"eps": 1e-5},
    )

    print(f"Shape: B={batch_size}, N={seq_len}, H_q={num_q_heads}, H_kv={num_kv_heads}, D={head_dim}, dtype={dtype}")
    print("="*60)

    # Test correctness first with a single run
    print("Testing correctness...")
    base_op = ComponentFactory.create("attn", config=config, component_impl="base").to(DEVICE)
    flash_op = ComponentFactory.create("attn", config=config, component_impl="my_op_flash").to(DEVICE)

    base_out = base_op(q, k, v)
    torch.cuda.synchronize()
    flash_out = flash_op(q, k, v)
    torch.cuda.synchronize()

    max_diff = (base_out - flash_out).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    if max_diff < 1e-2:
        print("✓ Outputs match!")
    else:
        print("✗ Outputs differ significantly!")

    print("="*60)

    # Benchmark
    for provider in ["base", "my_op_flash"]:
        attn_op = ComponentFactory.create(
            "attn",
            config=config,
            component_impl=provider,
        ).to(DEVICE)

        # Warmup
        for _ in range(3):
            _ = attn_op(q, k, v)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        # for _ in range(20):
        _ = attn_op(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / 20 * 1000  # ms
        print(f"{provider}: {avg_time:.4f} ms")

    print("="*60)

if __name__ == "__main__":
    benchmark()
