import torch
import triton
import triton.language as tl
from qwen3_from_scratch.kernels.triton.gemm import (
    ActivationType,
    gemm_kernel_core,
)


@triton.jit
def swiglu(
    x,
    up_proj_weight,
    gate_proj_weight,
    down_proj_weight,
    output,
    N,
    D: tl.constexpr,
    D1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D1: tl.constexpr,
    BLOCK_SIZE_REDUCE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    n_id = tl.program_id(0)
    d_id = tl.program_id(1)
    b_id = tl.program_id(2)
    x_ptr = tl.make_block_ptr(
        x + b_id * N * D,
        (N, D),
        (D, 1),
        (n_id * BLOCK_SIZE_N, 0),
        (BLOCK_SIZE_N, BLOCK_SIZE_REDUCE),
        (1, 0),
    )
    up_proj_ptr = tl.make_block_ptr(
        up_proj_weight,
        (D1, D),
        (D, 1),
        (0, 0),
        (BLOCK_SIZE_D1, BLOCK_SIZE_REDUCE),
        (0, 0),
    )
    gate_proj_ptr = tl.make_block_ptr(
        gate_proj_weight,
        (D1, D),
        (D, 1),
        (0, 0),
        (BLOCK_SIZE_D1, BLOCK_SIZE_REDUCE),
        (0, 0),
    )
    down_proj_ptr = tl.make_block_ptr(
        down_proj_weight,
        (D, D1),
        (D1, 1),
        (d_id * BLOCK_SIZE_D, 0),
        (BLOCK_SIZE_D, BLOCK_SIZE_D1),
        (0, 0),
    )
    output_ptr = tl.make_block_ptr(
        output + b_id * N * D,
        (N, D),
        (D, 1),
        (n_id * BLOCK_SIZE_N, d_id * BLOCK_SIZE_D),
        (BLOCK_SIZE_N, BLOCK_SIZE_D),
        (1, 0),
    )

    data_x = tl.load(x_ptr, bound_check=(0,1))
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D1), dtype=tl.float32)
    for k in tl.arange(0, D1, BLOCK_SIZE_D1):
        pass


def swiglu_feedback(
    x: torch.Tensor,
    up_proj_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
):
    activation_fc = ActivationType.SILU
    B, N, D = x.shape
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D1 = 32  # 用于 D1 循环
    BLOCK_SIZE_REDUCE = triton.next_power_of_2(max(D, 16))  # BD * DD1 的乘法
    BLOCK_SIZE_D = 32  # 用于 输出
    D1, _ = up_proj_weight.shape
    assert up_proj_weight.shape == (D1, D)
    assert gate_proj_weight.shape == (D1, D)
    assert down_proj_weight.shape == (D, D1)
    # 简单起见要求三个都连续，也是应该的
    assert x.is_contiguous()
    assert up_proj_weight.is_contiguous()
    assert gate_proj_weight.is_contiguous()
    assert down_proj_weight.is_contiguous()


if __name__ == "__main__":
    N = 16
    D = 32
    D1 = D * 3
    x = torch.rand(N, D, dtype=torch.float32)
    up_proj = torch.nn.Linear(D, D1)
    gate_proj = torch.nn.Linear(D, D1)
    down_proj = torch.nn.Linear(D1, D)

    # ===  (AB)C
    x1 = up_proj(x)
    x2 = torch.nn.functional.silu(gate_proj(x))
    x3 = down_proj(x1 * x2)
    # === A(BC)
    up_weights = up_proj.weight
    gate_weights = gate_proj.weight
    down_weights = down_proj.weight
    fused_weights1 = torch.matmul(up_weights.T, down_weights.T)
    fused_weights2 = torch.matmul(gate_weights.T, down_weights.T)

    x4 = torch.matmul(x, fused_weights1.T)
    x5 = torch.nn.functional.silu(torch.matmul(x, fused_weights2.T))
    x6 = x4 * x5

    print(x3.shape)
    print(x6.shape)
    diff = (x3 - x6).abs()
    print(diff.max())
