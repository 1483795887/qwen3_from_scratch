import triton
from triton import language as tl
import torch


def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE": 32, "num_warps": 1}),
        triton.Config({"BLOCK_SIZE": 64, "num_warps": 1}),
        triton.Config({"BLOCK_SIZE": 64, "num_warps": 2}),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 1}),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 2}),
        triton.Config({"BLOCK_SIZE": 128, "num_warps": 4}),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 1}),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 2}),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 4}),
        triton.Config({"BLOCK_SIZE": 256, "num_warps": 8}),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 1}),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 2}),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 4}),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 8}),
        triton.Config({"BLOCK_SIZE": 512, "num_warps": 16}),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 1}),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 2}),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 4}),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 8}),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 16}),
        triton.Config({"BLOCK_SIZE": 1024, "num_warps": 32}),
    ]


# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=["D"],
# )
@triton.jit
def rms_norm_forward_kernel(
    x,
    gamma,
    output,
    stride,
    eps: float,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)

    x += pid * stride
    output += pid * stride

    sum_sq = tl.zeros((BLOCK_SIZE,), tl.float32)
    for block_offset in tl.range(0, D, BLOCK_SIZE):
        cols = block_offset + tid
        mask = cols < D
        items = tl.load(x + cols, mask, 0.0).to(tl.float32)
        sum_sq += items * items
    var = tl.sum(sum_sq)
    scale = tl.rsqrt(var / D + eps)
    for block_offset in tl.range(0, D, BLOCK_SIZE):
        cols = block_offset + tid
        mask = cols < D
        items = tl.load(x + cols, mask, 0.0)
        gamma_items = tl.load(gamma + cols, mask, 0.0)
        tl.store(output + cols, items * scale * gamma_items, mask)


@triton.jit
def rms_norm_forward_kernel_one_step(
    x,
    gamma,
    output,
    stride,
    eps: float,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE >= D)
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)

    x += pid * stride
    output += pid * stride

    cols = tid
    mask = cols < D
    x_items = tl.load(x + cols, mask, 0.0)
    sum_sq = x_items * x_items
    var = tl.sum(sum_sq)
    scale = tl.rsqrt(var / D + eps)
    gamma_items = tl.load(gamma + cols, mask, 0.0)
    tl.store(output + cols, x_items * scale * gamma_items, mask)


def rms_norm_forward(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5):
    assert x.stride(-1) == 1
    assert gamma.stride(-1) == 1
    shape = x.shape
    D = shape[-1]
    assert D == gamma.shape[-1]
    seq_len = x.numel() // D

    grid = (seq_len,)
    output = torch.empty_like(x)
    # NEXT_POWER_OF_D = triton.next_power_of_2(D)
    # NEXT_POWER_OF_D = 128
    # kernel_func = rms_norm_forward_kernel
    # if (NEXT_POWER_OF_D > MAX_FUSED_SIZE):
    #     BLOCK_SIZE = MAX_FUSED_SIZE
    # else:
    kernel_func = rms_norm_forward_kernel_one_step
    kernel_func[grid](
        x,
        gamma,
        output,
        D,
        eps,
        D=D,
        BLOCK_SIZE=triton.next_power_of_2(D),
    )
    return output
