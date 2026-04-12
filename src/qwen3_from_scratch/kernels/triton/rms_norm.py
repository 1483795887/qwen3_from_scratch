from math import prod
import triton
from triton import language as tl
import torch


@triton.jit
def rms_norm_forward_kernel(x, D, output, BLOCK_SIZE: tl.constexpr, eps: float = 1e-5):
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)

    offsets = pid * D + tid
    sum_sq = 0.0
    for block_offset in tl.range(0, D, BLOCK_SIZE):
        items = tl.load(
            x + offsets + block_offset, block_offset + tid < D, 0.0
        )
        items = items * items
        sum_sq += tl.sum(items)
    scale = tl.rsqrt(sum_sq / D + eps)
    for block_offset in tl.range(0, D, BLOCK_SIZE):
        items = tl.load(
            x + offsets + block_offset, block_offset + tid < D, 0.0
        )
        items = items * scale
        tl.store(
            output + offsets + block_offset, items, block_offset + tid < D
        )


def rms_norm_forward(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5):
    assert x.is_contiguous()
    assert gamma.is_contiguous()
    shape = x.shape
    assert shape[-1] == gamma.shape[-1]
    seq_len = prod(shape[:-1])
    BLOCK_SIZE = 128

    grid = (seq_len,)
    output = torch.zeros_like(x)
    rms_norm_forward_kernel[grid](x, shape[-1], output, BLOCK_SIZE, eps)
    return output
