import pytest
import torch

# 非 CUDA 环境: 跳过整个模块
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
import triton
import triton.language as tl

from qwen3_from_scratch.kernels.triton.paged_attn import (
    load_paged_memory,
)


@triton.jit
def load_paged_memory_wrapper(
    cache,
    block_tables,
    output,
    i_start,
    i_end,
    NUM_HEADS: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    result = load_paged_memory(
        cache,
        block_tables,
        i_start,
        i_end,
        NUM_HEADS,
        PAGE_BLOCK_SIZE,
        HEAD_DIM,
        BLOCK_SIZE_N,
    )
    offset_rows = tl.arange(0, BLOCK_SIZE_N)
    offset_dim = tl.arange(0, HEAD_DIM)
    tl.store(
        output + offset_rows[:, None] * HEAD_DIM + offset_dim[None, :],
        result,
        mask=(offset_rows[:, None] < (i_end - i_start)),
    )


def _verify_with_shared_wrapper(
    i_start,
    i_end,
    cache,
    block_tables,
    NUM_HEADS,
    PAGE_BLOCK_SIZE,
    HEAD_DIM,
    BLOCK_SIZE_N,
):
    output = torch.empty(BLOCK_SIZE_N, HEAD_DIM, device="cuda")

    grid = (1,)
    load_paged_memory_wrapper[grid](
        cache,
        block_tables,
        output,
        i_start,
        i_end,
        NUM_HEADS,
        PAGE_BLOCK_SIZE,
        HEAD_DIM,
        BLOCK_SIZE_N,
    )
    return output


def test_load_paged_memory_basic():
    NUM_HEADS = 4
    BLOCK_SIZE = 16
    BLOCK_SIZE_DIM = 128
    BLOCK_SIZE_N = 32
    num_pages = 12
    # Cache layout: (num_pages, block_size, num_heads, head_dim)
    cache = torch.randn(
        num_pages, BLOCK_SIZE, NUM_HEADS, BLOCK_SIZE_DIM, device="cuda"
    )
    block_tables = torch.tensor([3, 4, 6], device="cuda", dtype=torch.long)
    i_start = 0
    i_end = 2 * BLOCK_SIZE

    output = _verify_with_shared_wrapper(
        i_start,
        i_end,
        cache,
        block_tables,
        NUM_HEADS,
        BLOCK_SIZE,
        BLOCK_SIZE_DIM,
        BLOCK_SIZE_N,
    )
    expected = torch.cat([cache[3, :, 0, :], cache[4, :, 0, :]], dim=0)
    assert torch.allclose(output, expected)


def test_load_paged_memory_non_aligned():
    """Test when i_start is not aligned to a block boundary.

    i_start = BLOCK_SIZE // 2 means the first token to load is at offset
    BLOCK_SIZE//2 within the first block. The kernel should only return tokens
    in [i_start, i_end), discarding those before i_start.

    Uses a shared-memory wrapper to verify load_paged_memory writes correct data
    into shared memory (as it would inside a real attention kernel).
    """
    NUM_HEADS = 4
    BLOCK_SIZE = 16
    BLOCK_SIZE_DIM = 128
    BLOCK_SIZE_N = 32

    num_pages = 12
    cache = torch.randn(
        num_pages, BLOCK_SIZE, NUM_HEADS, BLOCK_SIZE_DIM, device="cuda"
    )
    # block_tables maps absolute block indices to page IDs.
    block_tables = torch.arange(
        num_pages, device="cuda", dtype=torch.long
    )  # identity mapping

    i_start = 0
    i_end = 3

    output = _verify_with_shared_wrapper(
        i_start,
        i_end,
        cache,
        block_tables,
        NUM_HEADS,
        BLOCK_SIZE,
        BLOCK_SIZE_DIM,
        BLOCK_SIZE_N,
    )

    expected = torch.zeros(BLOCK_SIZE_N, BLOCK_SIZE_DIM, device="cuda")
    expected[i_start:i_end] = cache[0, i_start:i_end, 0, :]

    assert torch.allclose(output, expected, atol=1e-4)


def test_load_paged_memory_non_aligned2():
    """Test when i_start is not aligned to a block boundary.

    i_start = BLOCK_SIZE // 2 means the first token to load is at offset
    BLOCK_SIZE//2 within the first block. The kernel should only return tokens
    in [i_start, i_end), discarding those before i_start.

    Uses a shared-memory wrapper to verify load_paged_memory writes correct data
    into shared memory (as it would inside a real attention kernel).
    """
    NUM_HEADS = 4
    BLOCK_SIZE = 16
    BLOCK_SIZE_DIM = 128
    BLOCK_SIZE_N = 32
    num_pages = 16
    cache = torch.randn(
        num_pages, BLOCK_SIZE, NUM_HEADS, BLOCK_SIZE_DIM, device="cuda"
    )
    # block_tables maps absolute block indices to page IDs.
    block_tables = torch.arange(
        num_pages, device="cuda", dtype=torch.long
    )  # identity mapping

    # i_start = 8 (halfway through block 0), i_end = 48 (start of block 3)
    i_start = BLOCK_SIZE * 4
    i_end = BLOCK_SIZE * 5 + 1

    output = _verify_with_shared_wrapper(
        i_start,
        i_end,
        cache,
        block_tables,
        NUM_HEADS,
        BLOCK_SIZE,
        BLOCK_SIZE_DIM,
        BLOCK_SIZE_N,
    )

    expected = torch.zeros(BLOCK_SIZE_N, BLOCK_SIZE_DIM, device="cuda")
    expected[:BLOCK_SIZE, :] = cache[4, :, 0, :]
    expected[BLOCK_SIZE : BLOCK_SIZE + 1, :] = cache[5, :1, 0, :]

    assert torch.allclose(output, expected, atol=1e-4)
