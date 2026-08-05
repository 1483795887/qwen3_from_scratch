import pytest
import torch

# 非 CUDA 环境: 跳过整个模块
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
import triton
import triton.language as tl

from qwen3_from_scratch.kernels.triton.paged_attn import (
    load_contiguous_memory,
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




@triton.jit
def load_contiguous_memory_wrapper(
    ptr,
    output,
    i_start,
    i_end,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    result = load_contiguous_memory(
        ptr,
        i_start,
        i_end,
        NUM_HEADS,
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


def _verify_contiguous_wrapper(
    i_start,
    i_end,
    ptr,
    NUM_HEADS,
    HEAD_DIM,
    BLOCK_SIZE_N,
):
    output = torch.zeros(BLOCK_SIZE_N, HEAD_DIM, device="cuda")

    grid = (1,)
    load_contiguous_memory_wrapper[grid](
        ptr,
        output,
        i_start,
        i_end,
        NUM_HEADS,
        HEAD_DIM,
        BLOCK_SIZE_N,
    )
    return output


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
    output = torch.zeros(BLOCK_SIZE_N, HEAD_DIM, device="cuda")

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


def test_load_contiguous_memory_basic():
    """load_contiguous_memory 基础: 从头加载一个完整的 BLOCK_SIZE_N 块。"""
    NUM_HEADS = 4
    HEAD_DIM = 128
    BLOCK_SIZE_N = 32
    N = 64
    ptr = torch.randn(N, NUM_HEADS, HEAD_DIM, device="cuda")

    output = _verify_contiguous_wrapper(
        0, BLOCK_SIZE_N, ptr, NUM_HEADS, HEAD_DIM, BLOCK_SIZE_N,
    )
    expected = ptr[:BLOCK_SIZE_N, 0, :]
    assert torch.allclose(output, expected)


def test_load_contiguous_memory_non_aligned():
    """load_contiguous_memory 非对齐 i_start: 从偏移 8 开始加载。"""
    NUM_HEADS = 4
    HEAD_DIM = 128
    BLOCK_SIZE_N = 32
    N = 64
    ptr = torch.randn(N, NUM_HEADS, HEAD_DIM, device="cuda")

    output = _verify_contiguous_wrapper(
        8, 40, ptr, NUM_HEADS, HEAD_DIM, BLOCK_SIZE_N,
    )
    expected = ptr[8:40, 0, :]
    assert torch.allclose(output, expected)


def test_load_contiguous_memory_partial_end():
    """load_contiguous_memory 部分尾块: i_end - i_start < BLOCK_SIZE_N, 尾部补零。"""
    NUM_HEADS = 4
    HEAD_DIM = 128
    BLOCK_SIZE_N = 32
    N = 64
    ptr = torch.randn(N, NUM_HEADS, HEAD_DIM, device="cuda")

    output = _verify_contiguous_wrapper(
        0, 17, ptr, NUM_HEADS, HEAD_DIM, BLOCK_SIZE_N,
    )
    expected = torch.zeros(BLOCK_SIZE_N, HEAD_DIM, device="cuda")
    expected[:17] = ptr[:17, 0, :]
    assert torch.allclose(output, expected)


def test_load_paged_memory_ignores_block_table_padding():
    """不变量: block_tables 末尾的 -1 padding 永远不会被加载。

    i_end 覆盖 2 个有效页, block_tables 中第 3 位起是 -1 占位。
    只要 i_end 不越界, 加载不会踩到 -1, 输出无 NaN/inf。
    """
    NUM_HEADS = 4
    BLOCK_SIZE = 16
    HEAD_DIM = 128
    BLOCK_SIZE_N = 32
    num_pages = 8
    cache = torch.randn(
        num_pages, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, device="cuda"
    )
    # 前 2 个有效页 + 末尾 -1 padding, 模拟 (B, max_blocks) 表的空槽位
    block_tables = torch.tensor(
        [3, 4, -1, -1, -1, -1], device="cuda", dtype=torch.long
    )
    i_start = 0
    i_end = 2 * BLOCK_SIZE

    output = _verify_with_shared_wrapper(
        i_start,
        i_end,
        cache,
        block_tables,
        NUM_HEADS,
        BLOCK_SIZE,
        HEAD_DIM,
        BLOCK_SIZE_N,
    )
    expected = torch.cat([cache[3, :, 0, :], cache[4, :, 0, :]], dim=0)
    assert torch.allclose(output, expected, atol=1e-4)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
