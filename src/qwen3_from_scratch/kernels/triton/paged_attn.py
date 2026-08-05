import triton
import triton.language as tl
import torch
import math

@triton.jit
def load_paged_memory(
    cache,
    block_tables,
    i_start,
    i_end,
    NUM_HEADS: tl.constexpr,
    PAGE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    cache: 是PagedAttention的K/V缓存, 总体形状为 (NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE_N),
        这里传入的时候已经加上了 head 的偏差
    block_tables: 是每个block的id, 总体形状为 (BATCH, cdiv(max_seq_len, PAGE_BLOCK_SIZE))
        这里传入的时候已经加上了 batch 的偏差
        长度不及 max 的会在最后填充 -1 , 但 i_end 不会加载到那里
    """
    HIDDEN_DIM = NUM_HEADS * HEAD_DIM
    result = tl.zeros((BLOCK_SIZE_N, HEAD_DIM), dtype=cache.dtype.element_ty)
    dim_offsets = tl.arange(0, HEAD_DIM)
    row_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    i_size = i_end - i_start
    # i_start 总是 BLOCK_SIZE_N 的整倍数，从而是 PAGE_BLOCK_SIZE 的整倍数
    for i in tl.range(i_start, i_end, PAGE_BLOCK_SIZE):
        block_idx = i // PAGE_BLOCK_SIZE
        block_id = tl.load(block_tables + block_idx)
        # 通过 i_end 可以保证 block_id > 0, 而且 triton 中无法写 break 和 continue 就不写了
        loaded_rows = i - i_start
        # 从 -loaded_heads 开始加载 PAGE_BLOCK_SIZE'
        # BLOCK_SIZE_N 中加载 [loaded_heads, loaded_heads + PAGE_BLOCK_SIZE)
        # 所以 mask 要把前后的给遮掉
        global_row_offsets = block_id * PAGE_BLOCK_SIZE - loaded_rows + row_offsets[:, None]
        block_data = tl.load(
            cache 
            + global_row_offsets * HIDDEN_DIM
            + dim_offsets[None, :],
            mask=((row_offsets[:, None]) < tl.minimum(i_size, loaded_rows + PAGE_BLOCK_SIZE)) & (row_offsets[:, None] >= loaded_rows),
            other=0.0,
        )
        result += block_data
    return result


@triton.jit
def load_contiguous_memory(
    ptr,
    i_start,
    i_end,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    ptr: 指向 (N, NUM_HEADS, HEAD_DIM) 的数据，传入时已经加上了 head 的偏差
    """
    HIDDEN_DIM = NUM_HEADS * HEAD_DIM
    row_offsets = tl.arange(0, BLOCK_SIZE_N)
    dim_offsets = tl.arange(0, HEAD_DIM)
    mask = row_offsets[:, None] < tl.minimum(i_end - i_start, BLOCK_SIZE_N)
    global_row_offsets = i_start + row_offsets[:, None]
    return tl.load(
        ptr + global_row_offsets * HIDDEN_DIM + dim_offsets[None, :],
        mask=mask,
        other=0.0,
    )

def main():
    pass


if __name__ == '__main__':
    main()
