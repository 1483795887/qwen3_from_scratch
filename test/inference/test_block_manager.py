from qwen3_from_scratch.inference.block_manager import BlockManager
from qwen3_from_scratch.inference.sequence import Sequence


def make_seq(prompt_len: int = 16) -> Sequence:
    return Sequence([0] * prompt_len, max_new_tokens=10)


class TestCanAllocate:
    def test_true_when_enough_free_pages(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(16)
        assert bm.can_allocate(seq, 16)

    def test_false_when_insufficient_free_pages(self):
        bm = BlockManager(num_blocks=2, block_size=16)
        seq = make_seq(33)
        assert not bm.can_allocate(seq, 33)

    def test_true_when_num_tokens_stay_within_existing_blocks(self):
        # 新增 token 仍在已分配块内，无需新块 → 恒可分配
        bm = BlockManager(num_blocks=2, block_size=16)
        seq = make_seq(16)
        bm.allocate(seq, 16)
        seq.cached_len = 16
        assert bm.can_allocate(seq, 1)

    def test_false_when_boundary_cross_needs_block_but_pool_empty(self):
        bm = BlockManager(num_blocks=2, block_size=16)
        seq = make_seq(32)
        bm.allocate(seq, 16)
        seq.cached_len = 16
        bm.allocate(seq, 16)  # 占满 2 块
        seq.cached_len = 32
        assert not bm.can_allocate(seq, 1)


class TestAllocate:
    def test_allocates_exact_blocks_for_num_tokens(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(32)
        bm.allocate(seq, 32)
        assert len(seq.block_tables) == 2
        assert all(isinstance(pid, int) for pid in seq.block_tables)

    def test_allocate_incrementally_appends_only_missing_blocks(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(32)
        bm.allocate(seq, 16)  # 第一段 prefill 16 token
        assert len(seq.block_tables) == 1
        seq.cached_len = 16  # 模拟 post_process 写入后
        bm.allocate(seq, 16)  # 第二段 prefill 16 token
        assert len(seq.block_tables) == 2

    def test_allocate_appends_block_on_decode_boundary_cross(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(16)
        bm.allocate(seq, 16)
        seq.cached_len = 16
        bm.allocate(seq, 1)  # decode 1 token，跨块 → 追加第 2 块
        assert len(seq.block_tables) == 2

    def test_allocate_noop_when_within_existing_blocks(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(16)
        bm.allocate(seq, 16)
        before = list(seq.block_tables)
        seq.cached_len = 15  # 还有 1 token 空间
        bm.allocate(seq, 1)  # 15+1=16，仍在第 1 块内
        assert seq.block_tables == before


class TestDeallocate:
    def test_returns_all_blocks_to_free_pool(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(32)
        bm.allocate(seq, 32)
        allocated = list(seq.block_tables)
        bm.deallocate(seq)
        for pid in allocated:
            assert pid in bm.free_blocks

    def test_clears_seq_block_tables(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(32)
        bm.allocate(seq, 32)
        bm.deallocate(seq)
        assert seq.block_tables == []

    def test_freed_blocks_reusable_by_new_sequence(self):
        bm = BlockManager(num_blocks=2, block_size=16)
        seq_a = make_seq(32)
        bm.allocate(seq_a, 32)
        bm.deallocate(seq_a)
        seq_b = make_seq(32)
        bm.allocate(seq_b, 32)
        assert len(seq_b.block_tables) == 2
