from qwen3_from_scratch.inference.block_manager import BlockManager
from qwen3_from_scratch.inference.sequence import Sequence


def make_seq(prompt_len: int = 16) -> Sequence:
    return Sequence([0] * prompt_len)


class TestCanAllocate:
    def test_true_when_enough_free_pages(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(16)
        assert bm.can_allocate(seq)

    def test_false_when_insufficient_free_pages(self):
        bm = BlockManager(num_blocks=2, block_size=16)
        seq = make_seq(33)
        assert not bm.can_allocate(seq)


class TestAllocate:
    def test_writes_page_ids_to_seq_block_tables(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(prompt_len=32)
        bm.allocate(seq)
        assert len(seq.block_tables) == 2
        assert all(isinstance(pid, int) for pid in seq.block_tables)

    def test_reflects_current_free_pool_after_allocation(self):
        """已分配页面后，can_allocate 反映剩余空闲页。"""
        bm = BlockManager(num_blocks=3, block_size=16)
        seq1 = make_seq(prompt_len=31)  # 占 2 页
        bm.allocate(seq1)
        seq2 = make_seq(prompt_len=16)  # 需 1 页
        # 剩余 1 页，恰好够 1 页
        assert bm.can_allocate(seq2)

    def test_can_allocate_return_false_after_pool_exhausted(self):
        """空闲页耗尽后 can_allocate 返回 False。"""
        bm = BlockManager(num_blocks=2, block_size=16)
        seq1 = make_seq(prompt_len=32)
        bm.allocate(seq1)  # 占满 2 页
        seq2 = make_seq(prompt_len=16)
        assert not bm.can_allocate(seq2)

    def test_multiple_sequences_get_disjoint_pages(self):
        """多个 seq 分配的页面 ID 不重叠。"""
        bm = BlockManager(num_blocks=10, block_size=16)
        seq1 = make_seq(prompt_len=32)
        seq2 = make_seq(prompt_len=16)
        bm.allocate(seq1)
        bm.allocate(seq2)
        assert set(seq1.block_tables).isdisjoint(set(seq2.block_tables))


class TestCanAppend:
    def test_true_when_free_pages_available(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(prompt_len=16)
        bm.allocate(seq)  # 占 1 页，剩 9 页
        assert bm.can_append(seq)

    def test_true_when_no_need_allocate_new_page(self):
        bm = BlockManager(num_blocks=2, block_size=16)
        seq = make_seq(17)
        bm.allocate(seq)
        assert bm.can_append(seq)

    def test_false_when_no_free_pages_available(self):
        bm = BlockManager(num_blocks=2, block_size=16)
        seq = make_seq(32)
        bm.allocate(seq)
        assert not bm.can_append(seq)


class TestAppendBlock:
    def test_appends_one_page_to_block_tables(self):
        """追加后 block_tables 长度 +1。"""
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(prompt_len=16)
        bm.allocate(seq)
        before = len(seq.block_tables)
        bm.append_block(seq)
        assert len(seq.block_tables) == before + 1

    def test_reflects_state_after_append(self):
        """append_block 占用最后一页后 can_append 变 False。"""
        bm = BlockManager(num_blocks=3, block_size=16)
        seq = make_seq(prompt_len=16)
        seq2 = make_seq(16)
        bm.allocate(seq)
        bm.allocate(seq2)
        assert bm.can_append(seq)
        bm.append_block(seq)
        assert not bm.can_append(seq2)


class TestDeallocate:
    def test_returns_all_pages_to_free_pool(self):
        """释放后所有原页面回到 free_blocks。"""
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(prompt_len=32)
        bm.allocate(seq)
        allocated = list(seq.block_tables)
        bm.deallocate(seq)
        for pid in allocated:
            assert pid in bm.free_blocks

    def test_reflects_state_after_free(self):
        """free 回收页面后 can_append 变 True。"""
        bm = BlockManager(num_blocks=1, block_size=16)
        seq = make_seq(prompt_len=16)
        bm.allocate(seq)
        assert bm.can_append(seq) is False
        bm.deallocate(seq)
        assert bm.can_append(seq) is True

    def test_clears_seq_block_tables(self):
        """释放后 seq.block_tables 被清空。"""
        bm = BlockManager(num_blocks=10, block_size=16)
        seq = make_seq(prompt_len=32)
        bm.allocate(seq)
        bm.deallocate(seq)
        assert seq.block_tables == []

    def test_freed_pages_reusable_by_new_sequence(self):
        """释放的页面可被新 seq 重新分配。"""
        bm = BlockManager(num_blocks=2, block_size=16)
        seq_a = make_seq(prompt_len=32)  # 占满 2 页
        bm.allocate(seq_a)
        bm.deallocate(seq_a)
        seq_b = make_seq(prompt_len=32)  # 再分配 2 页
        bm.allocate(seq_b)
        assert len(seq_b.block_tables) == 2
