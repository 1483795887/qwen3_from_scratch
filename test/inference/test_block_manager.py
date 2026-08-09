from qwen3_from_scratch.inference.sequence import Sequence
from qwen3_from_scratch.inference.block_manager import BlockManager

def make_seq(prompt_len:int = 16)->Sequence:
    return Sequence([0]* prompt_len)

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