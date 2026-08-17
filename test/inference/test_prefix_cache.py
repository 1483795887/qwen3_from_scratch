from qwen3_from_scratch.inference.block_manager import BlockManager
from qwen3_from_scratch.inference.sequence import Sequence


def make_bm(
    num_blocks: int = 10,
    block_size: int = 16,
    enable_prefix_cache: bool = True,
) -> BlockManager:
    return BlockManager(
        num_blocks=num_blocks,
        block_size=block_size,
        enable_prefix_cache=enable_prefix_cache,
    )


def fill_and_register(bm: BlockManager, seq: Sequence) -> None:
    """模拟一次完整 prefill：分配整条 prompt 的块，写入后注册满块。"""
    old_cached_len = seq.cached_len
    bm.allocate(seq, len(seq.prompts) - seq.cached_len)
    seq.cached_len = len(seq.prompts)
    bm.register_full_blocks(seq, old_cached_len)


class TestMatchPrefix:
    def test_returns_shared_block_ids_for_common_prefix(self):
        bm = make_bm()
        prompt = list(range(32))  # 2 个满块
        a = Sequence(prompt, max_new_tokens=10)
        fill_and_register(bm, a)

        b = Sequence(prompt, max_new_tokens=10)
        assert bm.match_prefix(b.prompts) == a.block_tables

    def test_stops_at_first_miss(self):
        bm = make_bm()
        a = Sequence(list(range(32)), max_new_tokens=10)
        fill_and_register(bm, a)

        # 第一块相同 [0..15]，第二块 [16..30, 999] 不同
        b = Sequence(
            list(range(16)) + list(range(16, 31)) + [999], max_new_tokens=10
        )
        assert bm.match_prefix(b.prompts) == [a.block_tables[0]]

    def test_ignores_partial_last_block(self):
        bm = make_bm()
        a = Sequence(list(range(16)), max_new_tokens=10)
        fill_and_register(bm, a)

        # 前缀 = 满块 [0..15] + 不满尾巴 [1, 2, 3]
        b = Sequence(list(range(16)) + [1, 2, 3], max_new_tokens=10)
        assert bm.match_prefix(b.prompts) == [a.block_tables[0]]

    def test_empty_when_nothing_registered(self):
        bm = make_bm()
        a = Sequence(list(range(32)), max_new_tokens=10)
        bm.allocate(a, 32)  # 分配但未注册
        assert bm.match_prefix(a.prompts) == []


class TestSharePrefix:
    def test_shares_prefix_and_seeds_cached_len(self):
        bm = make_bm()
        prompt = list(range(32))
        a = Sequence(prompt, max_new_tokens=10)
        fill_and_register(bm, a)

        b = Sequence(prompt + [100, 101], max_new_tokens=10)
        matched_len = bm.share_prefix(b)
        assert matched_len == 32
        assert b.cached_len == 32  # 前缀命中，seed 到 32
        assert b.block_tables == a.block_tables  # 共享前 2 块
        bm.allocate(b, 2)  # 只 prefill 尾巴 2 token
        assert len(b.block_tables) == 3  # 2 共享 + 1 新分配

    def test_full_hit_allocates_no_extra_block(self):
        bm = make_bm()
        prompt = list(range(16))
        a = Sequence(prompt, max_new_tokens=10)
        fill_and_register(bm, a)

        b = Sequence(prompt, max_new_tokens=10)
        matched_len = bm.share_prefix(b)
        assert matched_len == 16
        assert b.block_tables == a.block_tables
        bm.allocate(b, 0)  # 无剩余 prompt，无需新块
        assert b.block_tables == a.block_tables

    def test_disabled_shares_nothing(self):
        bm = make_bm(enable_prefix_cache=False)
        a = Sequence(list(range(16)), max_new_tokens=10)
        fill_and_register(bm, a)

        b = Sequence(list(range(16)), max_new_tokens=10)
        assert bm.share_prefix(b) == 0
        assert b.cached_len == 0
        bm.allocate(b, 16)
        assert b.block_tables != a.block_tables


class TestRefCount:
    def test_deallocate_keeps_shared_block_alive(self):
        bm = make_bm()
        prompt = list(range(16))
        a = Sequence(prompt, max_new_tokens=10)
        fill_and_register(bm, a)

        b = Sequence(prompt, max_new_tokens=10)
        assert bm.share_prefix(b) == 16  # 全命中，共享 a 的块
        shared_id = a.block_tables[0]
        assert b.block_tables == [shared_id]

        bm.deallocate(a)  # a 释放，但 b 仍引用
        assert shared_id not in bm.free_blocks
        assert a.block_tables == []

        bm.deallocate(b)  # 最后一个引用释放，才回收
        assert shared_id in bm.free_blocks

    def test_deallocate_removes_prefix_map_entry(self):
        bm = make_bm()
        a = Sequence(list(range(16)), max_new_tokens=10)
        fill_and_register(bm, a)
        block_id = a.block_tables[0]
        assert block_id in [blk.block_id for blk in bm.prefix_map.values()]

        bm.deallocate(a)
        assert block_id not in [blk.block_id for blk in bm.prefix_map.values()]


class TestRegisterFullBlocks:
    def test_registered_blocks_become_matchable(self):
        bm = make_bm()
        a = Sequence(list(range(32)), max_new_tokens=10)
        bm.allocate(a, 32)
        # 未注册前，另一序列匹配不到
        assert bm.match_prefix(list(range(32))) == []

        a.cached_len = 32
        bm.register_full_blocks(a, 0)
        assert bm.match_prefix(list(range(32))) == a.block_tables
