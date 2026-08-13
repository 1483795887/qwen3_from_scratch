import math
from dataclasses import dataclass

from qwen3_from_scratch.inference.sequence import Sequence

_BASE = 1099511628211  # 大奇数（多项式散列基）
_MASK = (1 << 64) - 1  # 64-bit 掩码


@dataclass
class Block:
    """一个物理 KV 页。

    block_id: 物理页 id。
    ref_count: 引用计数，被多少条序列的 block_tables 引用；归零才可回收。
    token_ids: 写满后的内容（命中时的逐 token 校验依据）；未写满为 None。
    hash_key: 写满后的 64-bit 散列（0 表示未注册进 prefix_map）。
    """

    block_id: int
    ref_count: int = 0
    token_ids: list[int] | None = None
    hash_key: int = 0


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_cache: bool = True,
    ):
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self._blocks: dict[int, Block] = {
            i: Block(block_id=i) for i in range(num_blocks)
        }
        self.free_blocks: list[int] = list(range(num_blocks))
        self.prefix_map: dict[int, Block] = {}

    @staticmethod
    def _num_blocks_needed(token_count: int, block_size: int) -> int:
        return math.ceil(token_count / block_size)

    def _hash_block(self, tokens: list[int]) -> int:
        """64-bit 多项式散列（Rabin-Karp 风格）。"""
        h = 0
        for t in tokens:
            h = (h * _BASE + t) & _MASK
        return h

    def _match_blocks(self, prompts: list[int]) -> list[int]:
        """从 token 0 起逐满块匹配，遇第一个 miss 停；返回命中的物理块 id。"""
        matched: list[int] = []
        for i in range(0, len(prompts), self.block_size):
            block_tokens = prompts[i : i + self.block_size]
            if len(block_tokens) < self.block_size:
                break  # 不满块不参与匹配
            key = self._hash_block(block_tokens)
            block = self.prefix_map.get(key)
            if block is None or block.token_ids != block_tokens:
                break  # miss（含散列碰撞误判，token 比对兜底）
            matched.append(block.block_id)
        return matched

    def match_prefix(self, prompts: list[int]) -> list[int]:
        """对外前缀匹配（纯查询，不改变任何状态）：返回共享前缀的物理块 id。"""
        if not self.enable_prefix_cache:
            return []
        return self._match_blocks(prompts)

    def share_prefix(self, seq: Sequence) -> int:
        """首次调度时调用：前缀命中则复用共享块（ref_count + 1）并 seed cached_len。

        返回匹配到的 token 数（matched_len）。
        """
        if not self.enable_prefix_cache:
            return 0
        for block_id in self._match_blocks(seq.prompts):
            seq.block_tables.append(block_id)
            self._blocks[block_id].ref_count += 1
        seq.cached_len = len(seq.block_tables) * self.block_size
        return seq.cached_len

    def _allocate_one(self) -> int:
        block_id = self.free_blocks.pop()
        self._blocks[block_id].ref_count = 1
        return block_id

    def can_allocate(self, seq: Sequence, num_tokens: int) -> bool:
        """本步写入 num_tokens 个 token，块数是否足够（纯增量）。"""
        target = self._num_blocks_needed(
            seq.cached_len + num_tokens, self.block_size
        )
        return target - len(seq.block_tables) <= len(self.free_blocks)

    def allocate(self, seq: Sequence, num_tokens: int):
        """确保 block_tables 能容纳 cached_len + num_tokens 个 token（纯增量）。

        只补齐缺失的块，已分配 / 共享的块不动。
        """
        target = self._num_blocks_needed(
            seq.cached_len + num_tokens, self.block_size
        )
        for _ in range(target - len(seq.block_tables)):
            seq.block_tables.append(self._allocate_one())

    def register_full_blocks(self, seq: Sequence, old_cached_len: int):
        """注册本步刚写满的块进 prefix_map（forward 之后调用）。

        块 i 写满 ⟺ (i+1)*block_size <= seq.cached_len（写后）。
        old_cached_len 为写前 cached_len。
        """
        if not self.enable_prefix_cache:
            return
        old_full = old_cached_len // self.block_size
        new_full = seq.cached_len // self.block_size
        for i in range(old_full, new_full):
            block_tokens = seq.token_ids[
                i * self.block_size : (i + 1) * self.block_size
            ]
            block_id = seq.block_tables[i]
            key = self._hash_block(block_tokens)
            block = self._blocks[block_id]
            block.token_ids = block_tokens
            block.hash_key = key
            self.prefix_map[key] = block

    def deallocate(self, seq: Sequence):
        """释放 seq 持有的块：逐块 ref_count 递减，归零才回收。"""
        for block_id in seq.block_tables:
            block = self._blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_blocks.append(block_id)
                if self.prefix_map.get(block.hash_key) is block:
                    del self.prefix_map[block.hash_key]
        seq.block_tables.clear()
