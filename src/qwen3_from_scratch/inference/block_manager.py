from qwen3_from_scratch.inference.sequence import Sequence
import math


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.free_blocks = list(range(num_blocks))
        self.used_blocks: set[int] = set()
        self.block_size = block_size

    def can_allocate(self, seq: Sequence) -> bool:
        return math.ceil(len(seq) / self.block_size) <= len(self.free_blocks)

    def _allocate(self)->int:
        block_id = self.free_blocks.pop()
        self.used_blocks.add(block_id)
        return block_id

    def allocate(self, seq: Sequence):
        # 外部是单线程调用，不用管并发问题
        for i in range(0, len(seq), self.block_size):
            block_id = self._allocate()
            seq.block_tables.append(block_id)
