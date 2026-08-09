import math

from qwen3_from_scratch.inference.sequence import Sequence


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.free_blocks = list(range(num_blocks))
        self.used_blocks: set[int] = set()
        self.block_size = block_size

    def can_allocate(self, seq: Sequence) -> bool:
        return math.ceil(len(seq) / self.block_size) <= len(self.free_blocks)

    def _allocate(self) -> int:
        block_id = self.free_blocks.pop()
        self.used_blocks.add(block_id)
        return block_id

    def _deallocate_one_block(self, block_id: int):
        if block_id in self.used_blocks:
            self.used_blocks.remove(block_id)
        self.free_blocks.append(block_id)

    def allocate(self, seq: Sequence):
        # 外部是单线程调用，不用管并发问题
        assert len(seq.block_tables) == 0, f"{seq.req_id} 发生重复申请"
        for i in range(0, len(seq), self.block_size):
            block_id = self._allocate()
            seq.block_tables.append(block_id)

    def can_append(self, seq: Sequence) -> bool:
        # decode 时使用，追加一个词元能否申请一个新块，如果不需要申请，也算能够申请
        if (len(seq) + 1) % self.block_size != 1:
            return True
        return len(self.free_blocks) > 0

    def append_block(self, seq: Sequence):
        if (len(seq) + 1) % self.block_size != 1:
            return
        seq.block_tables.append(self._allocate())

    def deallocate(self, seq: Sequence):
        for block_id in seq.block_tables:
            self._deallocate_one_block(block_id)
        seq.block_tables.clear()
