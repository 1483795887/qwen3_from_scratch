from qwen3_from_scratch.inference.block_manager import BlockManager
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus
from collections import deque
from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    max_num_seqs: int
    max_num_tokens: int
    block_size: int
    max_blocks: int


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_tokens = config.max_num_tokens
        self.block_size = config.block_size
        self.waiting: deque[Sequence] = deque()
        self.active: deque[Sequence] = deque()
        self.block_manager = BlockManager(num_blocks=config.max_blocks, block_size=config.block_size)

    def add_request(self, req: Sequence):
        self.waiting.append(req)

    def schedule(self) -> list[Sequence]:
        return []

    def post_process(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        pass
