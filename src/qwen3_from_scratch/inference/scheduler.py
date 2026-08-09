from qwen3_from_scratch.inference.block_manager import BlockManager
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class SchedulerConfig:
    max_num_seqs: int
    max_num_tokens: int
    block_size: int
    max_blocks: int


class Scheduler:
    def __init__(self, config: SchedulerConfig, check_seq_finish_func: Callable[[Sequence], bool] = lambda seq: False):
        self.check_seq_finish_func = check_seq_finish_func
        self.max_num_seqs = config.max_num_seqs
        self.max_num_tokens = config.max_num_tokens
        self.block_size = config.block_size
        self.waiting: deque[Sequence] = deque()
        self.active: deque[Sequence] = deque()
        self.block_manager = BlockManager(num_blocks=config.max_blocks, block_size=config.block_size)

    def add_request(self, req: Sequence):
        if len(req.prompts) > self.max_num_tokens:
            return False
        self.waiting.append(req)
        return True

    def schedule_prefill(self) -> list[Sequence]:
        scheduled_reqs = []
        batched_tokens = 0

        waiting_nums = len(self.waiting)
        # 记录队列中的长度，因为后续会修改队列
        for i in range(waiting_nums):
            seq = self.waiting[0]
            self.waiting.popleft()
            remaining_tokens = self.max_num_tokens - batched_tokens
            if len(seq.prompts) > remaining_tokens:
                # 不满足的放到队尾，不会再次被遍历到
                self.waiting.append(seq)
                continue
            if not seq.block_tables:
                if not self.block_manager.can_allocate(seq):
                    self.waiting.append(seq)
                    continue
                self.block_manager.allocate(seq)
            
            seq.status = SequenceStatus.RUNNING
            self.active.append(seq)
            seq.is_prefill = False
            scheduled_reqs.append(seq)
            batched_tokens += len(seq)
            if (len(scheduled_reqs) >= self.max_num_seqs) or len(self.waiting) == 0:
                break
        return scheduled_reqs

    def schedule_decode(self) -> list[Sequence]:
        scheduled_reqs = []
        waiting_nums = len(self.active)
        for i in range(waiting_nums):
            seq = self.active[0]
            assert seq.block_tables
            if not self.block_manager.can_append(seq):
                # decode 都是申请一个的，如果一个都不能申请后面也都申请不了
                break
            self.block_manager.append_block(seq)
            scheduled_reqs.append(seq)
            if len(scheduled_reqs) >= min(self.max_num_seqs, self.max_num_tokens):
                break
        return scheduled_reqs


    def schedule(self) -> list[Sequence]:
        scheduled_reqs = self.schedule_prefill()

        if scheduled_reqs:
            return scheduled_reqs
        return self.schedule_decode()

    def post_process(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.last_token_id = token_id
            seq.token_ids.append(token_id)
            if self.check_seq_finish_func(seq):
                seq.status = SequenceStatus.FINISHED
                if seq in self.active:
                    self.active.remove(seq)
                self.block_manager.deallocate(seq)
