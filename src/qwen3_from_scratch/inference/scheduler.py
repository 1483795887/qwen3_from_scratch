from collections import deque
from dataclasses import dataclass
from typing import Callable

from qwen3_from_scratch.inference.block_manager import BlockManager
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerConfig:
    max_num_seqs: int
    max_num_tokens: int
    block_size: int
    max_blocks: int
    enable_prefix_cache: bool = True
    chunked_prefill_size: int = 512


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        check_seq_finish_func: Callable[[Sequence], bool] = lambda seq: False,
    ):
        self.config = config
        self.check_seq_finish_func = check_seq_finish_func
        self.max_num_seqs = config.max_num_seqs
        self.max_num_tokens = config.max_num_tokens
        self.block_size = config.block_size
        self.chunked_prefill_size = config.chunked_prefill_size
        self.waiting: deque[Sequence] = deque()
        self.active: deque[Sequence] = deque()
        self.block_manager = BlockManager(
            num_blocks=config.max_blocks,
            block_size=config.block_size,
            enable_prefix_cache=config.enable_prefix_cache,
        )

    def add_request(self, req: Sequence):
        if (
            len(req.prompts) + req.max_new_tokens
            > self.config.max_blocks * self.block_size
        ):
            return False
        self.waiting.append(req)
        return True

    def schedule(self) -> list[Sequence]:
        # 解码优先：先调度 decode，剩余额度补 prefill 分段
        decode_reqs = self._schedule_decode()
        used_seqs = len(decode_reqs)
        used_tokens = used_seqs  # decode 每条消费 1 token
        prefill_reqs = self._schedule_prefill(
            self.max_num_seqs - used_seqs,
            self.max_num_tokens - used_tokens,
        )
        return decode_reqs + prefill_reqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        budget = min(self.max_num_seqs, self.max_num_tokens)

        prefilling: deque[Sequence] = deque()
        decode_ready: deque[Sequence] = deque()
        for _ in range(len(self.active)):
            seq = self.active.popleft()
            (prefilling if seq.is_prefill else decode_ready).append(seq)

        while decode_ready and len(scheduled) < budget:
            seq = decode_ready.popleft()
            while not self.block_manager.can_allocate(seq, 1):
                if decode_ready:
                    # 抢占最新生成的，浪费计算最少
                    self._preempt(decode_ready.pop())
                else:
                    # 它就是最后一个解码请求，也是最新的，把自己释放了
                    self._preempt(seq)
                    break
            else:
                self.block_manager.allocate(seq, 1)
                seq.num_tokens = 1
                scheduled.append(seq)

        self.active.extend(prefilling)
        self.active.extend(decode_ready)
        self.active.extend(scheduled)
        return scheduled

    def _schedule_prefill(
        self, max_seqs: int, max_tokens: int
    ) -> list[Sequence]:
        scheduled: list[Sequence] = []
        batched_tokens = 0

        # 1. 优先继续 active 里还在 prefill 的序列（FCFS）
        prefilling: deque[Sequence] = deque()
        decode: deque[Sequence] = deque()
        for _ in range(len(self.active)):
            seq = self.active.popleft()
            (prefilling if seq.is_prefill else decode).append(seq)

        prefilling_nums = len(prefilling)
        for _ in range(prefilling_nums):
            if len(scheduled) >= max_seqs or batched_tokens >= max_tokens:
                break
            seq = prefilling.popleft()
            remaining = len(seq.prompts) - seq.cached_len
            chunk = min(
                remaining,
                self.chunked_prefill_size,
                max_tokens - batched_tokens,
            )
            if not self.block_manager.can_allocate(seq, chunk):
                prefilling.append(seq)
                continue
            self.block_manager.allocate(seq, chunk)
            seq.num_tokens = chunk
            scheduled.append(seq)
            batched_tokens += chunk

        # 2. 拉 waiting 新序列
        waiting_nums = len(self.waiting)
        for _ in range(waiting_nums):
            if len(scheduled) >= max_seqs or batched_tokens >= max_tokens:
                break
            seq = self.waiting.popleft()
            if not seq.block_tables:
                # 首次调度：前缀命中则 seed cached_len 并共享块
                self.block_manager.share_prefix(seq)
            remaining = len(seq.prompts) - seq.cached_len
            if remaining <= 0:
                # 前缀全命中：无需 prefill，直接进入 decode 阶段
                seq.status = SequenceStatus.RUNNING
                self.active.append(seq)
                continue
            chunk = min(
                remaining,
                self.chunked_prefill_size,
                max_tokens - batched_tokens,
            )
            if not self.block_manager.can_allocate(seq, chunk):
                self.waiting.append(seq)
                continue
            self.block_manager.allocate(seq, chunk)
            seq.num_tokens = chunk
            seq.status = SequenceStatus.RUNNING
            scheduled.append(seq)
            batched_tokens += chunk

        self.active.extend(decode)
        self.active.extend(prefilling)
        self.active.extend(scheduled)
        return scheduled

    def _preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        seq.cached_len = 0
        seq.num_tokens = 0
        seq.token_ids = seq.prompts.copy()
        seq.last_token_id = -1
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def post_process(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            old_cached_len = seq.cached_len
            seq.cached_len += seq.num_tokens
            # forward 之后注册本步刚写满的块
            self.block_manager.register_full_blocks(seq, old_cached_len)
            if not seq.is_prefill:
                # decode 或 prefill 末分段：append 采样 token
                seq.last_token_id = token_id
                seq.token_ids.append(token_id)
                if self.check_seq_finish_func(seq):
                    seq.status = SequenceStatus.FINISHED
                    if seq in self.active:
                        self.active.remove(seq)
                    self.block_manager.deallocate(seq)
