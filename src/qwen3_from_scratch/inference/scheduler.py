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

    def _drain_active(self) -> tuple[deque[Sequence], deque[Sequence]]:
        """把 active 按阶段分成两堆：decode 就绪 / 仍在 prefill。"""
        decode_ready: deque[Sequence] = deque()
        prefilling: deque[Sequence] = deque()
        for _ in range(len(self.active)):
            seq = self.active.popleft()
            (prefilling if seq.is_prefill else decode_ready).append(seq)
        return decode_ready, prefilling

    def _restore_active(self, *groups: deque[Sequence]):
        """按给定顺序把各堆序列放回 active（顺序即下轮 FCFS 优先级）。"""
        for group in groups:
            self.active.extend(group)

    def _schedule_decode(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        budget = min(self.max_num_seqs, self.max_num_tokens)

        decode_ready, prefilling = self._drain_active()

        while decode_ready and len(scheduled) < budget:
            seq = decode_ready.popleft()
            if self._allocate_decode_slot(seq, decode_ready):
                scheduled.append(seq)

        self._restore_active(prefilling, decode_ready, scheduled)
        return scheduled

    def _allocate_decode_slot(
        self, seq: Sequence, decode_ready: deque[Sequence]
    ) -> bool:
        """给 seq 分配 1 个 decode token；放不下时抢占最新 decode 让位。

        返回是否成功调度（抢占 seq 自身时返回 False）。
        """
        while not self.block_manager.can_allocate(seq, 1):
            if decode_ready:
                # 抢占最新生成的，浪费计算最少
                self._preempt(decode_ready.pop())
            else:
                # 它就是最后一个解码请求，也是最新的，把自己释放了
                self._preempt(seq)
                return False
        self.block_manager.allocate(seq, 1)
        seq.num_tokens = 1
        return True

    def _schedule_prefill(
        self, max_seqs: int, max_tokens: int
    ) -> list[Sequence]:
        scheduled: list[Sequence] = []
        batched_tokens = 0

        decode_ready, prefilling = self._drain_active()

        # 1. 优先继续 active 里还在 prefill 的序列（FCFS）
        batched_tokens = self._resume_prefilling(
            prefilling, scheduled, max_seqs, max_tokens, batched_tokens
        )
        # 2. 拉 waiting 新序列
        batched_tokens = self._admit_waiting(
            scheduled, max_seqs, max_tokens, batched_tokens
        )

        self._restore_active(decode_ready, prefilling, scheduled)
        return scheduled

    def _resume_prefilling(
        self,
        prefilling: deque[Sequence],
        scheduled: list[Sequence],
        max_seqs: int,
        max_tokens: int,
        batched_tokens: int,
    ) -> int:
        prefilling_nums = len(prefilling)
        for _ in range(prefilling_nums):
            if len(scheduled) >= max_seqs or batched_tokens >= max_tokens:
                break
            seq = prefilling.popleft()
            chunk = self._try_allocate_prefill_chunk(
                seq, max_tokens - batched_tokens
            )
            if chunk is None:
                prefilling.append(seq)
                continue
            scheduled.append(seq)
            batched_tokens += chunk
        return batched_tokens

    def _admit_waiting(
        self,
        scheduled: list[Sequence],
        max_seqs: int,
        max_tokens: int,
        batched_tokens: int,
    ) -> int:
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
            chunk = self._try_allocate_prefill_chunk(
                seq, max_tokens - batched_tokens
            )
            if chunk is None:
                self.waiting.append(seq)
                continue
            seq.status = SequenceStatus.RUNNING
            scheduled.append(seq)
            batched_tokens += chunk
        return batched_tokens

    def _try_allocate_prefill_chunk(
        self, seq: Sequence, tokens_budget: int
    ) -> int | None:
        """尝试给 seq 分段分配 prefill token：预算内取 chunk，块不足返回 None。

        调用方负责：分配成功后的记账（scheduled/status/batched_tokens），
        分配失败时的归属（回 prefill 队列 / waiting）。
        """
        remaining = len(seq.prompts) - seq.cached_len
        chunk = min(remaining, self.chunked_prefill_size, tokens_budget)
        if not self.block_manager.can_allocate(seq, chunk):
            return None
        self.block_manager.allocate(seq, chunk)
        seq.num_tokens = chunk
        return chunk

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
