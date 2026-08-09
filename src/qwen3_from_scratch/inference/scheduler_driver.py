from collections.abc import Callable

from qwen3_from_scratch.inference.scheduler import Scheduler
from qwen3_from_scratch.inference.sequence import Sequence


class SchedulerDriver:
    """共享调度驱动：统一「入队 → 调度 → worker 推理 → 回填」的循环主体。

    服务路径（LLMEngine）与同步路径（SyncEngine）共用同一份调度循环，
    差别仅在 `worker_forward`：服务路径传 mp.Queue 往返，同步路径传
    `ModelWorker.forward` 直调。

    `worker_forward` 约定：接受 `list[Sequence]`，返回 `list[int]`
    （每个 seq 一个新 token）。与 `ModelWorker` / worker 进程的接口一致。
    """

    def __init__(
        self,
        scheduler: Scheduler,
        worker_forward: Callable[[list[Sequence]], list[int]],
    ):
        self.scheduler = scheduler
        self.worker_forward = worker_forward

    def add_request(self, seq: Sequence) -> bool:
        """把一个新序列加入调度等待队列。"""
        return self.scheduler.add_request(seq)

    def step(self, new_seqs: list[Sequence]) -> list[Sequence]:
        """执行一轮调度：入队新序列 → 调度 → worker 推理 → 回填 token。

        返回本轮被调度的序列（已 post_process，`last_token_id` /
        `status` / `cached_len` 已更新）。没有可调度的序列时返回空列表。
        """
        for seq in new_seqs:
            self.scheduler.add_request(seq)
        planned = self.scheduler.schedule()
        if not planned:
            return []
        token_ids = self.worker_forward(planned)
        self.scheduler.post_process(planned, token_ids)
        return planned