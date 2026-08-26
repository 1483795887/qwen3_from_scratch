import time
from dataclasses import dataclass

from .entities import EngineStepOutput


@dataclass
class RequestMetricRecord:
    added_time: float
    num_prompts: int
    num_output_tokens: int = 0
    last_time: float | None = None
    finished: bool = False


class Metric:
    def __init__(self):
        self.metric_records: dict[str, RequestMetricRecord] = {}

        # 原始数据
        self._ttft_samples: list[float] = []
        self._total_output_tokens: int = 0
        self._num_finished_requests: int = 0

        # 活跃时间：只累计"有请求在处理中"的时段
        self._period_start: float | None = None
        self._total_active_time: float = 0.0

    # ── 事件入口 ──────────────────────────────────

    def on_new_records(self, req_id: str, num_prompts: int):
        self.metric_records[req_id] = RequestMetricRecord(
            added_time=time.time(), num_prompts=num_prompts
        )
        if self._period_start is None:
            self._period_start = time.time()

    def on_step_output(self, step_outputs: list[EngineStepOutput]):
        now = time.time()

        for output in step_outputs:
            rec = self.metric_records.get(output.req_id)
            if rec is None:
                continue

            if output.new_token_ids:
                if rec.last_time is None:
                    # 首个输出 token → 记录 TTFT 样本
                    self._ttft_samples.append(now - rec.added_time)
                rec.num_output_tokens += len(output.new_token_ids)
                rec.last_time = now
                self._total_output_tokens += len(output.new_token_ids)

            if output.finished:
                rec.finished = True
                self._num_finished_requests += 1

        # 所有请求都完成了 → 结算当前活跃时段
        if self._period_start is not None and not self._has_active_requests():
            self._total_active_time += now - self._period_start
            self._period_start = None

    def on_remove_record(self, req_id: str):
        """请求记录被清除时调用，避免 metric_records 无限增长。"""
        self.metric_records.pop(req_id, None)

    # ── 计算属性 ──────────────────────────────────

    @property
    def ttft(self) -> float:
        if not self._ttft_samples:
            return 0.0
        return sum(self._ttft_samples) / len(self._ttft_samples)

    @property
    def tps(self) -> float:
        t = self._active_elapsed_time()
        return self._total_output_tokens / t if t > 0 else 0.0

    @property
    def rps(self) -> float:
        t = self._active_elapsed_time()
        return self._num_finished_requests / t if t > 0 else 0.0

    @property
    def total_elapsed_time(self) -> float:
        return self._active_elapsed_time()

    @property
    def num_finished_requests(self) -> int:
        return self._num_finished_requests

    # ── 内部辅助 ──────────────────────────────────

    def _has_active_requests(self) -> bool:
        return any(not r.finished for r in self.metric_records.values())

    def _active_elapsed_time(self) -> float:
        total = self._total_active_time
        if self._period_start is not None:
            total += time.time() - self._period_start
        return total
