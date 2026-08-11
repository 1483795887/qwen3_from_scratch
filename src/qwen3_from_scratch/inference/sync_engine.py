import time
from collections.abc import Iterator
from dataclasses import dataclass

from qwen3_from_scratch.factory import BatchConfig, load_batch_config
from qwen3_from_scratch.inference.llm_engine import PerfMetrics, StreamChunk
from qwen3_from_scratch.inference.logger import get_logger
from qwen3_from_scratch.inference.model_manager import ModelManager
from qwen3_from_scratch.inference.model_worker import ModelWorker
from qwen3_from_scratch.inference.scheduler import Scheduler, SchedulerConfig
from qwen3_from_scratch.inference.scheduler_driver import SchedulerDriver
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus

logger = get_logger(__name__)


@dataclass
class BatchPerfMetrics:
    """批量请求的汇总性能指标。"""

    per_request: list[PerfMetrics]
    num_requests: int
    total_tokens: int
    total_elapsed: float
    aggregate_tps: float


class SyncEngine:
    """进程内同步推理引擎，带连续批处理。

    与服务路径（LLMEngine）共享同一套调度循环（SchedulerDriver）与
    Paged 推理组件（ModelWorker），差别在：
      - 模型在当前进程加载、推理，无运行线程、无子进程；
      - 不需要 close()，python 进程结束即自然退出；
      - 接口全部同步：generate_stream / generate / batch_generate / warmup。
    """

    def __init__(
        self, config_path: str, model_name: str, log_interval: int = 0
    ):
        """
        Args:
            config_path: 批处理配置文件路径。
            model_name: 模型名称。
            log_interval: 生成过程中统计日志的打印间隔（步数）。
                <=0 表示不打印；>0 表示每 log_interval 步打印一次。
        """
        self.config: BatchConfig = load_batch_config(config_path)
        self.model_name = model_name
        self.log_interval = log_interval
        if model_name not in self.config.list_model_names():
            raise KeyError(f"模型 {model_name} 不可用")

        self.tokenizer = ModelManager(self.config).load_tokenizer(model_name)
        self.worker = ModelWorker(self.config, model_name)
        self.worker.init_context()

        self.driver = SchedulerDriver(
            Scheduler(
                SchedulerConfig(
                    self.config.scheduler.max_num_seqs,
                    self.config.scheduler.max_num_tokens,
                    self.config.scheduler.block_size,
                    self.worker.kv_cache.num_pages,
                ),
                check_seq_finish_func=self._check_seq_finish,
            ),
            self.worker.forward,
        )

    # ── 接口 ──────────────────────────────────

    def warmup(self, prompt: str = "你好", num_tokens: int = 3):
        """跑一轮快速请求，触发 Triton 内核编译，避免首个请求 TTFT 被污染。"""
        logger.info("开始预热")
        for _ in self.generate_stream(prompt, num_tokens):
            pass
        logger.info("预热完成")

    def generate_stream(
        self, prompt: str | list[dict], max_new_tokens: int | None = None
    ) -> Iterator[StreamChunk]:
        """同步流式生成，逐 chunk yield（delta + 指标）。"""
        max_new_tokens = (
            max_new_tokens
            if max_new_tokens
            else self.config.generation.max_new_tokens
        )
        for chunk in self._stream_impl(prompt, max_new_tokens):
            yield chunk

    def generate(
        self, prompt: str | list[dict], max_new_tokens: int | None = None
    ) -> str:
        """非流式生成，返回完整文本。"""
        return "".join(
            chunk.delta
            for chunk in self.generate_stream(prompt, max_new_tokens)
        )

    def batch_generate(
        self,
        prompts: list[str | list[dict]],
        max_new_tokens: int | None = None,
    ) -> tuple[list[str], BatchPerfMetrics]:
        """整批同步生成：一次性入队多个请求，连续批处理直到全部结束。

        返回 (按 prompts 顺序的文本列表, 批量指标)。
        """
        max_new_tokens = (
            max_new_tokens
            if max_new_tokens
            else self.config.generation.max_new_tokens
        )
        seqs = [self._make_sequence(p, max_new_tokens) for p in prompts]
        request_start = {seq.req_id: time.perf_counter() for seq in seqs}
        first_token_time: dict[str, float] = {}
        metrics: dict[str, PerfMetrics] = {}
        pending = set(seq.req_id for seq in seqs)

        for seq in seqs:
            self.driver.add_request(seq)
        step_count = 0
        while pending:
            planned = self.driver.step([])
            if not planned:
                time.sleep(0.001)
                continue
            step_count += 1
            now = time.perf_counter()
            for seq in planned:
                if seq.req_id not in pending:
                    continue
                start = request_start[seq.req_id]
                if (
                    seq.req_id not in first_token_time
                    and seq.last_token_id != -1
                ):
                    first_token_time[seq.req_id] = now
                if seq.status == SequenceStatus.FINISHED:
                    token_count = max(seq.generated_lens, 1)
                    first = first_token_time.get(seq.req_id, now)
                    totals = seq.generated_lens
                    metrics[seq.req_id] = PerfMetrics(
                        ttft=max(first - start, 0.0),
                        token_count=token_count,
                        tps=self._compute_tps(totals, first, now),
                        total_elapsed=now - start,
                    )
                    pending.discard(seq.req_id)
            self._maybe_log_batch_step(
                step_count,
                len(pending),
                len(seqs) - len(pending),
                seqs,
                now,
                request_start,
            )

        texts = [self._decode_tail(seq) for seq in seqs]
        ordered_metrics = [metrics[seq.req_id] for seq in seqs]
        total_tokens = sum(m.token_count for m in ordered_metrics)
        all_elapsed = (
            max(m.total_elapsed for m in ordered_metrics)
            if ordered_metrics
            else 0.0
        )
        aggregate_tps = total_tokens / all_elapsed if all_elapsed > 0 else 0.0
        batch = BatchPerfMetrics(
            per_request=ordered_metrics,
            num_requests=len(prompts),
            total_tokens=total_tokens,
            total_elapsed=all_elapsed,
            aggregate_tps=aggregate_tps,
        )
        return texts, batch

    # ── 内部 ──────────────────────────────────

    def _stream_impl(
        self, prompt: str | list[dict], max_new_tokens: int
    ) -> Iterator[StreamChunk]:
        """单个请求的生成热循环，驱动共享调度循环直到序列结束。"""
        seq = self._make_sequence(prompt, max_new_tokens)
        self.driver.add_request(seq)
        start_time = time.perf_counter()
        first_token_time: float | None = None
        token_count = 0

        while True:
            planned = self.driver.step([])
            if not planned:
                time.sleep(0.001)
                continue
            now = time.perf_counter()
            for s in planned:
                if s.req_id != seq.req_id:
                    continue
                if s.last_token_id == -1:
                    continue
                token_count += 1
                if first_token_time is None:
                    first_token_time = now
                    metrics = PerfMetrics(
                        ttft=now - start_time,
                        token_count=1,
                        tps=0.0,
                        total_elapsed=now - start_time,
                    )
                else:
                    total_elapsed = now - start_time
                    decode_elapsed = now - first_token_time
                    tps = (
                        (token_count - 1) / decode_elapsed
                        if decode_elapsed > 0
                        else 0.0
                    )
                    metrics = PerfMetrics(
                        ttft=first_token_time - start_time,
                        token_count=token_count,
                        tps=tps,
                        total_elapsed=total_elapsed,
                    )
                self._maybe_log_step(token_count, metrics)
                delta = self.tokenizer.decode(
                    [s.last_token_id], skip_special_tokens=True
                )
                yield StreamChunk(delta=delta, metrics=metrics)
                if s.status == SequenceStatus.FINISHED:
                    return

    def _make_sequence(
        self, prompt: str | list[dict], max_new_tokens: int
    ) -> Sequence:
        """prompt → Sequence，str/list 走 chat template 处理。"""
        text = self._prompt_to_text(prompt)
        token_ids = self.tokenizer(text)
        return Sequence(token_ids.input_ids, max_new_tokens=max_new_tokens)

    def _prompt_to_text(self, prompt: str | list[dict]) -> str:
        if isinstance(prompt, str):
            return prompt
        return self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _compute_tps(
        token_count: int, first_token_time: float, now: float
    ) -> float:
        delta = now - first_token_time
        return (token_count - 1) / delta if delta > 0 else 0.0

    def _decode_tail(self, seq: Sequence) -> str:
        """解码生成的部分 token_ids（剔除 prompt）。"""
        tail = seq.token_ids[len(seq.prompts) :]
        if not tail:
            tail = [seq.last_token_id]
        return self.tokenizer.decode(tail, skip_special_tokens=True)

    def _check_seq_finish(self, seq: Sequence) -> bool:
        eos_ids = self.tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_hit = seq.last_token_id == eos_ids
        else:
            eos_hit = seq.last_token_id in eos_ids
        return eos_hit or (seq.generated_lens >= seq.max_new_tokens)

    # ── 统计日志 ────────────────────────────────

    def _maybe_log_step(self, token_count: int, metrics: PerfMetrics) -> None:
        """单请求模式下，每 log_interval 步打印一次中间统计。"""
        if self.log_interval <= 0:
            return
        if token_count % self.log_interval != 0:
            return
        logger.info(
            "[stream] step=%d  TTFT=%.4fs  tokens=%d  TPS=%.2f  elapsed=%.4fs",
            token_count,
            metrics.ttft,
            metrics.token_count,
            metrics.tps,
            metrics.total_elapsed,
        )

    def _maybe_log_batch_step(
        self,
        step_count: int,
        pending: int,
        completed: int,
        seqs: list[Sequence],
        now: float,
        request_start: dict[str, float],
    ) -> None:
        """批量模式下，每 log_interval 步打印一次中间统计。"""
        if self.log_interval <= 0:
            return
        if step_count % self.log_interval != 0:
            return
        total_tokens_so_far = sum(s.generated_lens for s in seqs)
        batch_start = min(request_start.values())
        elapsed = now - batch_start
        aggregate_tps = total_tokens_so_far / elapsed if elapsed > 0 else 0.0
        logger.info(
            "[batch] step=%d  pending=%d  completed=%d  total_tokens=%d  aggregate_tps=%.2f  elapsed=%.4fs",
            step_count,
            pending,
            completed,
            total_tokens_so_far,
            aggregate_tps,
            elapsed,
        )
