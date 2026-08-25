import asyncio
import multiprocessing as mp
import threading
import time
from asyncio import AbstractEventLoop, Queue
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

from qwen3_from_scratch.factory import BatchConfig, load_batch_config
from qwen3_from_scratch.utils.logger import get_logger
from qwen3_from_scratch.inference.model_manager import ModelManager
from qwen3_from_scratch.inference.model_worker import ModelWorker
from qwen3_from_scratch.inference.scheduler import Scheduler, SchedulerConfig
from qwen3_from_scratch.inference.scheduler_driver import SchedulerDriver
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus

logger = get_logger(__name__)


@dataclass
class RequestResult:
    delta: str
    is_finished: bool


class Request:
    def __init__(
        self,
        prompt: str,
        loop: AbstractEventLoop,
        queue: Queue[RequestResult],
        is_streaming: bool = True,
        max_new_tokens: int | None = None,
        ignore_eos: bool = False,
    ):
        self.req_id = str(uuid4())
        self.prompt = prompt
        self.loop = loop
        self.ignore_eos = ignore_eos
        self.queue = queue
        self.is_streaming = is_streaming
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos
        self.prompt_len = 0


@dataclass
class PerfMetrics:
    """单次请求的运行时性能指标快照，每个 chunk 更新。

    全部基于 consumer 侧 wall-clock 测量（见 CONTEXT.md）。
    """

    ttft: float
    token_count: int
    tps: float
    total_elapsed: float


@dataclass
class StreamChunk:
    """generate_stream 的每次 yield 单元。"""

    delta: str
    metrics: PerfMetrics
    req_id: str = ""
    prompt_tokens: int = 0


class LLMEngine:
    def __init__(self, config_path: str, model_name: str):
        self.config = load_batch_config(config_path)
        self.model_name = model_name
        if model_name not in self.config.list_model_names():
            raise KeyError(f"模型 {model_name} 不可用")
        self.tokenizer = ModelManager(self.config).load_tokenizer(model_name)
        self.incoming_requests: asyncio.Queue[Request] = asyncio.Queue()

        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        self.get_blocks_queue = mp.Queue()
        self.num_blocks = 0

        self.requests: dict[str, Request] = {}
        self._ready_event = threading.Event()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        self.finished = False

    def setup_worker(self, config: BatchConfig, model_name: str):
        worker_process = mp.Process(
            target=ModelWorker.run,
            args=(
                config,
                model_name,
                self.request_queue,
                self.response_queue,
                self.get_blocks_queue,
            ),
        )
        worker_process.start()
        return worker_process

    def _decode_and_send_result(self, tokens: list[int], request: Request):
        result = self.tokenizer.decode(tokens, skip_special_tokens=True)
        asyncio.run_coroutine_threadsafe(
            request.queue.put(RequestResult(result, False)), request.loop
        )

    def _post_process(self, seqs: list[Sequence]):
        for seq in seqs:
            assert seq.req_id in self.requests
            request = self.requests[seq.req_id]

            if request.is_streaming and seq.last_token_id != -1:
                self._decode_and_send_result([seq.last_token_id], request)

            if seq.status == SequenceStatus.FINISHED:
                logger.debug(f"seq finish: {seq.req_id}")
                if not request.is_streaming:
                    self._decode_and_send_result(
                        seq.token_ids[len(seq.prompts) :], request
                    )
                del self.requests[seq.req_id]
                asyncio.run_coroutine_threadsafe(
                    request.queue.put(RequestResult("", True)), request.loop
                )

    def _check_seq_finish(self, seq: Sequence):
        return (
            seq.last_token_id == self.tokenizer.eos_token_id
            and not seq.ignore_eos
        ) or (seq.generated_lens >= seq.max_new_tokens)

    def _get_incoming_requests(self) -> list[Request]:
        result = []
        while not self.incoming_requests.empty():
            try:
                req = self.incoming_requests.get_nowait()
                result.append(req)
            except asyncio.QueueEmpty:
                break
        return result

    def run(self):
        self.setup_worker(self.config, self.model_name)
        logger.info("等待模型启动中")
        blocks = self.get_blocks_queue.get()
        logger.info(f"可用块数:{blocks}")
        scheduler = Scheduler(
            SchedulerConfig(
                self.config.scheduler.max_num_seqs,
                self.config.scheduler.max_num_tokens,
                self.config.scheduler.block_size,
                blocks,
                enable_prefix_cache=self.config.scheduler.enable_prefix_cache,
                chunked_prefill_size=self.config.scheduler.chunked_prefill_size,
                watermark=self.config.scheduler.watermark,
            ),
            check_seq_finish_func=self._check_seq_finish,
        )
        # worker 已加载完模型并上报块数，此后请求的 TTFT 不再包含加载耗时
        self._ready_event.set()

        def worker_forward(seqs: list[Sequence]) -> list[int]:
            self.request_queue.put(seqs)
            return self.response_queue.get()

        self.driver = SchedulerDriver(scheduler, worker_forward)
        while not self.finished:
            # 接收请求
            reqs = self._get_incoming_requests()
            if reqs:
                logger.debug(f"get {len(reqs)} reqs")
            new_seqs = []
            for req in reqs:
                token_ids = self.tokenizer(req.prompt)
                max_new_tokens = (
                    req.max_new_tokens
                    if req.max_new_tokens
                    else self.config.generation.max_new_tokens
                )
                seq = Sequence(
                    token_ids.input_ids,
                    req_id=req.req_id,
                    max_new_tokens=max_new_tokens,
                    ignore_eos=req.ignore_eos,
                )
                req.prompt_len = len(seq.token_ids)
                self.requests[req.req_id] = req
                new_seqs.append(seq)
            seqs = self.driver.step(
                new_seqs
            )  # 共享调度驱动：入队、调度、推理、回填
            if len(seqs) == 0:
                time.sleep(0.1)
                continue
            self._post_process(seqs)
        # 约定，发送长度为0的就是结束
        self.request_queue.put([])

    async def generate_stream(
        self,
        prompt: str | list[dict],
        max_new_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        enable_thinking: bool | None = None,
        ignore_eos: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """异步流式生成，yield StreamChunk（解码文本 + 性能指标）。

        当 ``prompt`` 为 OpenAI 风格消息列表时，内部会经
        ``tokenizer.apply_chat_template`` 渲染为模型输入；``tools`` /
        ``tool_choice`` / ``enable_thinking`` 仅在此情况下生效，会透传给
        chat template（Qwen3 模板据此注入工具声明与思维模式）。

        注意：Qwen3 的工具调用/思维标记（``...``、``...``）
        在词表中被标记为 ``special=false``，因此即使解码时
        ``skip_special_tokens=True`` 也会保留在输出文本中，调用方可在流式
        文本里解析出工具调用。
        """
        queue: asyncio.Queue[RequestResult] = asyncio.Queue()
        if isinstance(prompt, list):
            # 补全会话必须加 assistant 生成提示（与 vLLM 一致）；缺了它模板会少
            # 渲染 <|im_start|>assistant\n，导致实际输入比客户端预期少 3 个 token
            template_kwargs: dict = {"add_generation_prompt": True}
            if tools is not None:
                template_kwargs["tools"] = tools
            if tool_choice is not None:
                template_kwargs["tool_choice"] = tool_choice
            if enable_thinking is not None:
                template_kwargs["enable_thinking"] = enable_thinking
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, **template_kwargs
            )
        request = Request(
            prompt,
            asyncio.get_event_loop(),
            queue,
            True,
            max_new_tokens=max_new_tokens
            if max_new_tokens
            else self.config.generation.max_new_tokens,
            ignore_eos=ignore_eos,
        )
        logger.debug(f"put req: {request.req_id}")

        start_time = time.perf_counter()
        await self.incoming_requests.put(request)

        first_token_time: float | None = None
        token_count = 0
        while True:
            item = await queue.get()
            if item.is_finished:
                break

            now = time.perf_counter()
            token_count += 1

            if first_token_time is None:
                # 第一个 token：只有 prefill，无 decode 阶段
                first_token_time = now
                ttft = first_token_time - start_time
                metrics = PerfMetrics(
                    ttft=ttft,
                    token_count=1,
                    tps=0.0,
                    total_elapsed=ttft,
                )
            else:
                # 后续 token：running 平均 TPS = (token_count - 1) / decode_elapsed
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

            yield StreamChunk(
                delta=item.delta,
                metrics=metrics,
                req_id=request.req_id,
                prompt_tokens=request.prompt_len,
            )

    def wait_ready(self):
        """阻塞直到推理进程完成模型加载。

        模型加载发生在 worker 进程，若在就绪前发请求，加载耗时会被计入 TTFT。
        调用本方法后，后续请求的 TTFT 不再包含加载耗时。
        """
        self._ready_event.wait()
        logger.info("模型加载完成")

    async def warmup(self, prompt: str = "你好", num_tokens: int = 3):
        """等待模型就绪并跑一轮预热请求（prefill + 若干 decode 步）。

        预热可同时触发 Triton 内核编译，避免首轮请求的 TTFT 被编译耗时污染。
        应在真实请求之前调用。预热期间会临时调低 max_new_tokens 以保持快速，结束后恢复。
        """
        self.wait_ready()
        logger.info("开始预热")
        old_max_tokens = self.config.generation.max_new_tokens
        try:
            self.config.generation.max_new_tokens = num_tokens
            async for _ in self.generate_stream(prompt):
                pass
        finally:
            self.config.generation.max_new_tokens = old_max_tokens
        logger.info("预热完成")

    def close(self):
        self.finished = True
        self.thread.join()
        # 约定，发送长度为0的就是结束
        self.request_queue.put([])
