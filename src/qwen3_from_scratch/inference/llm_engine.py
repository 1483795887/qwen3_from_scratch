import asyncio
import threading
import time
from uuid import uuid4

from qwen3_from_scratch.factory import load_batch_config, BatchConfig
from qwen3_from_scratch.inference.model_manager import ModelManager
from qwen3_from_scratch.inference.model_worker import ModelWorker
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus
from qwen3_from_scratch.inference.scheduler import Scheduler, SchedulerConfig
import multiprocessing as mp
from asyncio import Queue, AbstractEventLoop
from collections.abc import AsyncIterator
from dataclasses import dataclass

from qwen3_from_scratch.inference.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RequestResult:
    delta: str
    is_finished: bool


class Request:
    def __init__(self, prompt: str, loop: AbstractEventLoop, queue: Queue[RequestResult], is_streaming: bool = True):
        self.req_id = str(uuid4())
        self.prompt = prompt
        self.loop = loop
        self.queue = queue
        self.is_streaming = is_streaming


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
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        self.finished = False

    def setup_worker(self, config: BatchConfig, model_name: str):
        worker_process = mp.Process(
            target=ModelWorker.run,
            args=(config, model_name, self.request_queue, self.response_queue, self.get_blocks_queue)
        )
        worker_process.start()
        return worker_process

    def _decode_and_send_result(self, tokens:list[int], request:Request):
        result = self.tokenizer.decode(tokens, skip_special_tokens=True)
        asyncio.run_coroutine_threadsafe(
            request.queue.put(
                RequestResult(result, False)
            ),
            request.loop
        )

    def _post_process(self, seqs:list[Sequence]):
        for seq in seqs:
            assert seq.req_id in self.requests
            request = self.requests[seq.req_id]

            if request.is_streaming and seq.last_token_id != -1:
                self._decode_and_send_result([seq.last_token_id], request)

            if seq.status == SequenceStatus.FINISHED:
                logger.debug(f"seq finish: {seq.req_id}")
                if not request.is_streaming:
                    self._decode_and_send_result(seq.token_ids[len(seq.prompts):], request)
                del self.requests[seq.req_id]
                asyncio.run_coroutine_threadsafe(
                    request.queue.put(
                        RequestResult("", True)
                    ),
                    request.loop
                )

    def _check_seq_finish(self, seq:Sequence):
        return (seq.last_token_id == self.tokenizer.eos_token_id) or (
                    seq.generated_lens > self.config.generation.max_new_tokens)


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
        scheduler = Scheduler(SchedulerConfig(self.config.scheduler.max_num_seqs, self.config.scheduler.max_num_tokens,
                                              self.config.scheduler.block_size, blocks),
                              check_seq_finish_func=self._check_seq_finish)
        while not self.finished:
            # 接收请求
            reqs = self._get_incoming_requests()
            if reqs:
                logger.debug(f"get {len(reqs)} reqs")
            for req in reqs:
                token_ids = self.tokenizer(req.prompt)
                seq = Sequence(token_ids.input_ids, req_id=req.req_id)
                self.requests[req.req_id] = req
                scheduler.add_request(seq)
            # 调度
            seqs = scheduler.schedule()
            if len(seqs) == 0:
                time.sleep(0.1)
                continue
            self.request_queue.put(seqs)
            result_token_ids = self.response_queue.get()
            scheduler.post_process(seqs, result_token_ids)
            self._post_process(seqs)
        # 约定，发送长度为0的就是结束
        self.request_queue.put([])


    async def generate_stream(self, prompt: str | list[dict]) -> AsyncIterator[StreamChunk]:
        """异步流式生成，yield StreamChunk（解码文本 + 性能指标）。"""
        queue: asyncio.Queue[RequestResult] = asyncio.Queue()
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        request = Request(prompt, asyncio.get_event_loop(), queue, True)
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
                tps = (token_count - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
                metrics = PerfMetrics(
                    ttft=first_token_time - start_time,
                    token_count=token_count,
                    tps=tps,
                    total_elapsed=total_elapsed,
                )

            yield StreamChunk(delta=item.delta, metrics=metrics)

    def close(self):
        self.finished = True
        self.thread.join()
        # 约定，发送长度为0的就是结束
        self.request_queue.put([])
