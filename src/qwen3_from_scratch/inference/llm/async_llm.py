import asyncio
import uuid
from asyncio import Queue as AsyncQueue
from dataclasses import dataclass

from qwen3_from_scratch.inference.engine.client.async_mp_client import (
    AsyncMPClient,
)
from qwen3_from_scratch.inference.engine.entities import (
    AddRequestErrorMsg,
    EngineStepOutput,
    RequestFailedError,
    StepOutputMsg,
)
from qwen3_from_scratch.inference.llm.llm_base import (
    GenerateParams,
    LLMBase,
)
from qwen3_from_scratch.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _RequestState:
    """主进程侧跟踪的请求状态。"""

    req_id: str
    queue: AsyncQueue  # Queue[EngineStepOutput | RequestFailedError]
    prompt_len: int
    is_streaming: bool = True
    max_new_tokens: int = 0
    ignore_eos: bool = False


@dataclass
class StreamChunk:
    """generate_stream 的每次 yield 单元。"""

    delta: str
    req_id: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0


class AsyncLLM(LLMBase):
    def __init__(self, config_path: str, model_name: str, **kwargs):
        super().__init__(config_path, model_name, **kwargs)
        self._client = AsyncMPClient(self.config, self.model_name, self.eos)
        self._output_handler_task: asyncio.Task | None = None

        self._requests: dict[str, _RequestState] = {}
        self._started = False

    async def _ensure_started(self):
        if self._started:
            return
        await self._client.start()
        self._output_handler_task = asyncio.create_task(self._output_handler())
        self._started = True

    async def _output_handler(self):
        while True:
            try:
                msg = await self._client.get_output()
                if isinstance(msg, AddRequestErrorMsg):
                    req_state = self._requests.get(msg.req_id)
                    if req_state is not None:
                        await req_state.queue.put(
                            RequestFailedError(msg.req_id, msg.error_msg)
                        )
                        del self._requests[msg.req_id]
                elif isinstance(msg, StepOutputMsg):
                    self.on_step_output(msg.outputs)
                    finished_ids: list[str] = []
                    for out in msg.outputs:
                        req_state = self._requests.get(out.req_id)
                        if req_state is None:
                            continue
                        await req_state.queue.put(out)
                        if out.finished:
                            finished_ids.append(out.req_id)
                    for rid in finished_ids:
                        del self._requests[rid]
            except Exception as e:
                logger.exception("_output_handler 异常", exc_info=e)
                for req_state in self._requests.values():
                    await req_state.queue.put(e)
                self._requests.clear()
                break

    async def generate_stream(
        self, prompt: str | list[dict], params: GenerateParams
    ):
        await self._ensure_started()

        req_id = str(uuid.uuid4())
        token_ids = self._tokenize(prompt, params)
        max_tokens = (
            params.max_new_tokens
            if params.max_new_tokens
            else self.config.generation.max_new_tokens
        )

        queue: AsyncQueue[EngineStepOutput | RequestFailedError] = AsyncQueue()
        self._requests[req_id] = _RequestState(
            req_id=req_id,
            queue=queue,
            prompt_len=len(token_ids),
            max_new_tokens=max_tokens,
            ignore_eos=params.ignore_eos,
        )
        prompt_len = len(token_ids)
        self.record_req(req_id, prompt_len)
        await self._client.add_request(
            req_id,
            token_ids,
            max_new_tokens=max_tokens,
            ignore_eos=params.ignore_eos,
        )

        while True:
            item = await queue.get()
            if isinstance(item, Exception):
                raise item
            if len(item.new_token_ids) == 0:
                continue

            delta = self.decode(item.req_id, item.new_token_ids)
            yield StreamChunk(
                delta=delta,
                req_id=req_id,
                prompt_tokens=prompt_len,
                generated_tokens=item.generated_token_num,
            )
            if item.finished:
                break

    async def warmup(self, prompt: str = "你好", num_tokens: int = 3):
        await self._ensure_started()
        logger.info("开始预热")
        async for _ in self.generate_stream(
            prompt, GenerateParams(max_new_tokens=num_tokens)
        ):
            pass
        logger.info("预热完成")

    async def close(self):
        if self._output_handler_task is not None:
            self._output_handler_task.cancel()
            try:
                await self._output_handler_task
            except asyncio.CancelledError:
                pass

        if self._started:
            await self._client.shutdown()
            self._started = False
