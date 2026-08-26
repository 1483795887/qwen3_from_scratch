import asyncio
import multiprocessing as mp
import pickle
from asyncio import Queue as AsyncQueue

import zmq

from qwen3_from_scratch.factory import BatchConfig
from qwen3_from_scratch.inference.engine.engine_core_proc import EngineCoreProc
from qwen3_from_scratch.inference.engine.entities import (
    AddRequestErrorMsg,
    AddRequestMsg,
    ReadyMsg,
    ShutdownMsg,
    StepOutputMsg,
)
from qwen3_from_scratch.utils.logger import get_logger

logger = get_logger(__name__)
_READY_TIMEOUT = 120  # 等待子进程就绪的超时（秒）


class AsyncMPClient:
    def __init__(
        self,
        config: BatchConfig,
        model_name: str,
        eos_token_id: int | list[int],
    ):
        self._config = config
        self._model_name = model_name
        self._eos_token_id = eos_token_id

        self._ctx: zmq.Context = zmq.Context()
        self._push_sock: zmq.Socket | None = None
        self._pull_sock: zmq.Socket | None = None

        self._process: mp.Process | None = None

        self._output_queue: AsyncQueue[StepOutputMsg | AddRequestErrorMsg] = (
            AsyncQueue()
        )

        self._recv_thread_task: asyncio.Task | None = None
        self._num_blocks: int = 0
        self._closed = False

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def _init_zmq(self, input_addr: str, output_addr: str):
        self._push_sock = self._ctx.socket(zmq.PUSH)
        self._push_sock.connect(input_addr)
        self._pull_sock = self._ctx.socket(zmq.PULL)
        self._pull_sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._pull_sock.connect(output_addr)

    async def start(self):
        bind_ready = mp.Queue()

        self._process = mp.Process(
            target=EngineCoreProc.run,
            args=(
                self._config,
                self._model_name,
                self._eos_token_id,
                bind_ready,
            ),
        )
        self._process.start()
        logger.info("等待引擎子进程 bind ZMQ socket")

        loop = asyncio.get_running_loop()
        # 阻塞等子进程 bind 完成（用 run_in_executor 避免阻塞事件循环）
        (input_addr, output_addr) = await loop.run_in_executor(
            None, bind_ready.get
        )
        self._init_zmq(input_addr, output_addr)

        # 等待 ReadyMsg
        logger.info("等待引擎子进程就绪")
        try:
            frames = await asyncio.wait_for(
                loop.run_in_executor(None, self._pull_sock.recv_multipart),
                timeout=_READY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"引擎子进程在 {_READY_TIMEOUT}s 内未就绪")
        msg = pickle.loads(frames[0])
        if isinstance(msg, ReadyMsg):
            self._num_blocks = msg.num_blocks
            logger.info(f"引擎就绪，可用块数: {self._num_blocks}")
        else:
            raise RuntimeError(f"期望 ReadyMsg，收到 {type(msg)}")

        self._recv_thread_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self):
        assert self._pull_sock is not None
        loop = asyncio.get_running_loop()
        while not self._closed:
            try:
                frames = await loop.run_in_executor(
                    None, self._pull_sock.recv_multipart
                )
                msg = pickle.loads(frames[0])
                await self._output_queue.put(msg)
            except zmq.ZMQError as e:
                if isinstance(e, zmq.Again):
                    continue
                if not self._closed:
                    logger.error("ZMQ 接收异常")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._closed:
                    logger.error(f"接收循环异常: {e}")
                break

    async def add_request(
        self,
        req_id: str,
        token_ids: list[int],
        max_new_tokens: int,
        ignore_eos: bool = False,
    ):
        assert self._pull_sock is not None
        msg = AddRequestMsg(
            req_id=req_id,
            token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
        )
        self._push_sock.send(pickle.dumps(msg))

    async def get_output(self) -> StepOutputMsg | AddRequestErrorMsg:
        return await self._output_queue.get()

    async def shutdown(self):
        if self._closed:
            return
        self._closed = True

        if self._push_sock is not None:
            try:
                self._push_sock.send(pickle.dumps(ShutdownMsg()))
            except zmq.ZMQError:
                pass

        if self._recv_thread_task is not None:
            self._recv_thread_task.cancel()
            try:
                await self._recv_thread_task
            except asyncio.CancelledError:
                pass

        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()

        if self._push_sock is not None:
            self._push_sock.close(linger=0)
        if self._pull_sock is not None:
            self._pull_sock.close(linger=0)
        if self._ctx is not None:
            self._ctx.term()
        logger.info("AsyncMPClient 已关闭")
