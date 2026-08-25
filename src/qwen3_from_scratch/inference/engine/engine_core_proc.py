import multiprocessing as mp
import pickle

import zmq

from qwen3_from_scratch.factory.batch_config import BatchConfig
from qwen3_from_scratch.inference.sequence import Sequence
from qwen3_from_scratch.utils.logger import get_logger

from .engine_core import EngineCore
from .entities import (
    AbortRequestMsg,
    AddRequestErrorMsg,
    AddRequestMsg,
    ReadyMsg,
    ShutdownMsg,
    StepOutputMsg,
)

logger = get_logger(__name__)


def _handle_msg(msg, push_sock, core) -> bool:
    """处理一条从主进程收到的消息，返回是否应 shutdown。"""
    if isinstance(msg, ShutdownMsg):
        logger.info("收到关闭信号")
        return True
    elif isinstance(msg, AddRequestMsg):
        seq = Sequence(
            msg.token_ids,
            max_new_tokens=msg.max_new_tokens,
            req_id=msg.req_id,
            ignore_eos=msg.ignore_eos,
        )
        if not core.add_request(seq):
            push_sock.send(
                pickle.dumps(
                    AddRequestErrorMsg(
                        req_id=msg.req_id,
                        error_msg="prompt 过长，无法调度",
                    )
                )
            )
    elif isinstance(msg, AbortRequestMsg):
        core.abort_requests(msg.req_ids)
    return False


def _drain_input(pull_sock, push_sock, core) -> bool:
    """非阻塞排空输入队列，返回是否应 shutdown。"""
    while True:
        try:
            msg = pickle.loads(pull_sock.recv(flags=zmq.NOBLOCK))
        except zmq.Again:
            return False
        if _handle_msg(msg, push_sock, core):
            return True


def _recv_one(pull_sock, push_sock, core) -> bool:
    """阻塞等待一条消息（带超时），返回是否应 shutdown。"""
    try:
        msg = pickle.loads(pull_sock.recv())
        return _handle_msg(msg, push_sock, core)
    except zmq.Again:
        return False


class EngineCoreProc:
    RECV_TIMEOUT_MS = 100

    @staticmethod
    def run(
        config: BatchConfig,
        model_name: str,
        eos_token_id: int | list[int],
        bind_ready_queue: mp.Queue,
        input_addr: str | None = None,
        output_addr: str | None = None,
    ):
        core = EngineCore(config, model_name, eos_token_id)
        logger.info("Engine core 启动")
        ctx = zmq.Context(io_threads=2)
        pull_sock = ctx.socket(zmq.PULL)
        if input_addr:
            pull_sock.bind(input_addr)
        else:
            input_port = pull_sock.bind_to_random_port("tcp://*")
            input_addr = f"tcp://localhost:{input_port}"
        pull_sock.setsockopt(zmq.RCVTIMEO, EngineCoreProc.RECV_TIMEOUT_MS)

        push_sock = ctx.socket(zmq.PUSH)
        if output_addr:
            push_sock.bind(output_addr)
        else:
            output_port = push_sock.bind_to_random_port("tcp://*")
            output_addr = f"tcp://localhost:{output_port}"

        if bind_ready_queue is not None:
            bind_ready_queue.put((input_addr, output_addr))
        push_sock.send(pickle.dumps(ReadyMsg(num_blocks=core.num_blocks)))
        logger.info(f"就绪，可用块数:{core.num_blocks}")

        shutdown_flag = False

        try:
            while not shutdown_flag:
                shutdown_flag = _drain_input(pull_sock, push_sock, core)
                if shutdown_flag:
                    break

                if not core.has_requests():
                    shutdown_flag = _recv_one(pull_sock, push_sock, core)
                outputs = core.step()
                if outputs:
                    push_sock.send(pickle.dumps(StepOutputMsg(outputs)))

        finally:
            pull_sock.close(linger=0)
            push_sock.close(linger=0)
            ctx.term()
            logger.info("EngineCore 进程退出")
