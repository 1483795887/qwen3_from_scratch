from dataclasses import dataclass, field


@dataclass
class EngineStepOutput:
    req_id: str
    new_token_ids: list[int]
    finished: bool = False
    generated_token_num: int = 0


@dataclass
class StepOutputMsg:
    """子进程 → 主进程：一轮 step 的全部输出。"""

    outputs: list[EngineStepOutput] = field(default_factory=list)


class RequestFailedError(Exception):
    """add_request 失败的错误，比如提示词超长"""

    def __init__(self, req_id: str, error_msg: str):

        self.req_id = req_id
        self.error_msg = error_msg
        super().__init__(f"请求{req_id} 添加失败: {error_msg}")


@dataclass
class ReadyMsg:
    """子进程 → 主进程：引擎就绪，携带可用 KV 块数。"""

    num_blocks: int = 0


@dataclass
class AddRequestMsg:
    """主进程 → 子进程：新增推理请求。"""

    req_id: str
    token_ids: list[int]
    max_new_tokens: int
    ignore_eos: bool = False


@dataclass
class AbortRequestMsg:
    """主进程 → 子进程：中止指定请求。"""

    req_ids: list[str] = field(default_factory=list)


@dataclass
class ShutdownMsg:
    """主进程 → 子进程：关闭引擎。"""

    pass


@dataclass
class AddRequestErrorMsg:
    """子进程 → 主进程：add_request 失败。"""

    req_id: str = ""
    error_msg: str = ""
