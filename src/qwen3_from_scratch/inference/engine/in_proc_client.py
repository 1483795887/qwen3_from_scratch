from qwen3_from_scratch.inference.engine.engine_core import EngineCore
from qwen3_from_scratch.inference.engine.entities import (
    EngineStepOutput,
    RequestFailedError,
)
from qwen3_from_scratch.inference.sequence import Sequence


class InProcClient:
    """同进程客户端，直接调用EngineCore"""

    def __init__(self, core: EngineCore):
        self._core = core
        self._sequences: dict[str, Sequence] = {}

    def add_request(
        self,
        req_id: str,
        token_ids: list[int],
        max_new_tokens: int,
        ignore_eos: bool = False,
    ):
        seq = Sequence(
            token_ids,
            max_new_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
            req_id=req_id,
        )
        if not self._core.add_request(seq):
            raise RequestFailedError(req_id, "prompt 过长，无法添加")
        self._sequences[req_id] = seq

    def step(self) -> list[EngineStepOutput]:
        return self._core.step()

    def has_requests(self) -> bool:
        return self._core.has_requests()

    def get_sequence(self, req_id: str) -> Sequence | None:
        return self._sequences.get(req_id)

    def remove_sequence(self, req_id: str) -> None:
        self._sequences.pop(req_id)
