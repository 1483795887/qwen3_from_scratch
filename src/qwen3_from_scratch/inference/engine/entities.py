from dataclasses import dataclass


@dataclass
class EngineStepOutput:
    req_id: str
    new_token_ids: list[int]
    finished: bool = False
    generated_token_num: int = 0


class RequestFailedError(Exception):
    """add_request 失败的错误，比如提示词超长"""

    def __init__(self, req_id: str, error_msg: str):

        self.req_id = req_id
        self.error_msg = error_msg
        super().__init__(f"请求{req_id} 添加失败: {error_msg}")
