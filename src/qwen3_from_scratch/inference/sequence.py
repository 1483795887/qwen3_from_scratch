import enum
from copy import copy
from uuid import uuid4


class SequenceStatus(enum.Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


class Sequence:
    def __init__(
        self,
        prompts: list[int],
        max_new_tokens: int,
        req_id: str | None = None,
        ignore_eos: bool = False
    ):
        if req_id is None:
            req_id = str(uuid4())
        self.req_id = req_id
        self.prompts = prompts
        self.token_ids = copy(prompts)
        self.last_token_id = -1
        # 已写入 KV 的 token 数：分段 prefill 的进度指针 / 前缀命中长度
        self.cached_len = 0
        # 本步消费的 token 数，由 Scheduler 写：decode=1，prefill 分段=chunk 长度
        self.num_tokens = 0
        self.block_tables: list[int] = []
        self.status = SequenceStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos

    def __len__(self):
        return len(self.token_ids)

    @property
    def is_prefill(self):
        """仍在填充 prompt 阶段（cached_len 尚未覆盖全部 prompt）。

        注意：这是调度器/worker 层的阶段标志，与注意力层的 KV 来源判定
        （由 block_tables 是否为空决定）无关，不要混淆。
        """
        return self.cached_len < len(self.prompts)

    @property
    def generated_lens(self):
        return len(self) - len(self.prompts)
