import enum
from copy import copy
from uuid import uuid4


class SequenceStatus(enum.Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


class Sequence:
    def __init__(self, prompts: list[int], req_id: str | None = None):
        if req_id is None:
            req_id = str(uuid4())
        self.req_id = req_id
        self.prompts = prompts
        self.token_ids = copy(prompts)
        self.last_token_id = -1
        self.cached_len = 0
        self.block_tables: list[int] = []
        self.status = SequenceStatus.WAITING
        self.is_prefill = True

    def __len__(self):
        return len(self.token_ids)

    @property
    def generated_lens(self):
        return len(self) - len(self.prompts)
