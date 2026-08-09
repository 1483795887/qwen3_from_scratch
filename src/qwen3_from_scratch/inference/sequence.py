from itertools import count
from copy import copy
from uuid import uuid4

import enum


class SequenceStatus(enum.Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


class Sequence:
    def __init__(self, prompts: list[int]):
        self.req_id = str(uuid4())
        self.prompts = prompts
        self.token_ids = copy(prompts)
        self.last_token_id = -1
        self.block_tables: list[int] = []
        self.status = SequenceStatus.WAITING
        self.is_prefill = True

    def __len__(self):
        return len(self.token_ids)