from uuid import uuid4

from qwen3_from_scratch.inference.engine.engine_core import EngineCore
from qwen3_from_scratch.inference.engine.in_proc_client import InProcClient
from qwen3_from_scratch.inference.engine.llm_base import LLMBase
from qwen3_from_scratch.utils.logger import get_logger

logger = get_logger(__name__)


class LLM(LLMBase):
    def __init__(self, config_path: str, model_name: str, **kwargs):
        super().__init__(config_path, model_name, **kwargs)
        self._client = InProcClient(
            EngineCore(self.config, model_name, self.eos)
        )

    def warmup(
        self, prompt: str = "直接回复一个词：收到", num_tokens: int = 3
    ):
        logger.info("开始预热")
        self.generate(prompt, num_tokens)
        logger.info("预热完毕")

    def generate(
        self,
        prompt: str | list[dict],
        max_new_tokens: int | None = None,
        ignore_eos: bool = False,
        **template_kwargs,
    ):
        return self.batch_generate(
            [prompt], max_new_tokens, ignore_eos, **template_kwargs
        )[0]

    def batch_generate(
        self,
        prompts: list[str | list[dict]],
        max_new_tokens: int | None = None,
        ignore_eos: bool = False,
        **template_kwargs,
    ) -> list[str]:
        max_tokens = (
            max_new_tokens
            if max_new_tokens
            else self.config.generation.max_new_tokens
        )
        all_reqs = []
        for p in prompts:
            rid = str(uuid4())
            token_ids = self._tokenize(p, **template_kwargs)
            all_reqs.append(rid)
            self._client.add_request(rid, token_ids, max_tokens, ignore_eos)
            self.record_req(rid, len(token_ids))

        pending_reqs = set(all_reqs)
        while not self.is_all_finished(pending_reqs):
            outputs = self._client.step()
            self.on_step_output(outputs)

        result = [
            self.decode(req_id, self.records[req_id].token_ids)
            for req_id in all_reqs
        ]
        for req_id in pending_reqs:
            self.remove_req_record(req_id)
        return result
