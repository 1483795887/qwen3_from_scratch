import time

from qwen3_from_scratch.factory import BatchConfig, load_batch_config
from qwen3_from_scratch.inference.engine.entities import EngineStepOutput
from qwen3_from_scratch.inference.model_manager import ModelManager


class RequestRecord:
    def __init__(self, req_id: str, num_prompts: int):
        self.req_id = req_id
        self.token_ids: list[int] = []
        self.finished = False
        self.added_time = time.time()
        self.first_token_time = None
        self.num_prompts = num_prompts


class LLMBase:
    def __init__(
        self, config_path: str, model_name: str, log_interval: int = 0
    ):
        self.config: BatchConfig = load_batch_config(config_path)
        self.model_name = model_name
        self.log_interval = log_interval
        if model_name not in self.config.list_model_names():
            raise KeyError(f"模型 {model_name} 不可用")

        self.tokenizer = ModelManager(self.config).load_tokenizer(model_name)
        self.eos = self.tokenizer.eos_token_id
        self.step_count = 0
        self.records: dict[str, RequestRecord] = {}

    def record_req(self, req_id: str, num_prompts: int):
        self.records[req_id] = RequestRecord(req_id, num_prompts)

    def remove_req_record(self, req_id: str):
        if req_id in self.records:
            del self.record_req[req_id]

    def is_all_finished(self, req_ids: set[str]):
        return all(
            req_id not in self.records or self.records[req_id].finished
            for req_id in req_ids
        )

    def _tokenize(
        self, prompt: str | list[dict], **template_kwargs
    ) -> list[int]:
        if isinstance(prompt, list):
            text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            return self.tokenizer(text).input_ids
        return self.tokenizer(prompt).input_ids

    def decode(self, req_id: str, token_ids: list[int]):
        """TODO: 这里还需要处理单元不能单独组成字符串的场景"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def on_step_output(self, step_outputs: list[EngineStepOutput]):
        self.step_count += 1
        for output in step_outputs:
            assert output.req_id in self.records
            record = self.records[output.req_id]
            if record.first_token_time is None:
                record.first_token_time = time.time()
            record.token_ids.extend(output.new_token_ids)
            record.finished = output.finished
