from qwen3_from_scratch.factory import BatchConfig
from qwen3_from_scratch.inference.engine.entities import EngineStepOutput
from qwen3_from_scratch.inference.model_runner.model_worker import ModelWorker
from qwen3_from_scratch.inference.scheduler import Scheduler, SchedulerConfig
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus
from qwen3_from_scratch.utils.logger import get_logger

logger = get_logger(__name__)


def build_scheduler_config_from_batch_config(config: BatchConfig, pages: int):
    return SchedulerConfig(
        config.scheduler.max_num_seqs,
        config.scheduler.max_num_tokens,
        config.scheduler.block_size,
        pages,
        enable_prefix_cache=config.scheduler.enable_prefix_cache,
        chunked_prefill_size=config.scheduler.chunked_prefill_size,
        watermark=config.scheduler.watermark,
    )


class EngineCore:
    """调度 + 推理核心。持有 Scheduler + ModelWorker, 执行 step 循环"""

    def __init__(
        self,
        config: BatchConfig,
        model_name: str,
        eos_token_id: int | list[int],
    ):
        self.config = config
        self.model_name = model_name
        self.eos_token_id = eos_token_id

        self.worker = ModelWorker(config, self.model_name)
        self.worker.init_context()

        self.scheduler = Scheduler(
            build_scheduler_config_from_batch_config(
                config, self.worker.kv_cache.num_pages
            ),
            check_seq_finish_func=self._check_seq_finish,
        )

    def _check_seq_finish(self, seq: Sequence) -> bool:
        eos_ids = self.eos_token_id
        if isinstance(eos_ids, int):
            eos_hit = seq.last_token_id == eos_ids
        else:
            eos_hit = seq.last_token_id in eos_ids
        return (eos_hit and not seq.ignore_eos) or (
            seq.generated_lens >= seq.max_new_tokens
        )

    def add_request(self, seq: Sequence) -> bool:
        return self.scheduler.add_request(seq)

    def step(self) -> list[EngineStepOutput]:
        planned = self.scheduler.schedule()
        if not planned:
            return []

        token_ids = self.worker.forward(planned)
        self.scheduler.post_process(planned, token_ids)
        outputs: list[EngineStepOutput] = []
        for seq in planned:
            new_token_ids = (
                [] if seq.last_token_id == -1 else [seq.last_token_id]
            )
            outputs.append(
                EngineStepOutput(
                    req_id=seq.req_id,
                    new_token_ids=new_token_ids,
                    finished=seq.status == SequenceStatus.FINISHED,
                    generated_token_num=seq.generated_lens,
                )
            )
        return outputs

    def has_requests(self) -> bool:
        return (len(self.scheduler.waiting) > 0) or (
            len(self.scheduler.active) > 0
        )

    @property
    def num_blocks(self) -> int:
        return self.worker.kv_cache.num_pages
