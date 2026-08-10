import multiprocessing

import torch
from torch.nn.utils.rnn import pad_sequence

from qwen3_from_scratch.factory import BatchConfig, GenerationDefaults
from qwen3_from_scratch.inference import (
    GreedySampler,
    ModelContext,
    Sampler,
    TemperatureSampler,
    TopKSampler,
    get_forward_context,
    set_forward_context,
)
from qwen3_from_scratch.inference.kv_cache.paged_cache import PagedKVCache
from qwen3_from_scratch.inference.logger import get_logger
from qwen3_from_scratch.inference.model_manager import ModelManager
from qwen3_from_scratch.inference.sequence import Sequence

logger = get_logger(__name__)


class ModelWorker:
    def __init__(self, config: BatchConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model, self.kv_cache, self.device, self.dtype = self._init_model(
            config, model_name
        )
        self.sampler = self._build_sampler(config.generation)

    def _build_sampler(self, config: GenerationDefaults) -> Sampler:
        """根据 temperature 和 top_k 构建 Sampler。"""
        if config.temperature > 0.0 and config.top_k > 0:
            return TopKSampler(
                top_k=config.top_k, temperature=config.temperature
            )
        elif config.temperature > 0.0:
            return TemperatureSampler(temperature=config.temperature)
        else:
            return GreedySampler()

    def _init_model(self, config: BatchConfig, model_name: str):
        if model_name not in config.list_model_names():
            raise KeyError(f"模型 {model_name} 不可用")
        model_manager = ModelManager(config)
        model = model_manager.load_model(model_name)
        model_info = config.get_model(model_name)
        ava_mem = PagedKVCache.get_available_mem()
        alloc_mem = int(config.scheduler.gpu_utilization * ava_mem)
        model_config = model.config
        blocks = PagedKVCache.get_block_num(
            alloc_mem,
            model_config.num_hidden_layers,
            model_config.num_key_value_heads,
            model_config.head_dim,
            device=model_info.device,
            dtype=config.kv_cache_dtype,
        )
        kv_cache = PagedKVCache(
            blocks,
            model_config.num_hidden_layers,
            model_config.num_key_value_heads,
            model_config.head_dim,
            device=model_info.device,
            dtype=config.kv_cache_dtype,
        )

        return model, kv_cache, model_info.device, model_info.dtype

    def init_context(self):
        model_info = self.config.get_model(self.model_name)
        context = ModelContext(
            dtype=model_info.dtype,
            use_cache=True,
            kv_cache=self.kv_cache,
            block_size=self.config.scheduler.block_size,
        )
        set_forward_context(context)

    @staticmethod
    def run(
        config: BatchConfig,
        model_name: str,
        request_mp: multiprocessing.Queue,
        result_mp: multiprocessing.Queue,
        get_blocks_mp: multiprocessing.Queue,
    ):
        worker = ModelWorker(config, model_name)
        worker.init_context()
        get_blocks_mp.put(worker.kv_cache.num_pages)
        while True:
            reqs = request_mp.get()
            if len(reqs) == 0:
                logger.info("推理进程退出")
                break
            result = worker.forward(reqs)
            result_mp.put(result)

    def _fill_common_context(
        self, context: ModelContext, seqs: list[Sequence]
    ):
        slot_mapping = []
        block_tables = []
        for seq in seqs:
            slots = []
            for i in range(0, len(seq), context.block_size):
                remaining = min(len(seq) - i, context.block_size)
                slots.extend(
                    list(
                        map(
                            lambda x: (
                                seq.block_tables[i // context.block_size]
                                * context.block_size
                                + x
                            ),
                            range(remaining),
                        )
                    )
                )
            slot_mapping.extend(slots[seq.cached_len :])
            block_tables.append(seq.block_tables)
        block_tables = pad_sequence(
            [
                torch.tensor(
                    block_table, device=self.device, dtype=torch.int32
                )
                for block_table in block_tables
            ],
            True,
            -1,
        )
        context.block_tables = block_tables
        context.slot_mapping = torch.tensor(
            slot_mapping, device=self.device, dtype=torch.int32
        )

    def build_context_prefill(self, seqs: list[Sequence]):
        context = get_forward_context()
        positions = []
        cum_seq_lens_q = [0]
        cum_seq_lens_kv = [0]
        for seq in seqs:
            positions.extend(list(range(len(seq.prompts))))
            last_cum_seq_q = cum_seq_lens_q[-1]
            last_cum_seq_k = cum_seq_lens_kv[-1]
            cum_seq_lens_q.append(last_cum_seq_q + len(seq.prompts))
            cum_seq_lens_kv.append(last_cum_seq_k + len(seq.prompts))

        device = self.device
        context.cum_seq_lens_kv = torch.tensor(
            cum_seq_lens_kv, device=device, dtype=torch.int32
        )
        context.cum_seq_lens_q = torch.tensor(
            cum_seq_lens_q, device=device, dtype=torch.int32
        )
        context.position_ids = torch.tensor(
            positions, device=device, dtype=torch.int32
        )
        self._fill_common_context(context, seqs)
        set_forward_context(context)

    def build_inputs_prefill(self, seqs: list[Sequence]):
        inputs = []
        for seq in seqs:
            inputs.extend(seq.prompts)
        return torch.tensor(inputs, dtype=torch.int32, device=self.device)

    def build_inputs_decode(self, seqs: list[Sequence]):
        return torch.tensor(
            [seq.last_token_id for seq in seqs],
            device=self.device,
            dtype=torch.int32,
        )

    def build_context_decode(self, seqs: list[Sequence]):
        context = get_forward_context()
        positions = []
        cum_seq_lens_q = [0]
        cum_seq_lens_kv = [0]
        device = self.device
        for seq in seqs:
            # 当前输入 token (last_token_id) 已被 post_process 追加进
            # token_ids, 索引 = len(seq) - 1, 位置编码取该索引
            positions.append(len(seq) - 1)
            last_cum_seq_q = cum_seq_lens_q[-1]
            last_cum_seq_k = cum_seq_lens_kv[-1]
            cum_seq_lens_q.append(last_cum_seq_q + 1)
            cum_seq_lens_kv.append(last_cum_seq_k + len(seq))
        self._fill_common_context(context, seqs)
        context.cum_seq_lens_kv = torch.tensor(
            cum_seq_lens_kv, device=device, dtype=torch.int32
        )
        context.cum_seq_lens_q = torch.tensor(
            cum_seq_lens_q, device=device, dtype=torch.int32
        )
        context.position_ids = torch.tensor(
            positions, device=device, dtype=torch.int32
        )
        set_forward_context(context)

    @torch.inference_mode
    def forward(self, seqs: list[Sequence]):
        # 先构建context，然后调用
        assert len(seqs)
        is_prefill = seqs[0].is_prefill
        if is_prefill:
            self.build_context_prefill(seqs)
            inputs = self.build_inputs_prefill(seqs)
        else:
            self.build_context_decode(seqs)
            inputs = self.build_inputs_decode(seqs)

        logits = self.model(inputs)

        if is_prefill:
            context = get_forward_context()
            indices = context.cum_seq_lens_q[1:] - 1
            logits = logits[indices]  # [B, vocab]
        next_ids = self.sampler(logits)
        return next_ids[:, 0].tolist()  # length B
