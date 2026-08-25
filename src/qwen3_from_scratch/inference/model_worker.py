import multiprocessing

import torch

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
from qwen3_from_scratch.utils.logger import get_logger
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

    def _query_tokens(self, seq: Sequence) -> list[int]:
        """本步要处理的输入 token。"""
        if seq.is_prefill:
            return seq.prompts[
                seq.cached_len : seq.cached_len + seq.num_tokens
            ]
        # decode：处理 token_ids 的最后一个 token（全命中时为 prompts[-1]）
        return [seq.token_ids[-1]]

    def _query_positions(self, seq: Sequence) -> list[int]:
        """本步输入 token 的位置编码。"""
        if seq.is_prefill:
            return list(range(seq.cached_len, seq.cached_len + seq.num_tokens))
        return [len(seq.token_ids) - 1]

    def _kv_len(self, seq: Sequence) -> int:
        """本步注意力可读的 KV 长度（写后）。"""
        if seq.is_prefill:
            return seq.cached_len + seq.num_tokens
        return len(seq.token_ids)

    def _write_positions(self, seq: Sequence) -> range:
        """本步需要写入 KV 的绝对位置（全命中 decode 时为空）。"""
        if seq.is_prefill:
            return range(seq.cached_len, seq.cached_len + seq.num_tokens)
        return range(seq.cached_len, len(seq.token_ids))

    def _fill_common_context(
        self, context: ModelContext, seqs: list[Sequence]
    ):
        slot_mapping = []
        block_tables = []
        for seq in seqs:
            for pos in self._write_positions(seq):
                slot = (
                    seq.block_tables[pos // context.block_size]
                    * context.block_size
                    + pos % context.block_size
                )
                slot_mapping.append(slot)
            block_tables.append(seq.block_tables)
        # block_tables 一次 CPU 列表 → 单次 HtoD（T7：原实现逐条 .to(device) ×M 次）
        max_blocks = max(len(s.block_tables) for s in seqs)
        padded = [
            s.block_tables + [-1] * (max_blocks - len(s.block_tables))
            for s in seqs
        ]
        context.block_tables = torch.tensor(
            padded, device=self.device, dtype=torch.int32
        )
        context.slot_mapping = torch.tensor(
            slot_mapping, device=self.device, dtype=torch.int32
        )

    def _build_rope_cos_sin(self, positions: list[int]):
        """CPU 侧索引 cos/sin（无 CUDA 同步），一次 .to(device) 拷回（T4）。

        替代自注意力层每层每步的 position_ids.cpu() + cache[pos] + cat，
        把每步多次 D2H 同步 + 拷回压缩为每步 1 次。
        """
        from qwen3_from_scratch.models.rotary import get_rope

        cfg = self.model.config
        rotary = get_rope(
            cfg.head_dim,
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.pos_embed_params["rope_theta"],
        )
        dtype = self.dtype
        device = self.device
        cos_sin = rotary.cos_sin_cache[positions].to(device, dtype)
        half = cos_sin.shape[-1] // 2
        cos = torch.cat([cos_sin[..., :half], cos_sin[..., :half]], dim=-1)
        sin = torch.cat([cos_sin[..., half:], cos_sin[..., half:]], dim=-1)
        return cos.squeeze(1), sin.squeeze(1)

    def build_context(self, seqs: list[Sequence]):
        """混合 batch 的统一上下文构建（prefill 分段 + decode 共存）。"""
        context = get_forward_context()
        positions = []
        cum_seq_lens_q = [0]
        cum_seq_lens_kv = [0]
        for seq in seqs:
            positions.extend(self._query_positions(seq))
            cum_seq_lens_q.append(cum_seq_lens_q[-1] + seq.num_tokens)
            cum_seq_lens_kv.append(cum_seq_lens_kv[-1] + self._kv_len(seq))

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
        if device == "cuda":
            # T4：引擎侧预取 cos/sin 与 max_seqlen（每步一次，避免每层每步 D2H）
            context.cos, context.sin = self._build_rope_cos_sin(positions)
            context.max_seqlen_q = max(s.num_tokens for s in seqs)
            context.max_seqlen_k = max(self._kv_len(s) for s in seqs)
        set_forward_context(context)

    def build_inputs(self, seqs: list[Sequence]):
        inputs = []
        for seq in seqs:
            inputs.extend(self._query_tokens(seq))
        return torch.tensor(inputs, dtype=torch.int32, device=self.device)

    @torch.inference_mode
    def forward(self, seqs: list[Sequence]):
        assert len(seqs)
        self.build_context(seqs)
        inputs = self.build_inputs(seqs)

        # 每条序列取自身 query 区间最后一个 token 的 logits
        context = get_forward_context()
        indices = context.cum_seq_lens_q[1:] - 1

        hidden = self.model.forward_hidden(inputs)
        # 只用算最后一个词元的logits
        logits = self.model.compute_logits(hidden[indices])  # [B, vocabs]

        next_ids = self.sampler(logits)
        return next_ids[:, 0].tolist()  # length B
