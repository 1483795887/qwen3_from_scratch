import math

import torch
from tqdm import tqdm

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
from qwen3_from_scratch.inference.model_runner.model_manager import (
    ModelManager,
)
from qwen3_from_scratch.inference.sequence import Sequence
from qwen3_from_scratch.utils.logger import get_logger

logger = get_logger(__name__)


class ModelWorker:
    CUDA_GRAPH_MAX_LEN = 512

    def __init__(self, config: BatchConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model, self.kv_cache, self.device, self.dtype = self._init_model(
            config, model_name
        )
        self.sampler = self._build_sampler(config.generation)
        self.graphs = {}
        self.graph_bs = []
        self.graph_pool = None
        self.graph_vars = {}
        # 最大长度的 RoPE cos/sin 表，warmup 阶段建一次，运行时按 position_ids 切片
        self._cos_table, self._sin_table = self._build_cos_sin_tables()

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

    def warmup(self):
        model_info = self.config.get_model(self.model_name)
        context = ModelContext(
            dtype=model_info.dtype,
            use_cache=True,
            kv_cache=self.kv_cache,
            block_size=self.config.scheduler.block_size,
        )
        set_forward_context(context)
        if torch.cuda.is_available():
            self.graph_vars = self.capture_cudagraph()

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
            slot_mapping,
            device=self.device,
            dtype=torch.int32,
        )

    def _build_cos_sin_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """init 阶段建一次最大长度的 cos/sin 表，dtype/device 与 model 对齐。

        运行时每步只做 GPU 侧切片，无 .cpu() 同步（CUDA graph capture 兼容）。
        """
        from qwen3_from_scratch.models.rotary import build_cos_sin_table

        cfg = self.model.config
        return build_cos_sin_table(
            head_dim=cfg.head_dim,
            max_pos=cfg.max_position_embeddings,
            base=cfg.pos_embed_params["rope_theta"],
            device=self.device,
            dtype=self.dtype,
        )

    def _slice_cos_sin(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从最大长度表中按 position_ids 切片，返回 (N, head_dim) 的 cos / sin。"""
        # positions 已经是 device tensor (non_blocking HtoD)，直接高级索引
        return self._cos_table[positions], self._sin_table[positions]

    def _should_use_cuda_graph(self, seqs: list[Sequence]):
        if not torch.cuda.is_available() or (
            self.device == "cpu" or self.device == torch.device("cpu")
        ):
            return False
        return len(seqs) < ModelWorker.CUDA_GRAPH_MAX_LEN

    def _fill_decode_context(
        self, seqs: list[Sequence], context: ModelContext
    ):
        positions = []
        context.use_decode_graph = self._should_use_cuda_graph(seqs)
        context.context_lens = torch.tensor(
            [len(seq) for seq in seqs], dtype=torch.int32, device=self.device
        )
        # decode 每条序列只 query 最后一个 token 的位置
        for seq in seqs:
            positions.extend(self._query_positions(seq))
        context.position_ids = torch.tensor(
            positions, dtype=torch.int32, device=self.device
        )

    def _fill_prefill_or_mixed_context(
        self, seqs: list[Sequence], context: ModelContext
    ):
        positions = []
        context.use_decode_graph = False
        cum_seq_lens_q = [0]
        cum_seq_lens_kv = [0]
        for seq in seqs:
            positions.extend(self._query_positions(seq))
            cum_seq_lens_q.append(cum_seq_lens_q[-1] + seq.num_tokens)
            cum_seq_lens_kv.append(cum_seq_lens_kv[-1] + self._kv_len(seq))

        context.cum_seq_lens_kv = torch.tensor(
            cum_seq_lens_kv, dtype=torch.int32, device=self.device
        )
        context.cum_seq_lens_q = torch.tensor(
            cum_seq_lens_q, dtype=torch.int32, device=self.device
        )
        context.position_ids = torch.tensor(
            positions, dtype=torch.int32, device=self.device
        )
        context.position_ids = torch.tensor(
            positions, dtype=torch.int32, device=self.device
        )

    def build_context(self, seqs: list[Sequence]):
        """混合 batch 的统一上下文构建（prefill 分段 + decode 共存）。"""
        context = get_forward_context()
        is_pure_decode = all(not seq.is_prefill for seq in seqs)
        device = self.device

        if is_pure_decode:
            self._fill_decode_context(seqs, context)
        else:
            self._fill_prefill_or_mixed_context(seqs, context)
        self._fill_common_context(context, seqs)
        if device == "cuda":
            context.cos, context.sin = self._slice_cos_sin(
                context.position_ids
            )
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
        is_pure_decode = all(not seq.is_prefill for seq in seqs)
        if is_pure_decode and self._should_use_cuda_graph(seqs):
            # lm_head 已并入 decode graph，replay 直接返回 token ids
            logits = self.replay_cuda_graph(seqs)
        else:
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

    @torch.inference_mode
    def replay_cuda_graph(self, seqs: list[Sequence]):
        bs = len(seqs)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        host = self._host_stage
        host_np = self._host_np

        lens = [len(s) for s in seqs]
        block_size = self.kv_cache.block_size
        slots = []
        for s, le in zip(seqs, lens):
            p = le - 1
            slots.append(
                s.block_tables[p // block_size] * block_size + p % block_size
            )
        max_blocks = max(len(s.block_tables) for s in seqs)
        padded = [
            s.block_tables + [-1] * (max_blocks - len(s.block_tables))
            for s in seqs
        ]

        # 纯 CPU 填充, 直接写入numpy 无 CUDA 调用
        host_np["input_ids"][:bs] = [s.token_ids[-1] for s in seqs]
        host_np["context_lens"][:bs] = lens
        host_np["position_ids"][:bs] = [le - 1 for le in lens]
        host_np["slot_mapping"][:bs] = slots
        host_np["block_tables"][:bs, :max_blocks] = padded

        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["context_lens"].fill_(1)
        graph_vars["input_ids"][:bs].copy_(
            host["input_ids"][:bs], non_blocking=True
        )
        graph_vars["context_lens"][:bs].copy_(
            host["context_lens"][:bs], non_blocking=True
        )
        graph_vars["position_ids"][:bs].copy_(
            host["position_ids"][:bs], non_blocking=True
        )
        graph_vars["slot_mapping"][:bs].copy_(
            host["slot_mapping"][:bs], non_blocking=True
        )
        graph_vars["block_tables"][:bs, :max_blocks].copy_(
            host["block_tables"][:bs, :max_blocks], non_blocking=True
        )
        graph.replay()
        return graph_vars["logits"][:bs]

    @torch.inference_mode()
    def capture_cudagraph(self):
        model_info = self.config.get_model(self.model_name)
        model_config = self.model.config
        max_bs = min(
            self.config.scheduler.max_num_seqs, ModelWorker.CUDA_GRAPH_MAX_LEN
        )
        max_num_blocks = math.ceil(
            model_config.max_position_embeddings
            / self.config.scheduler.block_size,
        )
        input_ids = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        positions = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32, device=self.device
        )
        slot_mappings = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        # capture 阶段 slot 全填 -1：KV update 核遇到 slot<0 直接 return，
        # 避免 capture 把 token-0 的 k/v 写进真实 KV cache 的前 max_bs 个槽位
        # （这些垃圾数据在块复用/前缀命中场景可能被后续 decode 读到）。
        slot_mappings.fill_(-1)
        context_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        outputs = torch.zeros(
            max_bs,
            model_config.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        logits = torch.zeros(
            max_bs,
            model_config.vocab_size,
            dtype=self.dtype,
            device=self.device,
        )
        # cos/sin 占位：分配 max_bs 缓冲，capture 时按 positions[:bs]=0
        # 切片到 _cos_table / _sin_table 的 position-0 行 (cos=1, sin=0)，
        # 让 RoPE 在 capture 阶段是恒等映射；replay 时由 graph_vars["cos"/"sin"] 覆盖。
        cos = torch.empty(
            max_bs,
            model_config.head_dim,
            dtype=model_info.dtype,
            device=self.device,
        )
        sin = torch.empty(
            max_bs,
            model_config.head_dim,
            dtype=model_info.dtype,
            device=self.device,
        )
        self.graph_bs = [1, 2, 4, 8, 12, 16, 20, 24, 28] + list(
            range(32, max_bs + 1, 16)
        )

        context = get_forward_context()
        logger.info("记录 Cuda 图中")
        # 捕获阶段随机产生输入，防止计算出现异常，把kvcache清零
        # 这是启动时做的，不影响推理
        self.kv_cache.k_cache.zero_()
        self.kv_cache.v_cache.zero_()
        # 解码的长度都是1
        context_lens.fill_(1)
        context.use_decode_graph = True
        context.block_tables = block_tables[:1]
        context.context_lens = context_lens[:1]
        context.cos = self._cos_table[positions[:1]]
        context.sin = self._sin_table[positions[:1]]
        context.position_ids = positions[:1]
        context.slot_mapping = slot_mappings[:1]
        with torch.no_grad():
            _ = self.model.forward_hidden(input_ids[:1])
        torch.cuda.synchronize()

        for bs in reversed(tqdm(self.graph_bs)):
            graph = torch.cuda.CUDAGraph()
            context.use_decode_graph = True
            context.block_tables = block_tables[:bs]
            context.context_lens = context_lens[:bs]
            # capture 阶段 positions 全 0，cos/sin 切片为 position 0
            # （频率=0 的 cos=1, sin=0），RoPE 退化为恒等映射。
            cos[:bs] = self._cos_table[positions[:bs]]
            sin[:bs] = self._sin_table[positions[:bs]]
            context.cos = cos[:bs]
            context.sin = sin[:bs]
            context.position_ids = positions[:bs]
            context.slot_mapping = slot_mappings[:bs]
            with torch.cuda.graph(graph, self.graph_pool):
                cos[:bs] = self._cos_table[positions[:bs]]
                sin[:bs] = self._sin_table[positions[:bs]]
                outputs[:bs] = self.model.forward_hidden(input_ids[:bs])
                logits[:bs] = self.model.compute_logits(outputs[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph

            torch.cuda.synchronize()
        # 使用锁页内存加快H2D的复制速度
        host_stage = {
            name: torch.empty(max_bs, dtype=torch.int32, pin_memory=True)
            for name in (
                "input_ids",
                "context_lens",
                "position_ids",
                "slot_mapping",
            )
        }
        host_stage["block_tables"] = torch.empty(
            (max_bs, max_num_blocks), dtype=torch.int32, pin_memory=True
        )
        self._host_stage = host_stage
        self._host_np = {k: t.numpy() for k, t in host_stage.items()}

        return {
            "input_ids": input_ids,
            "block_tables": block_tables,
            "context_lens": context_lens,
            "sin": sin,
            "cos": cos,
            "position_ids": positions,
            "slot_mapping": slot_mappings,
            "outputs": outputs,
            "logits": logits,
        }
