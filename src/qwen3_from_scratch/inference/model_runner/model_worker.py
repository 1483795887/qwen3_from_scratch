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
            padded, device=self.device, dtype=torch.int32, pin_memory=True
        )
        context.slot_mapping = torch.tensor(
            slot_mapping,
            device=self.device,
            dtype=torch.int32,
            pin_memory=True,
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

    def build_context(self, seqs: list[Sequence]):
        """混合 batch 的统一上下文构建（prefill 分段 + decode 共存）。"""
        context = get_forward_context()
        is_pure_decode = all(not seq.is_prefill for seq in seqs)
        positions = []
        device = self.device

        if (
            torch.cuda.is_available()
            and is_pure_decode
            and len(seqs) < ModelWorker.CUDA_GRAPH_MAX_LEN
        ):
            context.use_decode_graph = True
            context.context_lens = torch.tensor(
                [len(seq) for seq in seqs],
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)
            # decode 每条序列只 query 最后一个 token 的位置
            for seq in seqs:
                positions.extend(self._query_positions(seq))
            cum_q = [0]
            cum_kv = [0]
            for seq in seqs:
                cum_q.append(cum_q[-1] + seq.num_tokens)
                cum_kv.append(cum_kv[-1] + self._kv_len(seq))
        else:
            context.use_decode_graph = False
            cum_seq_lens_q = [0]
            cum_seq_lens_kv = [0]
            for seq in seqs:
                positions.extend(self._query_positions(seq))
                cum_seq_lens_q.append(cum_seq_lens_q[-1] + seq.num_tokens)
                cum_seq_lens_kv.append(cum_seq_lens_kv[-1] + self._kv_len(seq))

            context.cum_seq_lens_kv = torch.tensor(
                cum_seq_lens_kv,
                dtype=torch.int32,
                pin_memory=True,
            ).to(device=self.device, non_blocking=True)
            context.cum_seq_lens_q = torch.tensor(
                cum_seq_lens_q,
                dtype=torch.int32,
                pin_memory=True,
            ).to(device=self.device, non_blocking=True)
        context.position_ids = torch.tensor(
            positions, dtype=torch.int32, pin_memory=True
        ).to(self.device, non_blocking=True)
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
        self.build_context(seqs)
        inputs = self.build_inputs(seqs)

        # 每条序列取自身 query 区间最后一个 token 的 logits
        context = get_forward_context()
        if context.use_decode_graph:
            logits = self.replay_cuda_graph(inputs)
        else:
            indices = context.cum_seq_lens_q[1:] - 1
            hidden = self.model.forward_hidden(inputs)
            # 只用算最后一个词元的logits
            logits = self.model.compute_logits(hidden[indices])  # [B, vocabs]

        next_ids = self.sampler(logits)
        return next_ids[:, 0].tolist()  # length B

    @torch.inference_mode
    def replay_cuda_graph(self, inputs: torch.Tensor):
        bs = inputs.size(0)
        context = get_forward_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = inputs
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
            context.block_tables
        )
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["cos"][:bs] = context.cos
        graph_vars["sin"][:bs] = context.sin
        graph_vars["position_ids"][:bs] = context.position_ids
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

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
        input_ids = torch.zeros(
            max_bs, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        positions = torch.zeros(
            max_bs, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = torch.zeros(
            max_bs,
            max_num_blocks,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        slot_mappings = (
            torch.zeros(max_bs, dtype=torch.int32, pin_memory=True)
        ).cuda(non_blocking=True)
        # capture 阶段 slot 全填 -1：KV update 核遇到 slot<0 直接 return，
        # 避免 capture 把 token-0 的 k/v 写进真实 KV cache 的前 max_bs 个槽位
        # （这些垃圾数据在块复用/前缀命中场景可能被后续 decode 读到）。
        slot_mappings.fill_(-1)
        context_lens = torch.zeros(
            max_bs, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        outputs = torch.zeros(
            max_bs, model_config.hidden_size, pin_memory=True, dtype=self.dtype
        ).cuda(non_blocking=True)
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
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        context = get_forward_context()
        logger.info("记录 Cuda 图中")
        # 预热一次 forward：让 cuBLAS handle / cublasLt 等 lazy 库在 default stream
        # 上完成初始化；进入 capture 后 cuBLAS 不会再次创建 handle，
        # 避免 F.linear 触发 cudaErrorStreamCaptureInvalidated。
        # 注意必须走 forward_hidden（与 capture 的入口一致），不能走 self.model(...)
        # —— 后者还含 lm_head，算子集合比 capture 范围大，预热不全。
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
                outputs[:bs] = self.model.forward_hidden(input_ids[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph

            torch.cuda.synchronize()
        return {
            "input_ids": input_ids,
            "block_tables": block_tables,
            "context_lens": context_lens,
            "sin": sin,
            "cos": cos,
            "position_ids": positions,
            "slot_mapping": slot_mappings,
            "outputs": outputs,
        }
