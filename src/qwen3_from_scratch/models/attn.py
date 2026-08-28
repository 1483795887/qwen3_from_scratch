import math

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.inference.context import (
    get_forward_context,
)
from qwen3_from_scratch.inference.kv_cache.paged_cache import PagedKVCache


@ComponentFactory.register("attn", "base")
class TorchGQA(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.n_head_embed = config.head_dim

    def forward(self, q, k, v):
        is_causal = q.shape[2] > 1
        return scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            enable_gqa=True,
            scale=self.n_head_embed**-0.5,
        )


def group_matmul(mat1, mat2):
    assert mat1.shape[-1] == mat2.shape[-2]
    batch, h1, s1, d1 = mat1.shape
    _, h2, d2, s2 = mat2.shape
    assert d1 == d2
    assert h1 % h2 == 0
    group = h1 // h2
    mat1_reshaped = mat1.reshape(batch, h2, group, s1, d1)
    mat2_reshaped = mat2.unsqueeze(2)
    return torch.matmul(mat1_reshaped, mat2_reshaped).reshape(
        batch, h1, s1, s2
    )


@ComponentFactory.register("attn", "py_flash_attn")
class PyFlashAttention(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__()
        self.is_causal = kwargs.get("is_causal", True)
        self.n_head_embed = config.head_dim
        self.q_tile_size = int(kwargs.get("q_tile_size", 64))
        self.k_tile_size = int(kwargs.get("k_tile_size", 64))

    def forward(self, q, k, v):
        # BxHxSxD
        batch_size, head_q, seq_len_q = q.shape[:3]
        seq_len_k = k.shape[2]
        scale = self.n_head_embed**-0.5
        output_shape = (batch_size, head_q, seq_len_q, self.n_head_embed)
        output = torch.zeros(output_shape, device=q.device)
        m = torch.ones((batch_size, head_q, seq_len_q, 1), device=q.device) * (
            -torch.inf
        )
        sum_exps = torch.zeros(
            (batch_size, head_q, seq_len_q, 1), device=q.device
        )
        for j in range(0, seq_len_k, self.k_tile_size):
            k_end = min(j + self.k_tile_size, seq_len_k)
            k_slice = slice(j, k_end)
            k_tile = k[:, :, k_slice].transpose(-2, -1)
            v_tile = v[:, :, k_slice]

            for i in range(0, seq_len_q, self.q_tile_size):
                q_end = min(i + self.q_tile_size, seq_len_q)
                q_slice = slice(i, q_end)
                q_tile = q[:, :, q_slice]
                # BxHxS1xS2
                attn = group_matmul(q_tile, k_tile) * scale
                if self.is_causal:
                    mask = (
                        torch.arange(i, q_end).unsqueeze(-1).to(q.device)
                    ) < (torch.arange(j, k_end).unsqueeze(0).to(q.device))
                    attn = attn.masked_fill(
                        mask.unsqueeze(0).unsqueeze(0), -torch.inf
                    )
                    attn = attn.masked_fill(mask, -torch.inf)
                m_old = m[:, :, q_slice]
                o_old = output[:, :, q_slice]
                s_old = sum_exps[:, :, q_slice]

                m_new = torch.maximum(
                    m_old,
                    attn.max(dim=-1, keepdim=True).values,
                )
                exp_attn = torch.exp(attn - m_new)
                scale_max_diff = torch.exp(m_old - m_new)
                s_new = s_old * scale_max_diff + exp_attn.sum(
                    dim=-1, keepdim=True
                )

                output[:, :, q_slice] = (
                    group_matmul(exp_attn, v_tile)
                    + o_old * scale_max_diff * s_old
                ) / torch.clamp(s_new, min=1e-10)
                m[:, :, q_slice] = m_new
                sum_exps[:, :, q_slice] = s_new
        return output


def create_causal_attention_mask(seq_len, device, dtype):
    """
    为 eager_attention_forward 生成纯因果掩码
    Args:
        seq_len: 序列长度（如 128）
        device: 设备（如 "cuda"）
        dtype: 数据类型（如 torch.float16）
    Returns:
        attention_mask: 形状 [1,1,seq_len,seq_len]，下三角为0，上三角为-inf
    """
    # 1. 生成下三角布尔掩码（True 表示有效位置）
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )
    # 2. 转换为数值掩码：有效位置=0.0，无效位置=-inf
    attention_mask = torch.zeros_like(causal_mask, dtype=dtype)
    attention_mask = attention_mask.masked_fill(
        ~causal_mask, torch.finfo(dtype).min
    )  # 用finfo.min避免溢出
    # 3. 扩展维度到 [1,1,seq_len,seq_len]（适配注意力分数的维度）
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    return attention_mask


@ComponentFactory.register("attn", "my_op")
class MyAttn(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head_dim = config.head_dim

    def forward(self, q, k, v):
        if q.is_cuda:
            from qwen3_from_scratch.kernels.triton.attn import (
                scaled_dot_production,
            )

            return scaled_dot_production(q, k, v, is_causal=True)
        return self.cpu_forward(q, k, v, is_causal=True)

    def decode(self, q, k, v):
        return scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            enable_gqa=True,
            scale=self.n_head_dim**-0.5,
        )

    def cpu_forward(self, q, k, v, is_causal: bool = True):
        batch_size, head_q, seq_len_q, head_dim = q.shape
        head_kv = k.shape[1]

        # 计算缩放因子
        scale = self.n_head_dim**-0.5

        # GQA: 将q的head分组，每组对应一个kv head
        assert head_q % head_kv == 0, (
            f"head_q ({head_q}) must be divisible by head_kv ({head_kv})"
        )
        n_groups = head_q // head_kv

        # 重塑q为 [batch, head_kv, n_groups, seq_len_q, head_dim]
        q_reshaped = q.reshape(
            batch_size, head_kv, n_groups, seq_len_q, head_dim
        )

        # 扩展k和v的维度以匹配q的分组 [batch, head_kv, 1, seq_len_k, head_dim]
        k_expanded = k.unsqueeze(2)
        v_expanded = v.unsqueeze(2)

        # 计算注意力分数: [batch, head_kv, n_groups, seq_len_q, seq_len_k]
        scores = torch.matmul(q_reshaped, k_expanded.transpose(-2, -1)) * scale

        # 应用因果掩码
        seq_len_k = k.shape[2]
        if seq_len_q > 1 and is_causal:
            # 创建因果掩码
            causal_mask = torch.tril(
                torch.ones(
                    seq_len_q, seq_len_k, device=q.device, dtype=torch.bool
                )
            )
            scores = scores.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # 计算输出: [batch, head_kv, n_groups, seq_len_q, head_dim]
        out = torch.matmul(attn_weights, v_expanded)

        # 重塑回原始形状 [batch, head_q, seq_len_q, head_dim]
        out = out.reshape(batch_size, head_q, seq_len_q, head_dim)

        return out


@ComponentFactory.register("attn", "my_op_flash")
class MyAttnFlash(MyAttn):
    def forward(self, q, k, v):
        if q.is_cuda:
            from qwen3_from_scratch.kernels.triton.attn import flash_attention

            return flash_attention(q, k, v, is_causal=True)
        return self.cpu_forward(q, k, v, is_causal=True)


@ComponentFactory.register("attn", "paged_attn_torch")
class TorchPagedAttn(MyAttn):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(config)
        self.layer_idx: int = kwargs.get("layer_idx")

    def forward(self, q, k, v):
        """
        q,k,v 为 BHSD格式
        KV肯定是最大长度，所以以它为完整的N，Q则为最后M个
        """
        context = get_forward_context()
        assert context.use_cache
        batch_size, num_heads_q, seq_len_q, hidden_dim = q.shape
        assert k.shape[-1] == hidden_dim
        assert k.shape == v.shape
        _, num_heads_k, seq_len_kv, _ = k.shape
        assert seq_len_kv >= seq_len_q
        generated_len = seq_len_kv - seq_len_q
        assert num_heads_q % num_heads_k == 0
        groups = num_heads_q // num_heads_k

        TILE_SIZE_N = 32
        TILE_SIZE_M = 32
        assert TILE_SIZE_N % context.block_size == 0
        scale = math.sqrt(1.0 / hidden_dim)
        batch_size = q.shape[0]
        output = torch.zeros_like(q)
        kv_cache = context.kv_cache
        assert isinstance(context.kv_cache, PagedKVCache)
        k_cache, v_cache = kv_cache.get(self.layer_idx)

        for b in range(batch_size):
            for h in range(num_heads_q):
                h_kv = h // groups
                for m in range(0, seq_len_q, TILE_SIZE_M):
                    curr_m_span = min(TILE_SIZE_M, seq_len_q - m)
                    sub_q = q[b, h, m : m + curr_m_span]
                    dominator = torch.zeros(
                        (curr_m_span, 1), dtype=torch.float32, device=q.device
                    )
                    max_val = (
                        torch.zeros(
                            (curr_m_span, 1),
                            dtype=torch.float32,
                            device=q.device,
                        )
                        - torch.inf
                    )
                    curr_output = torch.zeros_like(sub_q)
                    m_idx = torch.arange(
                        m + generated_len,
                        m + curr_m_span + generated_len,
                        device=q.device,
                        dtype=torch.int32,
                    )
                    for n in range(0, seq_len_kv, TILE_SIZE_N):
                        curr_n_span = min(TILE_SIZE_N, seq_len_kv - n)
                        n_idx = torch.arange(
                            n,
                            n + curr_n_span,
                            device=q.device,
                            dtype=torch.int32,
                        )
                        sub_k, sub_v = self._load_kv(
                            b,
                            n,
                            curr_n_span,
                            h_kv,
                            self.n_head_dim,
                            q.device,
                            q.dtype,
                            k_cache,
                            v_cache,
                            kv_cache.page_size,
                            context.block_tables,
                        )
                        attn = sub_q @ sub_k.t() * scale
                        # 没有 causal 就没有用 KVCache 的必要
                        attn += torch.where(
                            m_idx[:, None] < n_idx[None, :], -float("inf"), 0.0
                        )

                        curr_max = torch.maximum(
                            torch.max(attn, dim=-1, keepdim=True)[0], max_val
                        )
                        attn_score = torch.exp(attn - curr_max)
                        curr_dominator = torch.sum(
                            attn_score, dim=-1, keepdim=True
                        )

                        factor = torch.exp(max_val - curr_max)
                        dominator = dominator * factor + curr_dominator
                        curr_output = factor * curr_output + attn_score @ sub_v
                        max_val = curr_max
                    output[b, h, m : m + curr_m_span] = curr_output / dominator
        return output

    def _load_kv(
        self,
        b: int,
        n_start: int,
        n_size: int,
        h_kv: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        k_cache,
        v_cache,
        page_size: int,
        block_tables,
    ):
        k = torch.empty((n_size, head_dim), device=device, dtype=dtype)
        v = torch.empty((n_size, head_dim), device=device, dtype=dtype)

        num_blocks = (n_size + page_size - 1) // page_size
        for i in range(num_blocks):
            block_idx = (n_start // page_size) + i
            block_id = block_tables[b][block_idx].item()
            offset = i * page_size
            size = min(page_size, n_size - offset)

            k[offset : offset + size] = k_cache[block_id, :size, h_kv]
            v[offset : offset + size] = v_cache[block_id, :size, h_kv]

        return k, v


@ComponentFactory.register("attn", "var_len_paged_attn")
class VarLenPagedAttn(MyAttn):
    def __init__(self, config: ModelConfig, layer_idx: int, **kwargs) -> None:
        super().__init__(config)
        self.layer_idx: int = layer_idx

    def forward(self, q, k, v):
        """
        q,k,v 为 SHD格式
        这里HD在一起，和其他的H在前面，S在后面不一样
        """
        context = get_forward_context()
        assert context.use_cache
        _, num_heads_q, hidden_dim = q.shape
        assert k.shape[-1] == hidden_dim
        assert k.shape == v.shape
        _, num_heads_k, _ = k.shape
        assert num_heads_q % num_heads_k == 0

        if q.is_cuda:
            assert (
                context.cum_seq_lens_kv.shape == context.cum_seq_lens_q.shape
            )
            assert isinstance(context.kv_cache, PagedKVCache)
            from qwen3_from_scratch.kernels.triton.paged_attn import (
                flash_attn_decode_func,
                flash_attn_varlen_func,
            )

            k_cache, v_cache = context.kv_cache.get(self.layer_idx)
            scale = hidden_dim**-0.5

            if context.use_decode_graph:
                return flash_attn_decode_func(
                    q,
                    k_cache,
                    v_cache,
                    context.context_lens,
                    scale,
                    context.block_tables,
                )
            assert context.cum_seq_lens_q.shape[0] > 0
            cum_q = context.cum_seq_lens_q
            cum_kv = context.cum_seq_lens_kv
            if context.max_seqlen_q > 0:
                # 引擎侧已每步算好（build_context_*），免去两次 D2H 同步
                max_seqlen_q = context.max_seqlen_q
                max_seqlen_k = context.max_seqlen_k
            else:
                max_seqlen_q = int((cum_q[1:] - cum_q[:-1]).max())
                max_seqlen_k = int((cum_kv[1:] - cum_kv[:-1]).max())
            return flash_attn_varlen_func(
                q,
                k_cache,
                v_cache,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cum_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_k=cum_kv,
                softmax_scale=scale,
                causal=True,
                block_table=context.block_tables,
            )

        groups = num_heads_q // num_heads_k

        TILE_SIZE_N = 32
        TILE_SIZE_M = 32
        assert TILE_SIZE_N % context.block_size == 0
        scale = math.sqrt(1.0 / hidden_dim)
        assert context.cum_seq_lens_kv.shape == context.cum_seq_lens_q.shape
        assert context.cum_seq_lens_q.shape[0] > 0
        num_seqs = context.cum_seq_lens_q.shape[0] - 1
        output = torch.zeros_like(q)
        kv_cache = context.kv_cache
        assert isinstance(context.kv_cache, PagedKVCache)
        k_cache, v_cache = kv_cache.get(self.layer_idx)

        for b in range(num_seqs):
            for h in range(num_heads_q):
                h_kv = h // groups
                m_start, m_end = (
                    int(context.cum_seq_lens_q[b]),
                    int(context.cum_seq_lens_q[b + 1]),
                )
                n_start, n_end = (
                    int(context.cum_seq_lens_kv[b]),
                    int(context.cum_seq_lens_kv[b + 1]),
                )
                seq_len_q = m_end - m_start
                seq_len_kv = n_end - n_start
                assert seq_len_kv >= seq_len_q
                generated_len = seq_len_kv - seq_len_q
                # 采用 [start, end) 的方式
                for m in range(m_start, m_end, TILE_SIZE_M):
                    q_idx_start = m - m_start + generated_len
                    curr_m_span = min(TILE_SIZE_M, m_end - m)
                    sub_q = q[m : m + curr_m_span, h]
                    dominator = torch.zeros(
                        (curr_m_span, 1), dtype=torch.float32, device=q.device
                    )
                    max_val = (
                        torch.zeros(
                            (curr_m_span, 1),
                            dtype=torch.float32,
                            device=q.device,
                        )
                        - torch.inf
                    )
                    curr_output = torch.zeros_like(sub_q)
                    m_idx = torch.arange(
                        q_idx_start,
                        q_idx_start + curr_m_span,
                        device=q.device,
                        dtype=torch.int32,
                    )
                    for n in range(n_start, n_end, TILE_SIZE_N):
                        kv_idx_start = n - n_start
                        curr_n_span = min(TILE_SIZE_N, n_end - n)
                        n_idx = torch.arange(
                            kv_idx_start,
                            kv_idx_start + curr_n_span,
                            device=q.device,
                            dtype=torch.int32,
                        )
                        sub_k, sub_v = self._load_kv(
                            b,
                            kv_idx_start,
                            curr_n_span,
                            h_kv,
                            self.n_head_dim,
                            q.device,
                            q.dtype,
                            k_cache,
                            v_cache,
                            kv_cache.page_size,
                            context.block_tables,
                        )
                        attn = sub_q @ sub_k.t() * scale
                        # 没有 causal 就没有用 KVCache 的必要
                        attn += torch.where(
                            m_idx[:, None] < n_idx[None, :], -float("inf"), 0.0
                        )

                        curr_max = torch.maximum(
                            torch.max(attn, dim=-1, keepdim=True)[0], max_val
                        )
                        attn_score = torch.exp(attn - curr_max)
                        curr_dominator = torch.sum(
                            attn_score, dim=-1, keepdim=True
                        )

                        factor = torch.exp(max_val - curr_max)
                        dominator = dominator * factor + curr_dominator
                        curr_output = (
                            factor * curr_output
                            + attn_score.to(sub_v.dtype) @ sub_v
                        )

                        max_val = curr_max
                    output[m : m + curr_m_span, h] = curr_output / dominator
        return output

    def _load_kv(
        self,
        b: int,
        n_start: int,
        n_size: int,
        h_kv: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        k_cache,
        v_cache,
        page_size: int,
        block_tables,
    ):
        k = torch.empty((n_size, head_dim), device=device, dtype=dtype)
        v = torch.empty((n_size, head_dim), device=device, dtype=dtype)

        num_blocks = (n_size + page_size - 1) // page_size
        for i in range(num_blocks):
            block_idx = (n_start // page_size) + i
            block_id = block_tables[b][block_idx].item()
            offset = i * page_size
            size = min(page_size, n_size - offset)

            k[offset : offset + size] = k_cache[block_id, :size, h_kv]
            v[offset : offset + size] = v_cache[block_id, :size, h_kv]

        return k, v
