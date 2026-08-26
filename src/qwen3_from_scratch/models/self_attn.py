from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.inference.context import get_forward_context
from qwen3_from_scratch.models.common import assign
from qwen3_from_scratch.models.parameter_loader import ParameterLoader


@ComponentFactory.register("self_attn", "base")
class SelfAttention(nn.Module):
    def __init__(
        self, config: ModelConfig, name: str, layer_idx: int = 0, **kwargs
    ):
        super().__init__()
        hidden_size = config.hidden_size
        kv_embed_size = config.head_dim * config.num_key_value_heads
        q_embed_size = config.head_dim * config.num_attention_heads
        self.name = name
        self.layer_idx = layer_idx
        self.config = config
        self.gqa = ComponentFactory.create("attn", config, layer_idx=layer_idx)
        self.rope = ComponentFactory.create("rope", config)

        self.k_proj = nn.Linear(hidden_size, kv_embed_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_embed_size, bias=False)
        self.q_proj = nn.Linear(hidden_size, q_embed_size, bias=False)
        self.o_proj = nn.Linear(q_embed_size, hidden_size, bias=False)
        self.k_norm = ComponentFactory.create(
            "norm", config, name=f"{self.name}.k_norm"
        )
        self.q_norm = ComponentFactory.create(
            "norm", config, name=f"{self.name}.q_norm"
        )

    def forward(self, x):
        ctx = get_forward_context()
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)
        dtype = x.dtype
        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        q = self.rope(q).to(dtype)
        k = self.rope(k).to(dtype)
        if ctx.use_cache:
            k, v = ctx.kv_cache.update(
                k.transpose(1, 2),
                v.transpose(1, 2),
                self.layer_idx,
                ctx.cache_position,
            )
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        o = self.gqa(q, k, v).transpose(1, 2).reshape(*input_shape, -1)
        o = self.o_proj(o)
        return o

    def load_state(self, loader: ParameterLoader):
        self.q_proj.weight = assign(
            self.q_proj.weight, loader.get(f"{self.name}.q_proj.weight")
        )
        self.k_proj.weight = assign(
            self.k_proj.weight, loader.get(f"{self.name}.k_proj.weight")
        )
        self.v_proj.weight = assign(
            self.v_proj.weight, loader.get(f"{self.name}.v_proj.weight")
        )
        self.o_proj.weight = assign(
            self.o_proj.weight, loader.get(f"{self.name}.o_proj.weight")
        )
        self.q_norm.load_state(loader)
        self.k_norm.load_state(loader)


@ComponentFactory.register("self_attn", "my_op")
class FusedSelfAttention(nn.Module):
    def __init__(
        self, config: ModelConfig, name: str, layer_idx: int = 0, **kwargs
    ):
        super().__init__()
        self.name = name
        self.layer_idx = layer_idx
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.groups = self.num_heads // self.num_kv_heads
        hidden_size = config.hidden_size
        eps = config.norm_params.get("eps", 1e-5)
        self.eps = eps

        qkv_out = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(hidden_size, qkv_out, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, hidden_size, bias=False
        )
        # QK-Norm 权重融合为一个 (2, head_dim)：Q 用 [0]，K 用 [1]
        self.fused_norm_weight = nn.Parameter(torch.ones(2, self.head_dim))

    def _get_cos_sin(self, x: torch.Tensor):
        """直接读 ctx.cos / ctx.sin（引擎侧每步预取，与 CUDA graph capture 兼容）。

        路径固定、不再走 .cpu() 索引；要求 ctx 上 cos/sin 已就绪。
        """
        ctx = get_forward_context()
        assert ctx.cos is not None and ctx.sin is not None, (
            "context.cos / context.sin must be set before forward"
        )
        return ctx.cos, ctx.sin

    def _forward_pytorch(self, x, residual=None):
        ctx = get_forward_context()
        B, S, _ = x.shape
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        cos, sin = self._get_cos_sin(x)

        q_end = H_q * D
        k_end = q_end + H_kv * D
        qkv = torch.nn.functional.linear(x, self.qkv_proj.weight)

        q = qkv[:, :, :q_end].reshape(B, S, H_q, D)
        k = qkv[:, :, q_end:k_end].reshape(B, S, H_kv, D)
        v = qkv[:, :, k_end:].reshape(B, S, H_kv, D)

        q_gamma, k_gamma = self.fused_norm_weight.unbind()
        q = torch.nn.functional.rms_norm(q, (D,), q_gamma, self.eps)
        k = torch.nn.functional.rms_norm(k, (D,), k_gamma, self.eps)

        cos_e = cos.view(1, S, 1, D)
        sin_e = sin.view(1, S, 1, D)

        def _rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q = q * cos_e + _rotate_half(q) * sin_e
        k = k * cos_e + _rotate_half(k) * sin_e

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if ctx.use_cache:
            k, v = ctx.kv_cache.update(
                k.transpose(1, 2),
                v.transpose(1, 2),
                self.layer_idx,
                ctx.cache_position,
            )
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        k_exp = (
            k.unsqueeze(2)
            .expand(-1, -1, self.groups, -1, -1)
            .reshape(B, H_q, -1, D)
        )
        v_exp = (
            v.unsqueeze(2)
            .expand(-1, -1, self.groups, -1, -1)
            .reshape(B, H_q, -1, D)
        )
        scale = D**-0.5
        attn = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) * scale
        if S > 1:
            mask = torch.tril(torch.ones(S, attn.shape[-1], device=x.device))
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(x.dtype)
        o = torch.matmul(attn, v_exp)

        o = o.transpose(1, 2).reshape(B, S, -1)
        o = torch.nn.functional.linear(o, self.o_proj.weight)
        if residual is not None:
            o = o + residual
        return o

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ):
        ctx = get_forward_context()
        if x.device.type == "cpu":
            return self._forward_pytorch(x, residual)

        B, S, _ = x.shape
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        cos, sin = self._get_cos_sin(x)

        qkv = self.qkv_proj(x)
        g = self.fused_norm_weight
        from qwen3_from_scratch.kernels.triton.self_attn import (
            fused_qk_norm_rope,
        )

        fused_qk_norm_rope(qkv, g, cos, sin, D, self.groups, self.eps)

        q_end = H_q * D
        k_end = q_end + H_kv * D
        q = qkv[:, :, :q_end].contiguous().view(B, S, H_q, D).transpose(1, 2)
        k = (
            qkv[:, :, q_end:k_end]
            .contiguous()
            .view(B, S, H_kv, D)
            .transpose(1, 2)
        )
        v = qkv[:, :, k_end:].contiguous().view(B, S, H_kv, D).transpose(1, 2)

        if ctx.use_cache:
            k, v = ctx.kv_cache.update(
                k.transpose(1, 2),
                v.transpose(1, 2),
                self.layer_idx,
                ctx.cache_position,
            )
            k = k.contiguous().transpose(1, 2)
            v = v.contiguous().transpose(1, 2)

        from qwen3_from_scratch.kernels.triton.attn import flash_attention

        o = flash_attention(q, k, v, is_causal=S > 1)

        o = o.transpose(1, 2).reshape(B, S, -1)
        return F.linear(o, self.o_proj.weight, bias=residual)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        head_dim = self.head_dim
        q_end = self.num_heads * head_dim
        q_w, k_w, v_w = self.qkv_proj.weight.split(
            [
                q_end,
                self.num_kv_heads * head_dim,
                self.num_kv_heads * head_dim,
            ],
            dim=0,
        )

        def dst(v):
            return v if keep_vars else v.detach()

        destination[prefix + "q_proj.weight"] = dst(q_w)
        destination[prefix + "k_proj.weight"] = dst(k_w)
        destination[prefix + "v_proj.weight"] = dst(v_w)
        destination[prefix + "o_proj.weight"] = dst(self.o_proj.weight)
        q_norm_weight, k_norm_weight = self.fused_norm_weight.unbind()
        destination[prefix + "q_norm.weight"] = dst(q_norm_weight)
        destination[prefix + "k_norm.weight"] = dst(k_norm_weight)
        return destination

    def load_state(self, loader: ParameterLoader):
        q_w = loader.get(f"{self.name}.q_proj.weight")
        k_w = loader.get(f"{self.name}.k_proj.weight")
        v_w = loader.get(f"{self.name}.v_proj.weight")
        merged = torch.cat([q_w, k_w, v_w], dim=0)
        self.qkv_proj.weight = assign(self.qkv_proj.weight, merged)

        self.o_proj.weight = assign(
            self.o_proj.weight, loader.get(f"{self.name}.o_proj.weight")
        )
        self.fused_norm_weight = assign(
            self.fused_norm_weight,
            torch.stack(
                [
                    loader.get(f"{self.name}.q_norm.weight"),
                    loader.get(f"{self.name}.k_norm.weight"),
                ]
            ),
        )


@ComponentFactory.register("self_attn", "paged_attn")
class PagedSelfAttention(FusedSelfAttention):
    def __init__(
        self, config: ModelConfig, name: str, layer_idx: int = 0, **kwargs
    ):
        super().__init__(config, name, layer_idx, **kwargs)
        from qwen3_from_scratch.models.attn import VarLenPagedAttn

        self.attn = VarLenPagedAttn(config, layer_idx=layer_idx)

    def _forward_pytorch(self, x, residual=None):
        ctx = get_forward_context()
        total_seq_len, hidden_dim = x.shape
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        cos, sin = self._get_cos_sin(x)

        qkv = torch.nn.functional.linear(x, self.qkv_proj.weight).view(
            (total_seq_len, -1, D)
        )
        q, k, v = qkv.split([H_q, H_kv, H_kv], dim=1)

        q_gamma, k_gamma = self.fused_norm_weight.unbind()
        q = torch.nn.functional.rms_norm(q, (D,), q_gamma, self.eps)
        k = torch.nn.functional.rms_norm(k, (D,), k_gamma, self.eps)

        cos_e = cos.unsqueeze(1)
        sin_e = sin.unsqueeze(1)

        def _rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q = q * cos_e + _rotate_half(q) * sin_e
        k = k * cos_e + _rotate_half(k) * sin_e

        ctx.kv_cache.update(k, v, self.layer_idx, ctx.cache_position)

        o = self.attn(q, k, v)
        o = o.reshape(total_seq_len, -1)
        return F.linear(o, self.o_proj.weight, bias=residual)

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ):
        if not x.is_cuda:
            return self._forward_pytorch(x, residual)

        ctx = get_forward_context()
        total_seq_len, hidden_dim = x.shape
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        cos, sin = self._get_cos_sin(x)

        total_out = (H_q + 2 * H_kv) * D
        qkv = self.qkv_proj(x)

        g = self.fused_norm_weight
        from qwen3_from_scratch.kernels.triton.self_attn import (
            fused_qk_norm_rope,
        )

        fused_qk_norm_rope(
            qkv.view(1, total_seq_len, total_out),
            g,
            cos,
            sin,
            D,
            self.groups,
            self.eps,
        )

        q_end = H_q * D
        k_end = q_end + H_kv * D
        q = qkv[:, :q_end].contiguous().view(total_seq_len, H_q, D)
        k = qkv[:, q_end:k_end].contiguous().view(total_seq_len, H_kv, D)
        v = qkv[:, k_end:].contiguous().view(total_seq_len, H_kv, D)

        ctx.kv_cache.update(k, v, self.layer_idx, ctx.cache_position)

        o = self.attn(q, k, v)
        o = o.reshape(total_seq_len, -1)
        return F.linear(o, self.o_proj.weight, bias=residual)
