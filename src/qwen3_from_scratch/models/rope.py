import torch
from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.inference.context import (
    ModelContext,
    PositionEmbeddings,
)


@ComponentFactory.register("rope", "base")
class PythonRope(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.head_dim = config.head_dim
        self.base_freq = config.pos_embed_params["rope_theta"]
        self.max_seq_len = config.max_position_embeddings
        self.rope_type = config.pos_embed_params["rope_type"]

    def _rotate_half_neox(self, x: torch.Tensor) -> torch.Tensor:
        """NeoX风格的旋转：前后半段交叉"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _rotate_normal(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

    def build_cos_sin_embed(
        self, dtype, position_ids: torch.Tensor
    ) -> PositionEmbeddings:
        inv_freq = 1.0 / (
            self.base_freq
            ** (
                torch.arange(
                    0, self.head_dim, 2, device=position_ids.device
                ).float()
                / self.head_dim
            )
        ).unsqueeze(0)
        freqs = torch.einsum("bj,bk->bjk", position_ids, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return PositionEmbeddings(emb.cos().to(dtype), emb.sin().to(dtype))

    def forward(self, x: torch.Tensor, context: ModelContext):
        """
        Args:
            x: [batch, num_heads, seq_len, dim]
            context: 上下文数据，包含 position_ids 和 position_embeddings 等
        """
        seq_len = x.shape[2]

        if context.position_ids is None:
            context.position_ids = torch.arange(
                seq_len, device=x.device
            ).unsqueeze(0)
        if context.position_embeddings is None:
            context.position_embeddings = self.build_cos_sin_embed(
                x, context.position_ids
            )
        emb_cos = context.position_embeddings.cos_embed[None, :, :]
        emb_sin = context.position_embeddings.sin_embed[None, :, :]

        # NeoX风格旋转
        if self.rope_type == "neox":
            return (x * emb_cos) + (self._rotate_half_neox(x) * emb_sin)
        elif self.rope_type == "normal":
            return (x * emb_cos) + (self._rotate_normal(x) * emb_sin)
        else:
            raise ValueError(f"Unknown RoPE type: {self.rope_type}")
