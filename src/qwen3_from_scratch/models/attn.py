from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig


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
