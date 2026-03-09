from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.inference.context import ModelContext
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
        self.gqa = ComponentFactory.create("attn", config)
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

    def forward(self, x, context: ModelContext):
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)
        q = self.q_norm(self.q_proj(x).view(hidden_shape).transpose(1, 2))
        k = self.k_norm(self.k_proj(x).view(hidden_shape).transpose(1, 2))
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        q = self.rope(q, context)
        k = self.rope(k, context)
        if context.use_cache:
            k,v = context.kv_cache.update(k, v, self.layer_idx)
        o = (
            self.gqa(q, k, v)
            .transpose(1, 2)
            .reshape(*input_shape, -1)
        )
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
