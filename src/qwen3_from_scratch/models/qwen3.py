import torch
from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.models.common import assign
from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.models.parameter_loader import ParameterLoader


class Qwen3(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.tok_embd = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=None
        )
        self.output_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.final_norm = ComponentFactory.create(
            "norm", config=config, dim=config.hidden_size, name="model.norm"
        )
        self.rope = ComponentFactory.create("rope", config=config)
        self.trf_blocks = nn.ModuleList(
            [
                ComponentFactory.create(
                    "decoder_layer",
                    config=config,
                    name=f"model.layers.{i}",
                    layer_idx=i,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def load_state(self, loader: ParameterLoader):
        for i, layer in enumerate(self.trf_blocks):
            layer.load_state(loader)
        self.final_norm.load_state(loader)
        self.tok_embd.weight = assign(
            self.tok_embd.weight, loader.get("model.embed_tokens.weight")
        )
        self.output_head.weight = assign(
            self.output_head.weight, loader.get("lm_head.weight")
        )

    def forward(self, idx: torch.Tensor, context: ModelContext):
        tok_embd = self.tok_embd(idx)
        context.position_ids = torch.arange(
            context.cache_position,
            context.cache_position + idx.shape[1],
            dtype=torch.long,
            device=tok_embd.device,
        ).unsqueeze(0)
        context.position_embeddings = self.rope.build_cos_sin_embed(
            context.dtype, context.position_ids
        )
        x = tok_embd
        for layer in self.trf_blocks:
            x = layer(x, context)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
