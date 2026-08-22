import torch
from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.models.common import assign
from qwen3_from_scratch.models.parameter_loader import ParameterLoader


class Qwen3Model(nn.Module):
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

        self.final_norm = ComponentFactory.create(
            "norm", config=config, dim=config.hidden_size, name="model.norm"
        )
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

    def forward(self, idx: torch.Tensor):
        tok_embd = self.tok_embd(idx)
        x = tok_embd
        for layer in self.trf_blocks:
            x = layer(x)
        x = self.final_norm(x)
        return x


class Qwen3(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, **kwargs)

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def load_state(self, loader: ParameterLoader):
        self.model.load_state(loader)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embd.weight
        else:
            self.lm_head.weight = assign(
                self.lm_head.weight, loader.get("lm_head.weight")
            )

    def forward(self, idx: torch.Tensor):
        hidden = self.forward_hidden(idx)
        return self.compute_logits(hidden)

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)
