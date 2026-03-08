import torch.nn as nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.models.common import assign
from qwen3_from_scratch.models.parameter_loader import ParameterLoader

activation_map = {"silu": nn.SiLU}


@ComponentFactory.register("mlp", "base")
class PythonFeedback(nn.Module):

    def __init__(self, config: ModelConfig, name: str, **kwargs) -> None:
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.activation = activation_map[config.hidden_act]()
        self.name = name

    def forward(self, x):
        embed_up = self.up_proj(x)
        embed_gate = self.activation(self.gate_proj(x))
        return self.down_proj(embed_up * embed_gate)

    def load_state(self, loader: ParameterLoader):
        self.down_proj.weight = assign(
            self.down_proj.weight, loader.get(f"{self.name}.down_proj.weight")
        )
        self.up_proj.weight = assign(
            self.up_proj.weight, loader.get(f"{self.name}.up_proj.weight")
        )
        self.gate_proj.weight = assign(
            self.gate_proj.weight, loader.get(f"{self.name}.gate_proj.weight")
        )