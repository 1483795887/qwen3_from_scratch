import torch.nn as nn
import torch
import torch.nn.functional as F
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

    def forward(self, x, residual=None):
        embed_up = self.up_proj(x)
        embed_gate = self.activation(self.gate_proj(x))
        o = self.down_proj(embed_up * embed_gate)
        if residual is not None:
            o = o + residual
        return o

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

@ComponentFactory.register("mlp", "my_op")
class MyFeedback(PythonFeedback):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    merged_weight = torch.concat([self.up_proj.weight, self.gate_proj.weight], dim=0)
    self.register_buffer("merged_weight", merged_weight, persistent=False)

  def load_state(self, loader: ParameterLoader):
    super().load_state(loader)
    merged_weight = torch.concat([self.up_proj.weight, self.gate_proj.weight], dim=0)
    self.merged_weight = merged_weight

  def forward(self, x, residual=None):
    if x.is_cuda:
      from qwen3_from_scratch.kernels.triton.feedback import simple_swiglu
      output = torch.empty_like(x)
      simple_swiglu(x, self.merged_weight, self.down_proj.weight, output, residual=residual)
      return output
    return super().forward(x, residual=residual)

@ComponentFactory.register("mlp", "moe")
class MoE(nn.Module):
    def __init__(self, config: ModelConfig, name:str, **kwargs):
        super().__init__()
        self.name = name
        assert config.num_experts > 0 and config.num_experts_per_token >0, "Moe需要设置 num_experts 和 num_experts_per_token"
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        # 处理一下 name
        self.experts = nn.ModuleList([PythonFeedback(config, name + f".experts.{i}", **kwargs) for i in range(config.num_experts)])
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, x, residual=None):
        # 变长输入中直接就是 SxD 转不转无所谓，但 Batch 输入中是 BxSxD ，需要展平，不区分B
        hidden_states = x.reshape(-1, x.shape[-1])
        scores = F.softmax(self.gate(hidden_states), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_token)
        topk_weight /= topk_weight.sum(dim=-1, keepdim=True)

        result = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if not mask.any():
                continue
            token_idx = mask.any(dim=-1).nonzero().flatten()
            weight = topk_weight[mask].view(-1, 1)
            result.index_add_(0, token_idx, (expert(hidden_states[token_idx]) * weight).to(result.dtype))
        o = result.reshape(*x.shape)
        if residual is not None:
            o = o + residual
        return o

    def load_state(self, loader:ParameterLoader):
        self.gate.weight = assign(
            self.gate.weight, loader.get(f'{self.name}.gate.weight')
        )
        for export in self.experts:
            export.load_state(loader)