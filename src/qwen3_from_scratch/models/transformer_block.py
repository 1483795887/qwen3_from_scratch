from torch import nn

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig
from qwen3_from_scratch.inference.context import ModelContext
from qwen3_from_scratch.models.parameter_loader import ParameterLoader


@ComponentFactory.register("decoder_layer", "base")
class PythonTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, name: str, layer_idx: int, **kwargs):
        super().__init__()
        self.name = name
        self.layer_idx = layer_idx
        self.config = config
        self.self_attn = ComponentFactory.create("self_attn", config, name=f'{self.name}.self_attn',
                                                 layer_idx=layer_idx)
        self.input_layernorm = ComponentFactory.create("norm", config, name=f'{self.name}.input_layernorm',
                                                       dim=config.hidden_size)
        self.post_attention_layernorm = ComponentFactory.create("norm", config,
                                                                name=f'{self.name}.post_attention_layernorm',
                                                                dim=config.hidden_size)
        self.mlp = ComponentFactory.create("mlp", config, name=f'{self.name}.mlp')

    def forward(self, x, context: ModelContext):
        inp_x = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, context)
        ffn_inp_x = x + inp_x
        x = ffn_inp_x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return x + ffn_inp_x

    def load_state(self, loader: ParameterLoader):
        self.self_attn.load_state(loader)
        self.input_layernorm.load_state(loader)
        self.post_attention_layernorm.load_state(loader)
        self.mlp.load_state(loader)
