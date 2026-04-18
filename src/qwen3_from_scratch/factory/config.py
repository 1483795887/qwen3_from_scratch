import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

ACTIVATIONS = Literal["silu"]
NORM_TYPE = Literal["rms_norm"]
POS_EMBED_TYPE = Literal["rope"]
ROPE_TYPE = Literal["normal", "neox"]


@dataclass
class ComponentConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    hidden_act: ACTIVATIONS = "silu"
    num_hidden_layers: int = 28
    max_position_embeddings: int = 40960
    eos_token_id: int = 151645

    num_key_value_heads: int = 8
    num_attention_heads: int = 16
    head_dim: int = 128
    intermediate_size: int = 4096

    norm_type: NORM_TYPE = "rms_norm"
    norm_params: dict = field(default_factory=dict)

    pos_embed_type: POS_EMBED_TYPE = "rope"
    pos_embed_params: dict = field(default_factory=dict)

    self_attn: ComponentConfig = field(
        default_factory=lambda: ComponentConfig("base")
    )
    mlp: ComponentConfig = field(
        default_factory=lambda: ComponentConfig("base")
    )
    norm: ComponentConfig = field(
        default_factory=lambda: ComponentConfig("base")
    )
    attn: ComponentConfig = field(
        default_factory=lambda: ComponentConfig("base")
    )
    rope: ComponentConfig = field(
        default_factory=lambda: ComponentConfig("base")
    )
    decoder_layer: ComponentConfig = field(
        default_factory=lambda: ComponentConfig("base")
    )


def load_from_file(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
        return ModelConfig(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            hidden_act=data["hidden_act"],
            num_hidden_layers=data["num_hidden_layers"],
            max_position_embeddings=data["max_position_embeddings"],
            eos_token_id=data["eos_token_id"],
            num_key_value_heads=data["num_key_value_heads"],
            num_attention_heads=data["num_attention_heads"],
            head_dim=data["head_dim"],
            intermediate_size=data["intermediate_size"],
            norm_type="rms_norm",
            norm_params={"eps": data["rms_norm_eps"]},
            pos_embed_type="rope",
            pos_embed_params={
                "rope_theta": data["rope_theta"],
                "rope_type": "neox",
            },
        )
