import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Union

from transformers import Qwen3Config

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
    tie_word_embeddings: bool = False

    num_key_value_heads: int = 8
    num_attention_heads: int = 16
    head_dim: int = 128
    intermediate_size: int = 4096
    num_experts:int = 0
    num_experts_per_token: int = 0

    norm_type: NORM_TYPE = "rms_norm"
    norm_params: dict = field(default_factory=lambda :{"eps": 1e-5})

    pos_embed_type: POS_EMBED_TYPE = "rope"
    pos_embed_params: dict = field(default_factory=lambda :{"rope_theta": 100000, "rope_type": "neox"})

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

    def to_transformers_config(self) -> Qwen3Config:
        """转换为 transformers Qwen3Config，保证与 ModelConfig 一一对应。"""
        return Qwen3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            hidden_act=self.hidden_act,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            num_key_value_heads=self.num_key_value_heads,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            rms_norm_eps=self.norm_params["eps"],
            rope_parameters={
                "rope_theta": self.pos_embed_params["rope_theta"],
                "rope_type": "default",
            },
        )


@dataclass
class GenerationConfig:
    bos_token_id: int = 151643
    pad_token_id: int = 151643
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False

    @classmethod
    def load_from_file(cls, file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)
            return cls(
                bos_token_id=data.get("bos_token_id", 151643),
                pad_token_id=data.get("pad_token_id", 151643),
                temperature=data.get("temperature", 1.0),
                top_k=data.get("top_k", 0),
                top_p=data.get("top_p", 1.0),
                do_sample=data.get("do_sample", False),
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
            num_experts=data.get("num_experts", 0),
            num_experts_per_token=data.get("num_experts_per_tok", 0),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
        )
