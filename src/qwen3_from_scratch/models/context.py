from dataclasses import dataclass

import torch


@dataclass
class PositionEmbeddings:
    cos_embed: torch.Tensor
    sin_embed: torch.Tensor

@dataclass
class ModelContext:
    dtype: torch.dtype = torch.bfloat16
    position_ids: torch.Tensor = None
    position_embeddings: PositionEmbeddings = None
    cache_position: int = 0
