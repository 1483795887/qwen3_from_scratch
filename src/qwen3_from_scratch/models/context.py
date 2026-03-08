from dataclasses import dataclass

import torch


@dataclass
class PositionEmbeddings:
    cos_embed: torch.Tensor
    sin_embed: torch.Tensor

@dataclass
class ModelContext:
    position_ids: torch.Tensor = None
    position_embeddings: PositionEmbeddings = None
