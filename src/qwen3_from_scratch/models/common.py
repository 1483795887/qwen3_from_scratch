import torch


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(right.to(left.device))
