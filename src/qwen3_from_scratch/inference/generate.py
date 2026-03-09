from typing import Iterable

import torch
import torch.nn as nn

from qwen3_from_scratch.inference.context import ModelContext


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context: ModelContext,
    temperature: float = 1.0,
    top_k=None,
    eos_id=None,
    tokenizer=None,
    device="cuda",
    stream=False,
)->Iterable[str] | str:
    is_prefill = True
    model = model.to(device)
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if is_prefill or not context.use_cache:
                context.cache_position = 0
                logits = model(idx, context=context)
                is_prefill = False
            else:
                context.cache_position = idx.shape[1] - 1
                logits = model(idx[:, -1:], context=context)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
        if tokenizer is not None and context.use_cache:
            if stream:
                yield tokenizer.decode(
                    idx.squeeze(0).tolist()[-1:], skip_special_tokens=False
                )
    if not stream:
        return tokenizer.decode(idx.squeeze(0).tolist(), skip_special_tokens=False)
