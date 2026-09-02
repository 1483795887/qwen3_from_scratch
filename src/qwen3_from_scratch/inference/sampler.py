import torch
import torch.nn as nn


class Sampler(nn.Module):
    """采样器基类：从 logits [B, vocab] → token ids [B, 1]。

    所有采样策略（top_k 过滤、temperature 缩放、softmax、multinomial/argmax）
    封装在子类的 forward 内部，引擎不感知采样细节。
    替换策略 = 替换 Sampler 实例。
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GreedySampler(Sampler):
    """贪婪解码：argmax。"""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)


class TemperatureSampler(Sampler):
    """温度采样：temperature 缩放 + softmax + multinomial。"""

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class TopKSampler(Sampler):
    """Top-K 采样：top_k 过滤 + temperature + softmax + multinomial。"""

    def __init__(self, top_k: int, temperature: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        min_val = top_logits[:, -1].unsqueeze(-1)
        # masked_fill 而非 torch.where(-inf 张量)：标量经 kernel 参数传入，
        # 不会在 CUDA graph 捕获期间触发 pageable H2D 拷贝（会作废 capture）
        top_logits = top_logits.masked_fill(top_logits < min_val, float("-inf"))
        probs = torch.softmax(top_logits / self.temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(1, sampled)
