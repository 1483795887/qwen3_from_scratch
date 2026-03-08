import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from qwen3_from_scratch.factory import ComponentFactory, ModelConfig


@ComponentFactory.register("attn", "base")
class TorchGQA(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.n_head_embed = config.head_dim

    def forward(self, q, k, v):
        is_causal = q.shape[2] > 1
        return scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            enable_gqa=True,
            scale=self.n_head_embed**-0.5,
        )


def create_causal_attention_mask(seq_len, device, dtype):
    """
    为 eager_attention_forward 生成纯因果掩码
    Args:
        seq_len: 序列长度（如 128）
        device: 设备（如 "cuda"）
        dtype: 数据类型（如 torch.float16）
    Returns:
        attention_mask: 形状 [1,1,seq_len,seq_len]，下三角为0，上三角为-inf
    """
    # 1. 生成下三角布尔掩码（True 表示有效位置）
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )
    # 2. 转换为数值掩码：有效位置=0.0，无效位置=-inf
    attention_mask = torch.zeros_like(causal_mask, dtype=dtype)
    attention_mask = attention_mask.masked_fill(
        ~causal_mask, torch.finfo(dtype).min
    )  # 用finfo.min避免溢出
    # 3. 扩展维度到 [1,1,seq_len,seq_len]（适配注意力分数的维度）
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    return attention_mask
