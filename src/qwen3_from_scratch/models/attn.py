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
            scale=self.n_head_embed ** -0.5,
        )


def group_matmul(mat1, mat2):
    assert mat1.shape[-1] == mat2.shape[-2]
    batch, h1, s1, d1 = mat1.shape
    _, h2, d2, s2 = mat2.shape
    assert d1 == d2
    assert h1 % h2 == 0
    group = h1 // h2
    mat1_reshaped = mat1.reshape(batch, h2, group, s1, d1)
    mat2_reshaped = mat2.unsqueeze(2)
    return torch.matmul(mat1_reshaped, mat2_reshaped).reshape(batch, h1, s1, s2)

@ComponentFactory.register("attn", "py_flash_attn")
class PyFlashAttention(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__()
        self.is_causal = kwargs.get("is_causal", True)
        self.n_head_embed = config.head_dim
        self.q_tile_size = int(kwargs.get("q_tile_size", 64))
        self.k_tile_size = int(kwargs.get("k_tile_size", 64))

    def forward(self, q, k, v):
        # BxHxSxD
        batch_size, head_q, seq_len_q = q.shape[:3]
        seq_len_k = k.shape[2]
        scale = self.n_head_embed ** -0.5
        output_shape = (batch_size, head_q, seq_len_q, self.n_head_embed)
        output = torch.zeros(output_shape, device=q.device)
        m = (torch.ones((batch_size, head_q, seq_len_q, 1), device=q.device)
             * (-torch.inf))
        l = torch.zeros((batch_size, head_q, seq_len_q, 1), device=q.device)
        for j in range(0, seq_len_k, self.k_tile_size):
            k_end = min(j + self.k_tile_size, seq_len_k)
            k_slice = slice(j, k_end)
            k_tile = k[:, :, k_slice].transpose(-2, -1)
            v_tile = v[:, :, k_slice]

            for i in range(0, seq_len_q, self.q_tile_size):
                q_end = min(i + self.q_tile_size, seq_len_q)
                q_slice = slice(i, q_end)
                q_tile = q[:, :, q_slice]
                # BxHxS1xS2
                attn = group_matmul(q_tile, k_tile) * scale
                if self.is_causal:
                    mask = (
                               torch.arange(i, q_end)
                               .unsqueeze(-1)
                               .to(q.device)
                           ) < (
                               torch.arange(j, k_end)
                               .unsqueeze(0)
                               .to(q.device)
                           )
                    attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), -torch.inf)
                    attn = attn.masked_fill(mask, -torch.inf)
                m_old = m[:, :, q_slice]
                o_old = output[:, :, q_slice]
                l_old = l[:, :, q_slice]

                m_new = torch.maximum(
                    m_old,
                    attn.max(dim=-1, keepdim=True).values,
                )
                exp_attn = torch.exp(attn - m_new)
                scale_max_diff = torch.exp(m_old - m_new)
                l_new = l_old * scale_max_diff + exp_attn.sum(dim=-1, keepdim=True)

                output[:, :, q_slice] = (group_matmul(exp_attn, v_tile)
                                         + o_old * scale_max_diff * l_old) / torch.clamp(l_new, min=1e-10)
                m[:, :, q_slice] = m_new
                l[:, :, q_slice] = l_new
        return output


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
