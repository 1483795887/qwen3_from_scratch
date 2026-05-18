import torch
import triton
import triton.language as tl
from qwen3_from_scratch.kernels.triton.gemm import (
    ActivationType,
    gemm_kernel_core,
    linear
)
from torch import nn
from torch.nn import functional as F

# D足够小，一个CTA全装下
@triton.jit
def fused_qk_norm_rope_kernel(qkv, gamma, cos_embed, sin_embed, 
  N, H_q: tl.constexpr, H_all: tl.constexpr, D: tl.constexpr, eps):
  """
  QKV打包在一起，BxNx(Hq+Hk+Hq)xd，通过hid控制不计算V的部分
  QK的Norm打包在一起，2xd，通过hid 是否大于 H 判断是Q还是K的Norm
  cos和sin一起公用, Nxd
  """
  h_id = tl.program_id(0)
  n_id = tl.program_id(1)
  b_id = tl.program_id(2)

  stride_h = D
  stride_n = stride_h * H_all
  stride_b = stride_n * N

  qkv_ptr = qkv + b_id * stride_b + n_id * stride_n + h_id * stride_h

  cos_ptr = cos_embed + n_id * D
  sin_ptr = sin_embed + n_id * D

  offsets1 = tl.arange(0, D // 2)
  offsets2 = tl.arange(D // 2, D)

  items_1 = tl.load(qkv_ptr + offsets1)
  items_2 = tl.load(qkv_ptr + offsets2)

  sum_sq = tl.sum(items_1 * items_1 + items_2 * items_2)
  scale = tl.rsqrt(sum_sq / D + eps)

  norm_h_id = 0 if h_id < H_q else 1
  norm_ptr = gamma + norm_h_id * D
  norm_1 = tl.load(norm_ptr + offsets1)
  norm_2 = tl.load(norm_ptr + offsets2)

  items_1 = items_1 * norm_1
  items_2 = items_2 * norm_2

  cos_1 = tl.load(cos_ptr + offsets1)
  cos_2 = tl.load(cos_ptr + offsets2)
  sin_1 = tl.load(sin_ptr + offsets1)
  sin_2 = tl.load(sin_ptr + offsets2)

  result_1 = items_1 * cos_1 - items_2 * sin_1
  result_2 = items_1 * sin_2 + items_2 * cos_2

  result_1 = result_1 * scale
  result_2 = result_2 * scale
  tl.store(qkv_ptr + offsets1, result_1)
  tl.store(qkv_ptr + offsets2, result_2)

def fused_qk_norm_rope(
  qkv: torch.Tensor, gamma:torch.Tensor, cos_embed: torch.Tensor, sin_embed:torch.Tensor, HEAD_DIM:int,groups:int, eps:float=1e-5
):
  assert qkv.is_contiguous()
  assert len(qkv.shape) == 3
  B, N, D = qkv.shape
  assert D %((groups + 2)*HEAD_DIM) == 0
  H_kv = D //((groups+ 2) * HEAD_DIM)
  H_q = H_kv * groups
  assert cos_embed.shape == (N,HEAD_DIM) == sin_embed.shape
  assert gamma.shape == (2,HEAD_DIM)

  grid = [H_q + H_kv, N, B]
  fused_qk_norm_rope_kernel[grid](qkv, gamma, cos_embed, sin_embed,
    N, H_q, H_q + 2 * H_kv, HEAD_DIM, eps)



class ReferenceFusedQKNormRope(nn.Module):
  def __init__(self, q_norm_weight, k_norm_weight):
    super().__init__()
    self.q_norm_weight = q_norm_weight
    self.k_norm_weight = k_norm_weight


  def forward(self,q,k, cos_embed, sin_embed):
    q = F.rms_norm(q, self.q_norm_weight.shape, self.q_norm_weight)
    k = F.rms_norm(k, self.k_norm_weight.shape, self.k_norm_weight)
    cos_embed = cos_embed.unsqueeze(-2).unsqueeze(0)
    sin_embed = sin_embed.unsqueeze(-2).unsqueeze(0)
    q = (q * cos_embed) + (self._rotate_half_neox(q) * sin_embed)
    k = (k * cos_embed) + (self._rotate_half_neox(k) * sin_embed)

    return q, k

  def _rotate_half_neox(self, x: torch.Tensor) -> torch.Tensor:
      """NeoX风格的旋转：前后半段交叉"""
      x1, x2 = x.chunk(2, dim=-1)
      return torch.cat((-x2, x1), dim=-1)


def test_fused_norm_rope():
  dtype = torch.float16
  B = 2
  N = 1024
  H_q = 16
  groups = 2
  HEAD_DIM = 128
  device = 'cuda'
  qkv = torch.rand(B, N, (H_q *2) * HEAD_DIM, device=device)
  gamma = torch.rand(2, HEAD_DIM, device=device)
  cos_embed = torch.rand(N, HEAD_DIM, device=device)
  sin_embed = torch.rand(N, HEAD_DIM, device=device)

  ref = ReferenceFusedQKNormRope(gamma[0], gamma[1]).to(device)
  with torch.no_grad():
    ref_q,ref_k, _ = qkv.view(B, N, -1, HEAD_DIM).split([H_q, H_q // groups,H_q // groups], dim=-2)
    ref_q,ref_k = ref(ref_q,ref_k, cos_embed, sin_embed)

    fused_qk_norm_rope(qkv, gamma, cos_embed, sin_embed, HEAD_DIM, groups)
    target_q, target_k, _ = qkv.view(B, N, -1, HEAD_DIM).split([H_q, H_q // groups,H_q // groups], dim=-2)
    diff_q = (target_q - ref_q).abs()
    print(diff_q.max())

if __name__ == '__main__':
  test_fused_norm_rope()
