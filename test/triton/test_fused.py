import torch
import torch.nn as nn
import pytest
from qwen3_from_scratch.kernels.triton.fused.fused_rms_norm_linear import fused_rms_norm_linear_1d


class SimpleFusedRmsNormLinear(nn.Module):
  def __init__(self, input_features:int, output_features:int, dtype:torch.dtype=torch.float32, eps:float=1e-5, bias:bool =False):
    super().__init__()
    self.dtype = dtype
    self.norm = nn.RMSNorm((input_features,), eps=eps, dtype=dtype)
    self.linear = nn.Linear(input_features, output_features, bias=bias, dtype=dtype)

  def forward(self, x):
    ori_dtype = x.dtype
    x = self.norm(x)
    return self.linear(x.to(self.dtype)).to(ori_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
def test_fused_rms_norm_linear_1d(dtype, bias):
  B, N, D, D1 = 2, 32, 1024, 3072
  
  # 根据dtype设置精度要求
  if dtype == torch.float32:
    atol, rtol = 1e-5, 1e-5  # f32: 高精度要求
  else:
    atol, rtol = 1e-2, 1e-2   # f16/bf16: 低精度要求
  
  simple_fused_rms_norm = SimpleFusedRmsNormLinear(D, D1, dtype=dtype, bias=bias).cuda()
  x = torch.randn(B, N, D, dtype=torch.float32, device='cuda')
  ref_o = simple_fused_rms_norm(x)

  target = torch.empty(B, N, D1, dtype=torch.float32, device='cuda')
  fused_rms_norm_linear_1d(
    x, 
    simple_fused_rms_norm.linear.weight, 
    simple_fused_rms_norm.norm.weight, 
    target, 
    bias=simple_fused_rms_norm.linear.bias
  )
  torch.testing.assert_close(ref_o, target, atol=atol, rtol=rtol)