from typing import Optional

import torch
import triton
import triton.language as tl
from qwen3_from_scratch.kernels.triton.gemm import (
    ActivationType,
    gemm_kernel_core,
    linear
)


@triton.jit
def swiglu(
    x,
    up_proj_weight,
    gate_proj_weight,
    down_proj_weight,
    output,
    N,
    D: tl.constexpr,
    D1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D1: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)

    a_dtype = x.dtype.element_ty
    w_dtype = up_proj_weight.dtype.element_ty

    output_ptr = tl.make_block_ptr(
        output + b * N * D,
        (N, D),
        (D, 1),
        (i * BLOCK_SIZE_N, 0),
        (BLOCK_SIZE_N, BLOCK_SIZE_D),
        (1, 0),
    )

    for n in tl.range(0, tl.cdiv(D, BLOCK_SIZE_D)):
      down_proj_ptr = tl.make_block_ptr(
        down_proj_weight,
        (D, D1),
        (D1, 1),
        (n * BLOCK_SIZE_D, 0),
        (BLOCK_SIZE_D, BLOCK_SIZE_D1),
        (1, 0),
      )
      result_in = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D), dtype=tl.float32)
      for l in tl.range(0, tl.cdiv(D1, BLOCK_SIZE_D1)):
        up_il = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D1), dtype=tl.float32)
        gate_il = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_D1), dtype=tl.float32)
        x_ptr = tl.make_block_ptr(
            x + b * N * D,
            (N, D),
            (D, 1),
            (i * BLOCK_SIZE_N, 0),
            (BLOCK_SIZE_N, BLOCK_SIZE_D),
            (1, 0),
        )
        up_proj_ptr = tl.make_block_ptr(
            up_proj_weight,
            (D1, D),
            (D, 1),
            (l * BLOCK_SIZE_D1, 0),
            (BLOCK_SIZE_D1, BLOCK_SIZE_D),
            (1, 0),
        )
        gate_proj_ptr = tl.make_block_ptr(
            gate_proj_weight,
            (D1, D),
            (D, 1),
            (l * BLOCK_SIZE_D1, 0),
            (BLOCK_SIZE_D1, BLOCK_SIZE_D),
            (1, 0),
        )
        for j in tl.range(0, tl.cdiv(D, BLOCK_SIZE_D)):
          X_ij = tl.load(x_ptr, boundary_check=(0,1)).to(w_dtype)
          up_weight_lj = tl.load(up_proj_ptr, boundary_check=(0,1))
          gate_weight_lj = tl.load(gate_proj_ptr, boundary_check=(0,1))

          up_il = tl.dot(X_ij, up_weight_lj.T, up_il)
          gate_il = tl.dot(X_ij, gate_weight_lj.T, gate_il)

          x_ptr = x_ptr.advance([0, BLOCK_SIZE_D])
          up_proj_ptr = up_proj_ptr.advance([0, BLOCK_SIZE_D])
          gate_proj_ptr = gate_proj_ptr.advance([0, BLOCK_SIZE_D])

        merged_il = up_il * gate_il / (1 + tl.exp(-gate_il))

        down_weight_nl = tl.load(down_proj_ptr, boundary_check=(0, 1))
        result_in = tl.dot(merged_il.to(w_dtype), down_weight_nl.T, result_in)
        down_proj_ptr = down_proj_ptr.advance([0, BLOCK_SIZE_D1])

      tl.store(output_ptr, result_in.to(a_dtype), boundary_check=(0, 1))
      output_ptr = output_ptr.advance([0, BLOCK_SIZE_D])


def swiglu_feedback(
    x: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    output: torch.Tensor
):
    activation_fc = ActivationType.SILU
    B, N, D = x.shape
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D1 = 32
    BLOCK_SIZE_D = 128
    D1, _ = up_proj_weight.shape
    assert up_proj_weight.shape == (D1, D), f"up_proj_weight shape mismatch: expected {(D1, D)}, got {up_proj_weight.shape}"
    assert down_proj_weight.shape == (D, D1), f"down_proj_weight shape mismatch: expected {(D, D1)}, got {down_proj_weight.shape}"
    assert output.shape == x.shape, f"output shape mismatch: expected {x.shape}, got {output.shape}"
    # 简单起见要求三个都连续，也是应该的
    assert x.is_contiguous(), "x must be contiguous"
    assert up_proj_weight.is_contiguous(), "up_proj_weight must be contiguous"
    assert down_proj_weight.is_contiguous(), "down_proj_weight must be contiguous"
    assert output.is_contiguous(), "output must be contiguous"

    grid = [triton.cdiv(N, BLOCK_SIZE_N), B]
    swiglu[grid](
      x, up_proj_weight, gate_proj_weight, down_proj_weight, output, 
      N, D, D1,
      BLOCK_SIZE_N, BLOCK_SIZE_D1, BLOCK_SIZE_D
    )

@triton.jit
def swiglu_gate(
  up_embed, gate_embed,
  N, D: tl.constexpr,
  BLOCK_SIZE_D: tl.constexpr
):
  d_id = tl.program_id(0)
  n_id = tl.program_id(1)
  offsets = d_id * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
  items_up = tl.load(up_embed + n_id * D * 2 +  offsets, offsets < D, 0.0)
  items_gate = tl.load(gate_embed + n_id * D * 2 + offsets, offsets < D, 0.0).to(tl.float32)
  items_gate *= tl.sigmoid(items_gate)
  items_up *= items_gate.to(items_up.dtype)
  tl.store(up_embed + n_id * D * 2+ offsets, items_up, offsets < D)

def simple_swiglu(
  x: torch.Tensor,
  merged_weight: torch.Tensor,
  down_proj_weight: torch.Tensor,
  output: torch.Tensor,
  residual: Optional[torch.Tensor] = None
):
  D = x.shape[-1]
  D1, _ = merged_weight.shape
  assert D1 % 2 == 0
  assert merged_weight.shape == (D1, D), f"merged_weight shape mismatch: expected {(D1, D)}, got {merged_weight.shape}"
  assert down_proj_weight.shape == (D, D1 // 2), f"down_proj_weight shape mismatch: expected {(D, D1 // 2)}, got {down_proj_weight.shape}"
  assert output.shape == x.shape, f"output shape mismatch: expected {x.shape}, got {output.shape}"
  # 简单起见要求连续
  assert x.is_contiguous(), "x must be contiguous"
  assert merged_weight.is_contiguous(), "merged_weight must be contiguous"
  assert down_proj_weight.is_contiguous(), "down_proj_weight must be contiguous"
  assert output.is_contiguous(), "output must be contiguous"
  M = x.numel() // D
  merged_embed = torch.empty(M, D1, dtype=x.dtype, device=x.device)
  linear(x, merged_weight, merged_embed)
  split_D = D1 // 2
  up_embed, gate_embed = torch.split(merged_embed, [split_D, split_D], dim=-1)
  BLOCK_SIZE_D = 128

  grid = [triton.cdiv(split_D, BLOCK_SIZE_D), M]
  swiglu_gate[grid](up_embed, gate_embed, M, split_D, BLOCK_SIZE_D)

  linear(up_embed, down_proj_weight, output, bias=residual)


class StandardSwiglu(torch.nn.Module):
  def __init__(self, up_proj, gate_proj, down_proj):
    super().__init__()
    self.up_proj = up_proj
    self.gate_proj = gate_proj
    self.down_proj = down_proj

  def forward(self, x):
    embed_up = self.up_proj(x)
    embed_gate = self.gate_proj(x)
    embed_gate = torch.nn.functional.silu(embed_gate)
    merged = embed_up * embed_gate
    return self.down_proj(merged)

if __name__ == "__main__":
    B = 2
    N = 32
    D = 1024
    D1 = D * 3
    device = 'cuda'
    x = torch.rand(B, N, D, dtype=torch.float32, device=device)
    up_proj = torch.nn.Linear(D, D1, bias=False).cuda()
    gate_proj = torch.nn.Linear(D, D1, bias=False).cuda()
    down_proj = torch.nn.Linear(D1, D, bias=False).cuda()

    standard_model = StandardSwiglu(up_proj, gate_proj, down_proj).cuda()
    with torch.no_grad():
      ref_o = standard_model(x)
      # output = torch.empty_like(x)
      # swiglu_feedback(x, up_proj.weight, gate_proj.weight, down_proj.weight, output)
      merged_proj = torch.concat([up_proj.weight, gate_proj.weight], 0)
      output = torch.empty_like(x)
      simple_swiglu(x, merged_proj, down_proj.weight, output)
      diff = (ref_o - output).abs()
      print(diff.max())