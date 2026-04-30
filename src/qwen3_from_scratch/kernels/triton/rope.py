import torch
import triton
import triton.language as tl


@triton.jit
def neox_rope_kernel(
  Q, cos_t, sin_t, output,stride_qb, stride_cos_b, stride_sin_b, stride_ob,  D:tl.constexpr, BLOCK_SIZE:tl.constexpr
):
  half_D = D //2
  pid_b = tl.program_id(0)
  pid_n = tl.program_id(1)
  pid_d = tl.program_id(2)
  tid = tl.arange(0, BLOCK_SIZE)

  Q_ptr = Q + pid_b * stride_qb
  cos_ptr = cos_t + pid_b * stride_cos_b
  sin_ptr = sin_t + pid_b * stride_sin_b
  output_ptr = output + pid_b * stride_ob

  offsets_d_1 = pid_d * BLOCK_SIZE + tid
  offsets_d_2 = offsets_d_1 + half_D
  
  offsets_1 = pid_n * D + offsets_d_1
  offsets_2 = pid_n * D + offsets_d_2

  mask1 = offsets_d_1 < half_D
  mask2 = offsets_d_2 < D

  part1 = tl.load(Q_ptr + offsets_1, mask1, 0.0)
  part2 = tl.load(Q_ptr + offsets_2, mask2, 0.0)
  
  items_cos_1 = tl.load(cos_ptr + offsets_1, mask1, 0.0)
  items_sin_1 = tl.load(sin_ptr + offsets_1, mask1, 0.0)
  items_cos_2 = tl.load(cos_ptr + offsets_2, mask2, 0.0)
  items_sin_2 = tl.load(sin_ptr + offsets_2, mask2, 0.0)

  result1 = part1 * items_cos_1 - part2 * items_sin_1
  result2 = part1 * items_sin_2 + part2 * items_cos_2

  tl.store(output_ptr + offsets_1, result1, mask1)
  tl.store(output_ptr + offsets_2, result2, mask2)


# Q, cos, sin, output are tensors on the GPU
def neox_rope(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    B = Q.shape[0]
    N = Q.shape[-2]
    D = Q.shape[-1]
    assert D % 2 == 0
    Q_view = Q.reshape(-1, N, D)
    BLOCK_SIZE = 256
    grid = (Q_view.shape[0], N, triton.cdiv(D//2, BLOCK_SIZE))
    output = torch.empty_like(Q_view)
    neox_rope_kernel[grid](Q_view, cos, sin, output, Q_view.stride(0),0, 0, output.stride(0), D, BLOCK_SIZE)
    return output.view(*Q.shape)