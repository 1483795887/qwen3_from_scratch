#ifndef KERNEL_H
#define KERNEL_H
#include <torch/extension.h>

torch::Tensor rms_norm_forward_cuda(const torch::Tensor& x, const torch::Tensor& gamma, const float eps = 1e-6f);
torch::Tensor rms_norm_forward(const torch::Tensor& x, const torch::Tensor& gamma, const float eps = 1e-6f);
#endif