#ifndef UTILS_CUH
#define UTILS_CUH

#include <torch/extension.h>

template <typename T> inline void CheckCuda(T& x) {
    TORCH_CHECK(x.device().is_cuda(), "Must be on CUDA device");
}

template <typename T> inline void CheckContiguous(T& x) {
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous.");
}

template <typename T> inline void CheckInput(T& x) {
    CheckCuda(x);
    CheckContiguous(x);
}

#endif