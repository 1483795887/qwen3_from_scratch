#ifndef CUDA_FUNCS_H
#define CUDA_FUNCS_H

#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

template <int width = WARP_SIZE, typename T> __device__ __forceinline__ T warp_reduce_sum(T x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) << " | 文件: " << file << " | 行号: " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERR(x) checkCudaError(cudaGetLastError(), __FILE__, __LINE__);

#endif