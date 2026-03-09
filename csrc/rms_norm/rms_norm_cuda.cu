#include "utils.h"
#include "kernel.h"
#include "cuda_funcs.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T, size_t blockSize>
__global__ void rms_norm_kernel(const T *x, T *output, const T *gamma, const int seqLen, const int hiddenDim, const int hiddenDimStride, const float eps)
{
    uint32_t tid = threadIdx.x;
    uint32_t blockId = blockIdx.x;
    x += blockId * hiddenDimStride; // 如果是连续，hiddenDimStride就是 hiddenDim
    output += blockId * hiddenDim /* *1 output是刚申请的，stride肯定是1*/;

    float sumSq = 0.0f;
    for (uint32_t i = tid; i < hiddenDim; i += blockSize)
    {
        const float x_i = static_cast<float>(x[i]);
        sumSq += x_i * x_i;
    }

    sumSq = warp_reduce_sum(sumSq);
    if constexpr (blockSize > WARP_SIZE)
    {
        static_assert((blockSize <= 1024) && (blockSize % WARP_SIZE == 0), "blockSize must be a multiple of warpSize");
        __shared__ T s_sum[32];
        const uint32_t warpId = tid / WARP_SIZE;
        const uint32_t laneId = tid % WARP_SIZE;
        if (laneId == 0)
        {
            s_sum[warpId] = sumSq;
        }
        __syncthreads();
        sumSq = 0.0f;
        if (laneId < (blockSize / WARP_SIZE))
        {
            sumSq = s_sum[laneId];
        }
        sumSq = warp_reduce_sum(sumSq);
    }
    const float mean = sumSq / hiddenDim;
    const float scale = rsqrtf(mean + eps);

    for (uint32_t i = tid; i < hiddenDim; i += blockSize)
    {
        output[i] = static_cast<T>(static_cast<float>(x[i]) * scale * static_cast<float>(gamma[i]));
    }
}

template <typename T>
void rms_norm_kernel(const T *x, T *output, const T *gamma, const int seqLen, const int hiddenDim, const int hiddenDimStride, const float eps)
{
    const dim3 gridSize(seqLen);

    if (hiddenDim < 1024)
    {
        const dim3 blockSize(256);
        rms_norm_kernel<T, 256><<<gridSize, blockSize>>>(x, output, gamma, seqLen, hiddenDim, hiddenDimStride, eps);
    }
    else
    {
        const dim3 blockSize(1024);
        rms_norm_kernel<T, 1024><<<gridSize, blockSize>>>(x, output, gamma, seqLen, hiddenDim, hiddenDimStride, eps);
    }
}

torch::Tensor rms_norm_forward_cuda(const torch::Tensor &x, const torch::Tensor &gamma, const float eps)
{
    const int seqLen = x.size(0);
    const int hiddenDim = x.size(1);

    auto output = torch::empty_like(x);
    if (x.dtype() == torch::kF16)
    {
        rms_norm_kernel<at::Half>(x.data_ptr<at::Half>(), output.data_ptr<at::Half>(), gamma.data_ptr<at::Half>(), seqLen, hiddenDim, x.stride(0), eps);
    }
    else if (x.dtype() == torch::kF32)
    {
        rms_norm_kernel<float>(x.data_ptr<float>(), output.data_ptr<float>(), gamma.data_ptr<float>(), seqLen, hiddenDim, x.stride(0), eps);
    }
    else if (x.dtype() == torch::kBFloat16)
    {
        rms_norm_kernel<at::BFloat16>(x.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(), gamma.data_ptr<at::BFloat16>(), seqLen, hiddenDim, x.stride(0), eps);
    }
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    return output;
}