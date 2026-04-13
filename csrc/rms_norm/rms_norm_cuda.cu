#include "utils.h"
#include "kernel.h"
#include "cuda_funcs.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <type_traits>

template <typename T, size_t blockSize>
__global__ void rms_norm_kernel_vec(const T* __restrict__ x,
                                    T* __restrict__ output,
                                    const T* __restrict__ gamma,
                                    const int seqLen,
                                    const int hiddenDim,
                                    const int hiddenDimStride,
                                    const float eps) {
    uint32_t tid = threadIdx.x;
    uint32_t blockId = blockIdx.x;
    const T* x_ptr = x + blockId * hiddenDimStride;
    output += blockId * hiddenDim;

    constexpr size_t vecSize = sizeof(float4) / sizeof(T);
    constexpr size_t vecElements = blockSize * vecSize;
    const size_t vecLoops = (hiddenDim + vecElements - 1) / vecElements;

    float sumSq = 0.0f;
    for (size_t loop = 0; loop < vecLoops; loop++) {
        size_t baseIdx = loop * vecElements + tid * vecSize;
        if (baseIdx < hiddenDim) {
            float4 vecX = reinterpret_cast<const float4*>(x_ptr + baseIdx)[0];
            sumSq += vecX.x * vecX.x + vecX.y * vecX.y + vecX.z * vecX.z + vecX.w * vecX.w;
        }
    }
    sumSq = warp_reduce_sum(sumSq);
    if constexpr (blockSize > WARP_SIZE) {
        __shared__ float s_sum[32];
        const uint32_t warpId = tid / WARP_SIZE;
        const uint32_t laneId = tid % WARP_SIZE;
        if (laneId == 0) {
            s_sum[warpId] = sumSq;
        }
        __syncthreads();
        sumSq = 0.0f;
        if (laneId < (blockSize / WARP_SIZE)) {
            sumSq = s_sum[laneId];
        }
        sumSq = warp_reduce_sum(sumSq);
    }
    const float mean = sumSq / hiddenDim;
    const float scale = rsqrtf(mean + eps);

    for (size_t loop = 0; loop < vecLoops; loop++) {
        size_t baseIdx = loop * vecElements + tid * vecSize;
        if (baseIdx < hiddenDim) {
            size_t remaining = hiddenDim - baseIdx;
            float4 vecX = reinterpret_cast<const float4*>(x_ptr + baseIdx)[0];
            float4 vecGamma = reinterpret_cast<const float4*>(gamma + baseIdx)[0];
            float4 result;
            result.x = vecX.x * scale * vecGamma.x;
            if (remaining > 1)
                result.y = vecX.y * scale * vecGamma.y;
            if (remaining > 2)
                result.z = vecX.z * scale * vecGamma.z;
            if (remaining > 3)
                result.w = vecX.w * scale * vecGamma.w;
            reinterpret_cast<float4*>(output + baseIdx)[0] = result;
        }
    }
}

template <typename T, size_t blockSize, size_t hiddenDim>
__global__ void rms_norm_kernel_arr(const T* __restrict__ x,
                                    T* __restrict__ output,
                                    const T* __restrict__ gamma,
                                    const int seqLen,
                                    const int hiddenDimStride,
                                    const float eps) {
    uint32_t tid = threadIdx.x;
    uint32_t blockId = blockIdx.x;
    const T* x_ptr = x + blockId * hiddenDimStride; // 如果是连续，hiddenDimStride就是 hiddenDim
    output += blockId * hiddenDim /* *1 output是刚申请的，stride肯定是1*/;

    float sumSq = 0.0f;
    T tempX[hiddenDim / blockSize];
    int tempXj = 0;
#pragma unroll
    for (uint32_t i = tid; i < hiddenDim; i += blockSize) {
        tempX[tempXj] = x_ptr[i];
        sumSq += tempX[tempXj] * tempX[tempXj];
        tempXj++;
    }
    sumSq = warp_reduce_sum(sumSq);
    if constexpr (blockSize > WARP_SIZE) {
        static_assert((blockSize <= 1024) && (blockSize % WARP_SIZE == 0), "blockSize must be a multiple of warpSize");
        __shared__ T s_sum[32];
        const uint32_t warpId = tid / WARP_SIZE;
        const uint32_t laneId = tid % WARP_SIZE;
        if (laneId == 0) {
            s_sum[warpId] = sumSq;
        }
        __syncthreads();
        sumSq = 0.0f;
        if (laneId < (blockSize / WARP_SIZE)) {
            sumSq = s_sum[laneId];
        }
        sumSq = warp_reduce_sum(sumSq);
    }
    const float mean = sumSq / hiddenDim;
    const float scale = rsqrtf(mean + eps);

    tempXj = 0;
#pragma unroll
    for (uint32_t i = tid; i < hiddenDim; i += blockSize) {
        output[i] = static_cast<T>(tempX[tempXj] * scale * gamma[i]);
        tempXj++;
    }
}

template <typename T>
void rms_norm_kernel(const T* __restrict__ x,
                     T* __restrict__ output,
                     const T* __restrict__ gamma,
                     const int seqLen,
                     const int hiddenDim,
                     const int hiddenDimStride,
                     const float eps) {
    const dim3 grid(seqLen);
    if (hiddenDim <= 64) {
        dim3 block(WARP_SIZE);
        rms_norm_kernel_vec<T, WARP_SIZE><<<grid, block>>>(x, output, gamma, seqLen, hiddenDim, hiddenDimStride, eps);
    } else if (hiddenDim <= 256) {
        dim3 block(64);
        rms_norm_kernel_vec<T, 64><<<grid, block>>>(x, output, gamma, seqLen, hiddenDim, hiddenDimStride, eps);
    } else if (hiddenDim <= 1024) {
        dim3 block(256);
        rms_norm_kernel_vec<T, 256><<<grid, block>>>(x, output, gamma, seqLen, hiddenDim, hiddenDimStride, eps);
    } else if (hiddenDim <= 2048) {
        dim3 block(512);
        rms_norm_kernel_vec<T, 512><<<grid, block>>>(x, output, gamma, seqLen, hiddenDim, hiddenDimStride, eps);
    } else {
        dim3 block(1024);
        rms_norm_kernel_vec<T, 1024><<<grid, block>>>(x, output, gamma, seqLen, hiddenDim, hiddenDimStride, eps);
    }
}

torch::Tensor rms_norm_forward_cuda(const torch::Tensor& x, const torch::Tensor& gamma, const float eps) {
    const int hiddenDim = x.size(-1);
    const int stride = x.stride(-2);
    const int seqLen = x.numel() / hiddenDim;

    auto output = torch::empty_like(x);
    if (x.dtype() == torch::kF16) {
        rms_norm_kernel<at::Half>(x.data_ptr<at::Half>(),
                                  output.data_ptr<at::Half>(),
                                  gamma.data_ptr<at::Half>(),
                                  seqLen,
                                  hiddenDim,
                                  stride,
                                  eps);
    } else if (x.dtype() == torch::kF32) {
        rms_norm_kernel<float>(
            x.data_ptr<float>(), output.data_ptr<float>(), gamma.data_ptr<float>(), seqLen, hiddenDim, stride, eps);
    } else if (x.dtype() == torch::kBFloat16) {
        rms_norm_kernel<at::BFloat16>(x.data_ptr<at::BFloat16>(),
                                      output.data_ptr<at::BFloat16>(),
                                      gamma.data_ptr<at::BFloat16>(),
                                      seqLen,
                                      hiddenDim,
                                      stride,
                                      eps);
    }
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    return output;
}