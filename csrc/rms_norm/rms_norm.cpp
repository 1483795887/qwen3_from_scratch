#include "utils.h"
#include "kernel.h"

template <typename T>
void rms_norm_forward_(
    const T *x,
    T *output,
    const T *gamma,
    const int seqLen,
    const int hiddenDim,
    const int hiddenDimStride,
    const float eps)
{
    // 逐序列位置计算RMSNorm
    for (int i = 0; i < seqLen; i++)
    {
        const T *currentX = x + i * hiddenDimStride;
        T *currentOutput = output + i * hiddenDim;
        // 1. 计算当前序列位置的平方和
        float sumSq = 0.0f;
        for (int k = 0; k < hiddenDim; k++)
        {
            const float val = static_cast<float>(currentX[k]);
            sumSq += val * val;
        }

        // 2. 计算均方根（RMS）
        const float rms = sqrtf(sumSq / static_cast<float>(hiddenDim) + eps);

        // 3. 归一化 + 缩放
        for (int j = 0; j < hiddenDim; j++)
        {
            const float val = static_cast<float>(currentX[j]);
            currentOutput[j] = static_cast<T>(val / rms * static_cast<float>(gamma[j]));
        }
    }
}

// -------------------------- RMSNorm CPU实现 --------------------------
torch::Tensor rms_norm_forward_cpu(
    const torch::Tensor &x,
    const torch::Tensor &gamma,
    const float eps = 1e-6f)
{
    const int seqLen = x.size(0);
    const int hiddenDim = x.size(1);

    auto output = torch::empty_like(x);

    if (x.dtype() == torch::kF16)
    {
        rms_norm_forward_<at::Half>(x.data_ptr<at::Half>(), output.data_ptr<at::Half>(), gamma.data_ptr<at::Half>(), seqLen, hiddenDim, x.stride(0), eps);
    }
    else if (x.dtype() == torch::kF32)
    {
        rms_norm_forward_<float>(x.data_ptr<float>(), output.data_ptr<float>(), gamma.data_ptr<float>(), seqLen, hiddenDim, x.stride(0), eps);
    }
    else if (x.dtype() == torch::kBFloat16)
    {
        rms_norm_forward_<at::BFloat16>(x.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(), gamma.data_ptr<at::BFloat16>(), seqLen, hiddenDim, x.stride(0), eps);
    }

    return output;
}

// -------------------------- 设备自动选择的封装函数 --------------------------
torch::Tensor rms_norm_forward(
    const torch::Tensor &x,
    const torch::Tensor &gamma,
    const float eps)
{
    TORCH_CHECK(x.dtype() == torch::kBFloat16 ||
                    x.dtype() == torch::kFloat16 ||
                    x.dtype() == torch::kFloat32,
                "Only support BF16/FP16/FP32");
    TORCH_CHECK(gamma.dtype() == x.dtype(), "x和gamma的数据类型必须相同");
    TORCH_CHECK(x.size(1) == gamma.size(0), "gamma的维度必须等于x的hidden_dim");
#ifdef USE_CUDA
    // 根据输入设备自动选择CPU/CUDA版本
    if (x.is_cuda())
    {
        return rms_norm_forward_cuda(x, gamma, eps);
    }
    else
    {
        return rms_norm_forward_cpu(x, gamma, eps);
    }
#else
    return rms_norm_forward_cpu(x, gamma, eps);
#endif
}