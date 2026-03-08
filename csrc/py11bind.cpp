#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "qwen3_from_scratch kernels";
}
