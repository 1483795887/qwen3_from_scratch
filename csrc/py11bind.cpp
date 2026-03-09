#include "kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "qwen3_from_scratch kernels";
    m.def(
        "rms_norm_forward",
        &rms_norm_forward,
        "RMSNorm forward computation (CPU/CUDA)",
        py::arg("x"),
        py::arg("gamma"),
        py::arg("eps") = 1e-6f);
}
