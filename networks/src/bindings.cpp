#include <ATen/ATen.h>
#include <torch/extension.h>
#include "abstractconv3d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("abstract_conv3d_forward", &abstract_conv3d_forward, "abstract_conv3d forward (CUDA)");
    m.def("abstract_conv3d_backward", &abstract_conv3d_backward, "abstract_conv3d backward (CUDA)");
}
