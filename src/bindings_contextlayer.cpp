#include <ATen/ATen.h>
#include <torch/extension.h>
#include "abstractcontextlayer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("abstract_contextlayer_forward", &abstract_contextlayer_forward, "abstract_context_layer forward (CUDA)");
    m.def("abstract_contextlayer_backward", &abstract_contextlayer_backward, "abstract_context_layer backward (CUDA)");
}
