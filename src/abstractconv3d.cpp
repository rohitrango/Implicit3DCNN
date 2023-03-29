#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor abstract_conv3d_forward(torch::Tensor input, torch::Tensor offset, torch::Tensor weight, torch::Tensor bias) {
    CHECK_CUDA(input);
    CHECK_CUDA(offset);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(offset);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    
    torch::Tensor output = input + 1;
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("abstract_conv3d_forward", &abstract_conv3d_forward, "abstract_conv3d forward (CUDA)");
}
