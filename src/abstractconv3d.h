#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor abstract_conv3d_forward(torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size);