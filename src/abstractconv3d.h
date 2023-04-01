#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

torch::Tensor abstract_conv3d_forward(torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size);

std::vector<at::optional<torch::Tensor>> abstract_conv3d_backward(torch::Tensor grad_output, torch::Tensor grad_input, torch::Tensor grad_weight, at::optional<torch::Tensor> grad_bias,
        torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size);