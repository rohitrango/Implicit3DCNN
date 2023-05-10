#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

torch::Tensor abstract_conv3d_forward(torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size, at::optional<torch::Tensor> fwd_index);

std::vector<at::optional<torch::Tensor>> abstract_conv3d_backward(torch::Tensor grad_output, torch::Tensor grad_input, torch::Tensor grad_weight, at::optional<torch::Tensor> grad_bias,
        bool inp_requires_grad, bool weight_requires_grad, torch::Tensor input, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size, at::optional<torch::Tensor> fwd_index,
        at::optional<torch::Tensor> bwd_index);

std::vector<torch::Tensor> abstract_conv3d_cache_index(torch::Tensor offsets, torch::Tensor resolutions, int num_levels, 
    int hashmap_size, int K1, int K2, int K3);