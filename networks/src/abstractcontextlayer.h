#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

torch::Tensor abstract_contextlayer_forward(torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
                         int num_levels, int hashmap_size);

torch::Tensor abstract_contextlayer_backward(torch::Tensor grad_output, torch::Tensor grad_input, 
        torch::Tensor offsets, torch::Tensor resolutions, int num_levels, int hashmap_size);