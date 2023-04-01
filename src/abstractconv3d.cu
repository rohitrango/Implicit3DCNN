/*

v1: <B, N, C_out> blocks, very slow

v2: <B, N, K1,K2,K3> blocks, hopefully faster with lot of coalescing
*/
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 512
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ int compute_ravel_hash(const int *coord, const int resolution, const int hashmap_size) {
    // compute hash function for tiled grid
    int index = 0;
    int stride = 1;
    #pragma unroll
    for(int i=0; i<3; i++) {
        index += coord[i]*stride;
        stride *= resolution;
    }
    return index % hashmap_size;
}

__device__ void unravel_index(const int index, const int resolution, int* coord) {
    // unravel index 
    int res2 = resolution*resolution;
    coord[2] = index / res2;
    coord[1] = (index%res2) / resolution;
    coord[0] = index % resolution;
}

__device__ bool out_of_bounds(const int* coord, const int resolution) {
    // check if the coordinate is out of bounds
    return coord[0] < 0 || coord[0] >= resolution || coord[1] < 0 || coord[1] >= resolution || coord[2] < 0 || coord[2] >= resolution;
}

__device__ int get_level(const int* __restrict__  offsets, const int tableoffset, const int num_levels) {
    // given the tableoffset in the big hash table, get its level
    for(int i=1; i<num_levels; i++) {
        if(tableoffset < offsets[i]) {
            return i-1;
        }
    }
    return num_levels-1;
}


template <typename scalar_t>
__global__ void abstract_conv3d_forward_kernel_v3(
    const scalar_t* __restrict__ input, 
    scalar_t* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ resolutions,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size) 
{
    // For each block, fetch the embedding id, block number and output
    int n = blockIdx.x;
    int b = blockIdx.y;
    int kernel_idx = blockIdx.z;
    // unravel the kernel_index  (offset by half the kernel size)
    int kernel_volume = K1*K2*K3;
    int k1 = (kernel_idx/(K2*K3)) - K1/2;
    int k2 = ((kernel_idx % (K2*K3))/K3) - K2/2;
    int k3 = (kernel_idx%K3) - K3/2;

    // this is the composition of (channel_in * channel_out) number (for more coalesced memory access)
    int c_idx = threadIdx.x;
    // get the level of the table
    while(n < num_embeddings) {
        int level = get_level(offsets, n, num_levels);  // get the level of the table
        int offset_lvl = offsets[level];
        int local_n = n - offset_lvl;
        int lvl_res = resolutions[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        int iosize = input_channels*output_channels;

        // 512 + 64 
        __shared__ scalar_t weight_[THREADS];
        __shared__ scalar_t res_[THREADS];
        __shared__ scalar_t bias_[64];
        __shared__ scalar_t inp_[64];
        // `kernel_idx` = 0 makes sure only one copy of bias is added
        if((bias != NULL) && kernel_idx == 0 && threadIdx.x < output_channels) {   // load bias weights directly
            bias_[threadIdx.x] = bias[level*output_channels + threadIdx.x];
        }
        else {
            bias_[threadIdx.x] = 0;
        }
        res_[c_idx] = 0;
        __syncthreads();

        // assuming c_out is less than 64, in each subsequent pass, we iterate over
        // { ... C_in_set ... } but all values of C_out
        while(c_idx < iosize) {    // c_idx = c_in * output_channels + c_out
            // get the channel index
            int c_in = c_idx / output_channels;
            // load weight and bias
            weight_[threadIdx.x] = weights[level*(kernel_volume*iosize) + kernel_idx*iosize + c_idx];
            // if(threadIdx.x == 0)
            //     start_cin = c_in;
            __syncthreads();
            int x_index;

            // iterate over inputs
            while(local_n < lvl_res3) {
                int coord[3];
                unravel_index(local_n, lvl_res, coord);
                coord[0] += k1;
                coord[1] += k2;
                coord[2] += k3;
                // for each thread either add or discard
                if(out_of_bounds(coord, lvl_res)) {
                }
                else {
                    x_index = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl; // global offset
                    // x_index = b*(num_embeddings*input_channels) + x_index*input_channels;
                    x_index = x_index*(batch_size*input_channels) + b*input_channels;
                    // read input
                    if(threadIdx.x < input_channels) {
                        inp_[threadIdx.x] = input[x_index + threadIdx.x];
                    }
                    __syncthreads();
                    // only the first batch gets to pull out the input
                    res_[threadIdx.x] += weight_[threadIdx.x]*inp_[c_in];
                }
                local_n += hashmap_size;
            }
            // increment by threads
            c_idx += THREADS;
        }
        // we have res[THREAD] = sum_{partial c_in} w_{c_in, c_out} * x_{c_in} 
        if(threadIdx.x < output_channels) {
            for(int i=threadIdx.x+output_channels; i<min(iosize, THREADS); i+=output_channels) {
                res_[threadIdx.x] += res_[i];
            }
            // atomicAdd(output + b*(num_embeddings*output_channels) + n*output_channels + threadIdx.x, res_[threadIdx.x]+bias_[threadIdx.x]);
            atomicAdd(output + b*output_channels + n*batch_size*output_channels + threadIdx.x, res_[threadIdx.x]+bias_[threadIdx.x]);
        }
        n += gridDim.x;
    }
}

template <typename scalar_t>
__global__ void abstract_conv3d_forward_kernel_v2(
    const scalar_t* __restrict__ input, 
    scalar_t* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ resolutions,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size)
{
    // For each block, fetch the embedding id, block number and output
    int n = blockIdx.x;
    int b = blockIdx.y;
    int kernel_idx = blockIdx.z;
    // unravel the kernel_index  (offset by half the kernel size)
    int kernel_volume = K1*K2*K3;
    int k1 = (kernel_idx/(K2*K3)) - K1/2;
    int k2 = ((kernel_idx % (K2*K3))/K3) - K2/2;
    int k3 = (kernel_idx%K3) - K3/2;

    // this is the composition of (channel_in * channel_out) number (for more coalesced memory access)
    int c_idx = threadIdx.x;

    // get the level of the table
    while(n < num_embeddings) {
        int level = get_level(offsets, n, num_levels);  // get the level of the table
        int offset_lvl = offsets[level];
        int local_n = n - offset_lvl;
        int lvl_res = resolutions[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        int iosize = input_channels*output_channels;

        // 512 + 64 
        __shared__ scalar_t weight_[THREADS];
        __shared__ scalar_t res_[THREADS];
        __shared__ scalar_t bias_[64];
        __shared__ scalar_t inp_[64];
        // `kernel_idx` = 0 makes sure only one copy of bias is added
        if((bias != NULL) && (kernel_idx == 0) && (threadIdx.x < output_channels)) {   // load bias weights directly
            bias_[threadIdx.x] = bias[level*output_channels + threadIdx.x];
        }
        else {
            bias_[threadIdx.x] = 0;
        }
        res_[c_idx] = 0;
        __syncthreads();

        // assuming c_out is less than 64, in each subsequent pass, we iterate over
        // { ... C_in_set ... } but all values of C_out
        while(c_idx < iosize) {    // c_idx = c_in * output_channels + c_out
            // get the channel index
            int c_in = c_idx / output_channels;
            // load weight and bias
            weight_[threadIdx.x] = weights[level*(kernel_volume*iosize) + kernel_idx*iosize + c_idx];
            // if(threadIdx.x == 0)
            //     start_cin = c_in;
            __syncthreads();
            int x_index;

            // iterate over inputs
            while(local_n < lvl_res3) {
                int coord[3];
                unravel_index(local_n, lvl_res, coord);
                coord[0] += k1;
                coord[1] += k2;
                coord[2] += k3;
                // for each thread either add or discard
                if(out_of_bounds(coord, lvl_res)) {
                }
                else {
                    x_index = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl; // global offset
                    x_index = b*(num_embeddings*input_channels) + x_index*input_channels;
                    // read input
                    if(threadIdx.x < input_channels) {
                        inp_[threadIdx.x] = input[x_index + threadIdx.x];
                    }
                    __syncthreads();
                    // only the first batch gets to pull out the input
                    res_[threadIdx.x] += weight_[threadIdx.x]*inp_[c_in];
                }
                local_n += hashmap_size;
            }
            // increment by threads
            c_idx += THREADS;
        }
        // we have res[THREAD] = sum_{partial c_in} w_{c_in, c_out} * x_{c_in} 
        if(threadIdx.x < output_channels) {
            for(int i=threadIdx.x+output_channels; i<min(iosize, THREADS); i+=output_channels) {
                res_[threadIdx.x] += res_[i];
            }
            atomicAdd(output + b*(num_embeddings*output_channels) + n*output_channels + threadIdx.x, res_[threadIdx.x]+bias_[threadIdx.x]);
        }
        n += gridDim.x;
    }
}

template <typename scalar_t>
__global__ void abstract_conv3d_backward_input_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ resolutions,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size
) {
    // do nothing
}

template <typename scalar_t>
__global__ void abstract_conv3d_backward_weight_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ resolutions,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size
) {
    // get information about block
    int kernel_volume = K1*K2*K3;
    int kernel_idx = blockIdx.x % kernel_volume;
    int lvl = blockIdx.x / kernel_volume;
    // calculate grad of weight[lvl, k1, k2, k3]
    int k1 = (kernel_idx/(K2*K3)) - K1/2;
    int k2 = ((kernel_idx % (K2*K3))/K3) - K2/2;
    int k3 = (kernel_idx%K3) - K3/2;
    // from thread index, infer output channel, batch size, etc
    int channelmax = max(input_channels, output_channels);
    int channelmin = min(input_channels, output_channels);
    // get channel index, batch size and embedding index
    int c_idx = threadIdx.x % channelmax;
    int b = threadIdx.x / channelmax;
    int n = threadIdx.x / (channelmax*batch_size);
    // auxiliary variables (to enable coalesced memory access)
    int max_ns_per_block = THREADS / (channelmax*batch_size);  
    int max_batch_sizes_per_block = max(batch_size, THREADS/channelmax);
    // 

}


template <typename scalar_t>
void abstract_conv3d_forward_wrapper(
    const scalar_t* __restrict__ input, 
    scalar_t* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ resolutions,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size
) { 
    // call the kernel
    // assume input and output are both below 64 channels (for now)
    // const dim3 blocks(min(num_embeddings/32, 65536), batch_size, K1*K2*K3);
    const dim3 blocks(num_embeddings/32, batch_size, K1*K2*K3);
    const uint32_t threads = min(THREADS, input_channels*output_channels); 

    // abstract_conv3d_forward_kernel_v3<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    abstract_conv3d_forward_kernel_v2<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input, output, offsets, resolutions, weights, bias,
        batch_size, num_embeddings, input_channels, output_channels,
        K1, K2, K3, num_levels, hashmap_size
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

template<typename scalar_t>
void abstract_conv3d_backward_wrapper(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_weights,
    scalar_t* __restrict__ grad_bias,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ resolutions,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size
) {
    const dim3 blocks(min(num_embeddings/32, 65536), batch_size, K1*K2*K3);
    const uint32_t threads = min(THREADS, input_channels*output_channels);
    // call gradient w.r.t. input
    abstract_conv3d_backward_input_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output, grad_input, 
        input, output, offsets, resolutions, weights, bias,
        batch_size, num_embeddings, input_channels, output_channels,
        K1, K2, K3, num_levels, hashmap_size
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    int weightblocks = K1*K2*K3*num_levels;
    abstract_conv3d_backward_weight_kernel<<<weightblocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output, grad_weights, grad_bias,
        input, output, offsets, resolutions, weights, bias,
        batch_size, num_embeddings, input_channels, output_channels,
        K1, K2, K3, num_levels, hashmap_size
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

torch::Tensor abstract_conv3d_forward(torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size) {
    /* 
    
    input: input hash entries    (B, N, C_in)
    output: output hash entries  (B, N, C_out)
    offset: offset entries (L)
    resolutions: resolutions of each level (L)
    weight: weight entries (L x C_out x C_in x K1 x K2 x K3) or (L x K1 x K2 x K3 x C_out x C_in) -> depending on coaslescing
    bias: bias entries (L x C_out)
    num_levels: int (L)
    hashmap_size: int (H)
    
    Notes: 
        - reducing `THREADS` to 32 from 512 led to improved performance, but reducing further down hurt performance
        - writing loops for `num_embeddings` improved performance (down from 1.7s to 0.63s)
    */

    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(offsets);
    CHECK_CUDA(resolutions);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(weight);

    if(bias.has_value()) {
        CHECK_CUDA(bias.value());
        CHECK_CONTIGUOUS(bias.value());
    }

    if(input.dim() != 3) {
        throw std::runtime_error("Input must have 3 dimensions");
    }
    if(weight.dim() != 6) {
        throw std::runtime_error("Weight must have 6 dimensions");
    }
    if(bias.has_value() && bias.value().dim() != 2) {
        throw std::runtime_error("Bias must have 2 dimensions");
    }

    // define extra variables
    const int batch_size = input.size(0);
    const int num_embeddings = input.size(1);
    // const int num_embeddings = input.size(0);
    // const int batch_size = input.size(1);
    const int input_channels = input.size(2);
    const int output_channels = output.size(2);
    // kernel sizes (first index is levels)

    // this is v2 kernel
    const int k1 = weight.size(1);
    const int k2 = weight.size(2);
    const int k3 = weight.size(3);

    AT_DISPATCH_FLOATING_TYPES( 
    input.scalar_type(), "abstract_conv3d_forward_wrapper", ([&] {
        abstract_conv3d_forward_wrapper<scalar_t>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), offsets.data_ptr<int>(), resolutions.data_ptr<int>(), weight.data_ptr<scalar_t>(), \
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr, batch_size, num_embeddings, input_channels, output_channels, k1, k2, k3, num_levels, hashmap_size);
    }));
    return output;
}

std::vector<at::optional<torch::Tensor>> abstract_conv3d_backward(torch::Tensor grad_output, torch::Tensor grad_input, torch::Tensor grad_weight, at::optional<torch::Tensor> grad_bias,
        torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size) {
    /* Kernel for backward pass */
    // define extra variables
    const int batch_size = input.size(0);
    const int num_embeddings = input.size(1);
    const int input_channels = input.size(2);
    const int output_channels = output.size(2);
    // kernel sizes
    const int k1 = weight.size(1);
    const int k2 = weight.size(2);
    const int k3 = weight.size(3);
    // call kernels
    AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "abstract_conv3d_backward_wrapper", ([&] {
        abstract_conv3d_backward_wrapper<scalar_t>(
            grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(), grad_weight.data_ptr<scalar_t>(), grad_bias.has_value() ? grad_bias.value().data_ptr<scalar_t>() : nullptr,
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), offsets.data_ptr<int>(), resolutions.data_ptr<int>(), weight.data_ptr<scalar_t>(), \
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr, batch_size, num_embeddings, input_channels, output_channels, k1, k2, k3, num_levels, hashmap_size);
    }));
    // return ret;
    return {grad_input, grad_weight, grad_bias};
}