#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>

#define THREADS 512
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ int compute_ravel_hash(const int *coord, const int resolution, const int hashmap_size) {
    // compute hash function for tiled grid
    int index = 0;
    int stride = 1;
    for(int i=0; i<3 && stride <= hashmap_size; i++) {
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

__device__ int get_level(const int* offsets, const int tableoffset, const int num_levels) {
    // given the tableoffset in the big hash table, get its level
    for(int i=1; i<num_levels; i++) {
        if(tableoffset < offsets[i]) {
            return i-1;
        }
    }
    return num_levels-1;
}

template <typename scalar_t>
__global__ void abstract_conv3d_forward_kernel(
    const scalar_t* input, 
    scalar_t* output,
    const int* offsets,
    const int* resolutions,
    const scalar_t* weights,
    const scalar_t* bias,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size
) {
    /* 
    blocks are divided as (embedding_id, batch_id, output_channel_id)

    for each block, different threads compute different locations from the kernel
    y[b, n, c_out] = (sum over kernel locations) and (sum over x that map into n)

    the second sum cannot be optimized too much because of random hash memory access
    however, the first sum can be parallelized over different threads, with each thread
    assigned a location (k1, k2, k3) where it stores the kernel K[lvl, k1, k2, k3, C_o] -> a vector of size C_in

    this way, each thread makes a different memory access from the input which are randomly scattered anyway
    kernel is stored internally, so memory request is only made once

    */
    int n = blockIdx.x;
    int b = blockIdx.y;
    int c_out = blockIdx.z;

    int kernel_idx = threadIdx.x;   // this is the kernel index to compute
    int level = get_level(offsets, n, num_levels);  // get the level of the table
    int offset_lvl = offsets[level];
    int local_n = n - offset_lvl;
    int lvl_res = resolutions[level];
    int lvl_res3 = lvl_res*lvl_res*lvl_res;
    // int lvl_tablesize = offsets[level+1] - offset_lvl;

    // temp array to store kernel values  (max supported is 128)
    scalar_t kernel[128];
    // shared memory for kernel 
    __shared__ scalar_t result[THREADS];     // for storing y[b, n, c_out] which is a scalar (partial sum from (k1, k2, k3))
    __shared__ int kernelvolume;
    result[threadIdx.x] = 0;
    if(threadIdx.x == 0)
        kernelvolume = K1*K2*K3;
    __syncthreads();
    if(kernel_idx >= kernelvolume) {  // small kernel size, dont need all threads
        return;
    }
    int x_index;   // query [b, n, c_in]

    // kernel index is incremented by `THREADS` as we only have atmost `THREADS` threads 
    int kernel_offset = level*(kernelvolume*output_channels*input_channels) + c_out*input_channels;  // lvl*(K1*K2*K3*C_out*C_in) + c_out*C_in

    while(kernel_idx < kernelvolume) {
        int kernel_localoffset = kernel_offset + kernel_idx*output_channels*input_channels;  // adds k*(C_out*C_in)
        // unravel the kernel_index   (offset by half the kernel size)
        int k1 = (kernel_idx/(K2*K3)) - K1/2;
        int k2 = ((kernel_idx % (K2*K3))/K3) - K2/2;
        int k3 = (kernel_idx%K3) - K3/2;

        // pull this kernel from memory 
        for(int i=0; i<input_channels; i++) {
            // weights are of shape [levels, K1, K2, K3, C_out, C_in]
            kernel[i] = weights[kernel_localoffset + i];
        }
        // we have kernel [l, o, k1, k2, k3]
        // pull in the x-values
        // loop invariant: `local_n % T` : this local_n goes through all the (x, y, z) indices till it reaches res**3
        while(local_n < lvl_res3) {
            int coord[3];
            unravel_index(local_n, lvl_res, coord);
            coord[0] += k1;
            coord[1] += k2;
            coord[2] += k3;
            // now contains the neighbor
            if(out_of_bounds(coord, lvl_res)) {
                local_n += hashmap_size;
                continue;
            }
            x_index = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl; // global offset
            x_index = b*(num_embeddings*input_channels) + x_index*input_channels;
            for(int i=0; i<input_channels; i++) {
                result[threadIdx.x] += kernel[i]*input[x_index + i];
            }
            local_n += hashmap_size;
        }
        // consider next batch of threads
        kernel_idx += THREADS;
    }
    __syncthreads();   // sync threads to add up the result from all kernel locations
    if(threadIdx.x == 0) {
        scalar_t sum = 0;
        for(int i=0; i<THREADS; i++) {
            sum += result[i];
        }
        if(bias) {
            sum += bias[level*output_channels + c_out];
        }
        output[b*(num_embeddings*output_channels) + n*output_channels + c_out] = sum;
    }
    __syncthreads();
    // done
}

template <typename scalar_t>
void abstract_conv3d_forward_wrapper(
    const scalar_t* input, 
    scalar_t* output,
    const int* offsets,
    const int* resolutions,
    const scalar_t* weights,
    const scalar_t* bias,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size
) { 
    const dim3 blocks(batch_size, num_embeddings, output_channels);
    // call the kernel
    abstract_conv3d_forward_kernel<<<blocks, THREADS>>>(
        input, output, offsets, resolutions, weights, bias,
        batch_size, num_embeddings, input_channels, output_channels,
        K1, K2, K3, num_levels, hashmap_size
    );
}

torch::Tensor abstract_conv3d_forward(torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size) {
    /* 
    
    input: input hash entries    (B, N, C_in)
    output: output hash entries  (B, N, C_out)
    offset: offset entries (L)
    resolutions: resolutions of each level (L)
    weight: weight entries (L x C_in x C_out x K1 x K2 x K3)
    bias: bias entries (L x C_out)
    num_levels: int (L)
    hashmap_size: int (H)
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
    const int input_channels = input.size(2);
    const int output_channels = output.size(2);
    // kernel sizes (first index is levels)
    const int k1 = weight.size(1);
    const int k2 = weight.size(2);
    const int k3 = weight.size(3);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "abstract_conv3d_forward_wrapper", ([&] {
        abstract_conv3d_forward_wrapper<scalar_t>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), offsets.data_ptr<int>(), resolutions.data_ptr<int>(), weight.data_ptr<scalar_t>(), \
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr, batch_size, num_embeddings, input_channels, output_channels, k1, k2, k3, num_levels, hashmap_size);
    }));
    return output;
}