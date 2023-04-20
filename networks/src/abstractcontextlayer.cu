#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <stdexcept>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 512
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define div_up(x, n) ((x)+(n)-1)/(n)
#define min(a, b) ((a)<(b))?(a):(b)

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

__device__ void unravel_index(const int index, const int resolution, int* coord) {
    // unravel index 
    int res2 = resolution*resolution;
    coord[2] = index / res2;
    coord[1] = (index%res2) / resolution;
    coord[0] = index % resolution;
}

template <typename scalar_t>
__global__ void abstract_contextlayer_forward_kernel(
    const scalar_t *input, 
    scalar_t *output,
    const int *offsets, 
    const int *resolutions,
    const int batch_size, 
    const int num_embeddings, 
    const int input_channels, 
    const int num_levels, 
    const int hashmap_size) 
{
    int num = batch_size*num_embeddings*input_channels;
    int output_channels = input_channels;

    CUDA_KERNEL_LOOP(index, num) {
        // this is at least the first level
        int c_idx = index % output_channels;
        int b_idx = (index / output_channels) % batch_size;
        int n_idx = (index / output_channels) / batch_size;
        // level related variables
        int level = get_level(offsets, n_idx, num_levels);
        // this is the first layer, no context from above
        if(level == 0)
            continue;
        int offset_lvl_prev = offsets[level - 1];
        int offset_lvl = offsets[level];
        int local_n = n_idx - offset_lvl;
        int lvl_res_prev = resolutions[level-1];
        int lvl_res = resolutions[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        // if this is a tail end, skip t
        if(local_n >= lvl_res3)
            continue;
        // initialize the result
        scalar_t res = 0;
        // initialize the index here and iterate through all possible coordinates at this resolution 
        int _iter_local_n = local_n;
        while(_iter_local_n < lvl_res3) {
            int coord[3];
            unravel_index(_iter_local_n, lvl_res, coord); 
            float coord_f[3];
            #pragma unroll
            for(int i=0; i<3; i++) 
                coord_f[i] = (((float)coord[i]) / (lvl_res - 1)) * (lvl_res_prev - 1);   // dividing makes it from [0, 1] and then resized to [0, lvl-1]
            //// given the nearest float coordinates at the previous level, find nearest int pixel location
            #pragma unroll
            for(int i=0; i<3; i++)
                coord[i] = (int)(coord_f[i] + 0.5 - 1e-6);
            // Get index from x
            int xindex = compute_ravel_hash(coord, lvl_res_prev, hashmap_size) + offset_lvl_prev;
            res += input[xindex*batch_size*input_channels + b_idx*input_channels + c_idx];
            // trilinearly interpolate (this is too slow)
            // for(int nbr=0; nbr<8; nbr++) {
            //     scalar_t wt = 1;
            //     for(int i=0; i<3; i++) {
            //         coord[i] = ((int)floorf(coord_f[i])) + nbr%2;   // if remainder is 0, its floor, else ceil
            //         wt *= 1 + ((coord[i] - coord_f[i]))*(nbr%2)?1:-1;
            //         nbr >>= 1;
            //     }
            //     if(wt > 0 && !out_of_bounds(coord, lvl_res_prev)) {
            //         int xindex = compute_ravel_hash(coord, lvl_res_prev, hashmap_size) + offset_lvl_prev;
            //         res += wt * input[xindex*batch_size*input_channels + b_idx*input_channels + c_idx];
            //     }
            // }
            // this coordinate is in the previous level
            // get to the next entry
            _iter_local_n += hashmap_size;
        }
        output[n_idx*batch_size*output_channels + b_idx*output_channels + c_idx] = res;
    }
}

template <typename scalar_t>
__global__ void abstract_contextlayer_backward_kernel(const scalar_t* grad_output, scalar_t* grad_input, const int *offsets, const int *resolutions,
            const int batch_size, const int num_embeddings, const int input_channels, const int num_levels, const int hashmap_size) {
    //
    int num = batch_size * num_embeddings * input_channels;
    int output_channels = input_channels;
    CUDA_KERNEL_LOOP(index, num) {
        int c_idx = index % output_channels;
        int b_idx = (index / output_channels) % batch_size;
        int n_idx = (index / output_channels) / batch_size;
        // get level and check 
        int level = get_level(offsets, n_idx, num_levels);
        // this is the last layer, no context from below, return since there is no more level to consider
        if(level == num_levels-1)
            return; 
        // level related variables 
        int offset_lvl_next = offsets[level + 1];
        int lvl_res_next = resolutions[level + 1];
        int offset_lvl = offsets[level];
        int local_n = n_idx - offset_lvl;
        int lvl_res = resolutions[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        // if this is a tail end, skip t
        if(local_n >= lvl_res3)
            continue; 
        // initialize result
        scalar_t res = 0;
        int _iter_local_n = local_n;
        while(_iter_local_n < lvl_res3) {
            int coord[3];
            unravel_index(_iter_local_n, lvl_res, coord);  // coord contains the (x, y, z) at level 'l'
            int coord_fmin[3], coord_fmax[3];    // store min and max from each layer
            #pragma unroll
            for(int i=0; i<3; i++) {
                coord_fmin[i] = (int)ceilf(((float)coord[i] - 0.5 + 1e-6)/(lvl_res - 1)*(lvl_res_next - 1));
                coord_fmin[i] = max(coord_fmin[i], 0);
                coord_fmax[i] = (int)floorf(((float)coord[i]+0.5)/(lvl_res - 1)*(lvl_res_next - 1));
                coord_fmax[i] = min(coord_fmax[i], lvl_res_next-1);
            }
            // given these coordinates, add them to result
            for(int i=coord_fmin[2]; i<=coord_fmax[2]; i++) {
                for(int j=coord_fmin[1]; j<=coord_fmax[1]; j++) {
                    for(int k=coord_fmin[0]; k<=coord_fmax[0]; k++) {   // index is (k, j, i) 
                        coord[0] = k; coord[1] = j; coord[2] = i;
                        if(!out_of_bounds(coord, lvl_res_next)) {
                            // fetch this gradient
                            int index = compute_ravel_hash(coord, lvl_res_next, hashmap_size) + offset_lvl_next;
                            res += grad_output[index*batch_size*output_channels + b_idx*output_channels + c_idx];
                        }
                    }
                }
            }
            // coord_f[i] = ((float)coord[i]) / (lvl_res - 1) * (lvl_res_next - 1);  // float coordinate in next layer
            // given float coordinates, find all coordinates in next layer that have this as nearest neighbor
            // update local index
            _iter_local_n += hashmap_size;
        }
        // write the result
        grad_input[n_idx*batch_size*output_channels + b_idx*output_channels + c_idx] = res;
    }
}

template <typename scalar_t>
void abstract_contextlayer_backward_wrapper(const scalar_t* grad_output, scalar_t* grad_input, const int *offsets, const int *resolutions,
            const int batch_size, const int num_embeddings, const int input_channels, const int num_levels, const int hashmap_size) {
    // Simply call the kernel
    const uint32_t blocks = min(div_up(num_embeddings*batch_size*input_channels, THREADS), 1<<30-1);
    abstract_contextlayer_backward_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output, grad_input, offsets, resolutions, batch_size, num_embeddings, input_channels, num_levels, hashmap_size
    );
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());
}

template <typename scalar_t>
void abstract_contextlayer_forward_wrapper(const scalar_t *input, scalar_t *output, const int *offsets, const int *resolutions,
        const int batch_size, const int num_embeddings, const int input_channels, const int num_levels, const int hashmap_size) {
    
    const uint32_t blocks = min(div_up(num_embeddings*batch_size*input_channels, THREADS), 1<<30 - 1);
    abstract_contextlayer_forward_kernel<<<blocks, THREADS, 0,  at::cuda::getCurrentCUDAStream()>>>(
        input, output, offsets, resolutions, batch_size, num_embeddings, input_channels, num_levels, hashmap_size);
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());
}

torch::Tensor abstract_contextlayer_forward(torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
                         int num_levels, int hashmap_size) {
    CHECK_CUDA(input);
    CHECK_CUDA(output);
    CHECK_CUDA(offsets);
    CHECK_CUDA(resolutions);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(output);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(resolutions);

    if(input.dim() != 3) {
        throw std::runtime_error("Input must have 3 dimensions");
    }
    // define extra variables
    const int num_embeddings = input.size(0);
    const int batch_size = input.size(1);
    const int input_channels = input.size(2);
    if(input.size(2) != output.size(2)) {
        throw std::runtime_error("Input and output must have same sizes");
    }
    // kernel sizes (first index is levels)

    AT_DISPATCH_FLOATING_TYPES( 
    input.scalar_type(), "abstract_contextlayer_forward_wrapper", ([&] {
        abstract_contextlayer_forward_wrapper<scalar_t>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), offsets.data_ptr<int>(), resolutions.data_ptr<int>(), 
            batch_size, num_embeddings, input_channels, num_levels, hashmap_size);
    }));
    return output;
}

torch::Tensor abstract_contextlayer_backward(torch::Tensor grad_output, torch::Tensor grad_input, 
        torch::Tensor offsets, torch::Tensor resolutions, int num_levels, int hashmap_size) {
    // define extra variables
    const int num_embeddings = grad_output.size(0);
    const int batch_size = grad_output.size(1);
    const int input_channels = grad_output.size(2);

    AT_DISPATCH_FLOATING_TYPES( 
    grad_output.scalar_type(), "abstract_contextlayer_backward_wrapper", ([&] {
        abstract_contextlayer_backward_wrapper<scalar_t>(grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(), offsets.data_ptr<int>(), resolutions.data_ptr<int>(), 
            batch_size, num_embeddings, input_channels, num_levels, hashmap_size);
    }));
    return grad_input; 
}