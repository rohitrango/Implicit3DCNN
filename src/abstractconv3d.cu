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

__device__ int neighbor_index(const int index, const int resolution, int hashmap_size, const int *delta) {
    // compute index from delta
    int delta_index = 0, delta_stride = 1;
    #pragma unroll
    for(int i=0; i<3; i++) {
        delta_index += delta[i]*delta_stride;
        delta_stride *= resolution;
    }
    return (index + delta_index + hashmap_size) % hashmap_size;
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
    // get the level of the table
    while(n < num_embeddings) {
        int c_idx = threadIdx.x;
        int level = get_level(offsets, n, num_levels);  // get the level of the table
        int offset_lvl = offsets[level];
        int local_n = n - offset_lvl;
        int lvl_res = resolutions[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        int iosize = input_channels*output_channels;
        // this is a bad embedding (padded to make it divisible by 8), skip it
        if(local_n >= lvl_res3) {
            n += gridDim.x;
            continue;
        }
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
            __syncthreads();
            int x_index;

            int coord[3];
            unravel_index(local_n, lvl_res, coord);
            coord[0] += k1;
            coord[1] += k2;
            coord[2] += k3;
            while((local_n < lvl_res3) && out_of_bounds(coord, lvl_res)) {
                local_n += hashmap_size;
                unravel_index(local_n, lvl_res, coord);
                coord[0] += k1;
                coord[1] += k2;
                coord[2] += k3;
            }
            if(local_n < lvl_res3) {
                x_index = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl; // global offset
                x_index = x_index*(batch_size*input_channels) + b*input_channels;
                // read input
                if(threadIdx.x < input_channels) {
                    inp_[threadIdx.x] = input[x_index + threadIdx.x];
                }
                __syncthreads();
                // only the first batch gets to pull out the input
                res_[threadIdx.x] += weight_[threadIdx.x]*inp_[c_in];
            }
            c_idx += THREADS;
        }
        // we have res[THREAD] = sum_{partial c_in} w_{c_in, c_out} * x_{c_in} 
        if(threadIdx.x < output_channels) {
            for(int i=threadIdx.x+output_channels; i<min(iosize, THREADS); i+=output_channels) {
                res_[threadIdx.x] += res_[i];
            }
            atomicAdd(output + b*output_channels + n*batch_size*output_channels + threadIdx.x, res_[threadIdx.x]+bias_[threadIdx.x]);
        }
        n += gridDim.x;
    }
}

template <typename scalar_t>
__global__ void abstract_conv3d_backward_input_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    const scalar_t* __restrict__ input,
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
    // get starting `n` and kernel index
    int n = blockIdx.x;
    int kernel_idx = blockIdx.z;
    int max_channels = max(input_channels, output_channels);
    int max_batches_per_block = THREADS / max_channels;      // use this to load dL/dy and x
    
    __shared__ scalar_t weight_[64 * 64];
    __shared__ scalar_t grad_out_[THREADS];
    __shared__ scalar_t res_[THREADS];

    // keep track of previous level
    int prev_level = -1;
    // unravel the kernel_index  (offset by half the kernel size)
    int kernel_volume = K1*K2*K3;
    int k1 = (kernel_idx/(K2*K3)) - K1/2;
    int k2 = ((kernel_idx % (K2*K3))/K3) - K2/2;
    int k3 = (kernel_idx%K3) - K3/2;
    int iosize = input_channels*output_channels;

    // this is the composition of (channel_in * channel_out) number (for more coalesced memory access)
    // get the level of the table
    while(n < num_embeddings) {
        int level = get_level(offsets, n, num_levels);  // get the level of the table
        int offset_lvl = offsets[level];
        int local_n = n - offset_lvl;
        int lvl_res = resolutions[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        // this is a bad embedding (padded to make it divisible by 8), skip it
        if(local_n >= lvl_res3) {
            n += gridDim.x;
            continue;
        }
        // check if a dL/dy(n-del(x)) exists for dL/dx(n)
        // for y[n] we found x[n + dx] so for x'[n] we need to find y'[n - dx]
        int coord[3];
        int _iter_local_n = local_n;
        unravel_index(_iter_local_n, lvl_res, coord);
        coord[0] -= k1;
        coord[1] -= k2;
        coord[2] -= k3;
        while(_iter_local_n < lvl_res3 && out_of_bounds(coord, lvl_res)) {
            _iter_local_n += hashmap_size;
            unravel_index(_iter_local_n, lvl_res, coord);
            coord[0] -= k1;
            coord[1] -= k2;
            coord[2] -= k3;
        }
        // for this dL/dx(n) we don't have a dL/dy(n-del(x)) so skip it for all batches
        if(_iter_local_n >= lvl_res3) {
            n += gridDim.x;
            continue;
        }
        int nbr = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl; // compute neighbor
        // if we jumped to a new resolution, then load the corresponding weights
        if(level != prev_level) {
            for(int i=threadIdx.x; i<iosize; i+=THREADS) {
                weight_[i] = weights[level*(kernel_volume*iosize) + kernel_idx*iosize + i];
            }
        }
        // update prev level to this one
        prev_level = level;
        // load the grad_output
        int b_idx = threadIdx.x / max_channels;
        // int c_idx = threadIdx.x % max_channels;
        __shared__ int start_b;
        // iterate over batch sizes  [b_0, b_1 .... b_n]
        int batch_size_div_up = ((batch_size + max_batches_per_block - 1) / max_batches_per_block) * max_batches_per_block;
        for(int b=b_idx; b < batch_size_div_up; b+=max_batches_per_block) {
            if(threadIdx.x == 0) {
                start_b = b;
            }
            __syncthreads();
            int batch_remaining_this_loop = min(max_batches_per_block, batch_size - start_b);   // check how many batches are left in this loop
            // fetch dL/dy[n-dx]
            if(threadIdx.x < batch_remaining_this_loop*output_channels) {
                grad_out_[threadIdx.x] = grad_output[nbr*batch_size*output_channels + start_b*output_channels + threadIdx.x];
            }
            __syncthreads();
            // now compute result
            if(threadIdx.x < batch_remaining_this_loop*input_channels) {
                int bidx = threadIdx.x / input_channels;
                int cidx = threadIdx.x % input_channels; // input channel index
                res_[threadIdx.x] = 0;
                for(int i=0; i<output_channels; i++) {
                    res_[threadIdx.x] += weight_[cidx*output_channels + i] * grad_out_[bidx*output_channels + i];
                }
                atomicAdd(&grad_input[n*batch_size*input_channels + bidx*input_channels + cidx], res_[threadIdx.x]);
            }
        }
        // increment the embedding index
        n += gridDim.x;
    }
}

template <typename scalar_t>
__global__ void abstract_conv3d_backward_weight_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    scalar_t* __restrict__ grad_bias,
    const scalar_t* __restrict__ input,
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
    // // get information about block
    int kernel_volume = K1*K2*K3;
    int kernel_idx = blockIdx.x % kernel_volume;
    int level = blockIdx.x / kernel_volume;

    // // calculate grad of weight[lvl, k1, k2, k3]
    int k1 = (kernel_idx/(K2*K3)) - K1/2;
    int k2 = ((kernel_idx % (K2*K3))/K3) - K2/2;
    int k3 = (kernel_idx%K3) - K3/2;
    // // from thread index, infer output channel, batch size, etc
    int channelmax = max(input_channels, output_channels);
    int channelmin = min(input_channels, output_channels);

    // auxiliary variables (to enable coalesced memory access)
    int max_ns_per_block = THREADS / (channelmax*batch_size);  
    int max_batch_sizes_per_block = min(batch_size, THREADS/channelmax);
    // level variables (starting offset and resolution)
    int offset_lvl = offsets[level];
    int lvl_res = resolutions[level];
    int lvl_res3 = lvl_res*lvl_res*lvl_res;
    int lvl_size = min(lvl_res3, hashmap_size); /// total number of hash entries in this level
    int iosize = input_channels*output_channels;

    // get channel index, batch size and embedding index
    int c_idx = threadIdx.x % channelmax;
    int b_start = (threadIdx.x/channelmax)%batch_size;
    int local_n = threadIdx.x / (channelmax*batch_size);

    __shared__ scalar_t grad_weight_[64 * 64];  // max of [input_channels * output_channels] <= 64 * 64
    __shared__ scalar_t grad_bias_[64];   // [num_lvl, out_channels] 
    __shared__ scalar_t inp_[THREADS];         // [input_channels]
    __shared__ scalar_t grad_out_[THREADS];

    // // init grad weight and bias for index [lvl, k1, k2, k3] for weight and index [lvl] for bias
    for(int i=threadIdx.x; i<iosize; i+=THREADS) {
        grad_weight_[i] = 0;
    }
    if(threadIdx.x < 64)
        grad_bias_[threadIdx.x] = 0;  // add dL/dy from all n and b
    __syncthreads();

    __shared__ int current_batch_block_;   // to keep starting batch in sync for all threads (if part of loop)
    __shared__ int current_n_block_;       // to keep `starting n` in sync for all threads (else part of loop)
    __shared__ uint8_t valid_n_thread_[THREADS]; // to keep track if this n is valid or not 

    // // if max_ns_per_block = 0, we have big batch size, and therefore we have to loop over n and batch size
    if(max_ns_per_block == 0) {
        while(local_n < lvl_size) {
            // fetch dL/dy[n, b] first
            for(int b=b_start; b<batch_size; b+=max_batch_sizes_per_block) {
                // save the starting number 
                if(threadIdx.x == 0) {
                    current_batch_block_ = b;
                }
                __syncthreads();
                if(threadIdx.x < output_channels*max_batch_sizes_per_block) {
                    int cur_batch_id = threadIdx.x / output_channels + current_batch_block_;
                    grad_out_[threadIdx.x] = grad_output[(local_n+offset_lvl)*(batch_size*output_channels) + cur_batch_id*output_channels + threadIdx.x%output_channels];
                }
                __syncthreads();
                // now fetch the inputs from locations that contributed to this y_n  (x s.t. h(x) = local_n)
                int _iter_local_n = local_n;
                int nbr;
                int coord[3];
                unravel_index(_iter_local_n, lvl_res, coord);
                coord[0] += k1;
                coord[1] += k2;
                coord[2] += k3;
                while(_iter_local_n < lvl_res3 && out_of_bounds(coord, lvl_res)) {
                    // add the offset
                    _iter_local_n += hashmap_size;
                    // compute new neighbor coordinates
                    unravel_index(_iter_local_n, lvl_res, coord);
                    coord[0] += k1;
                    coord[1] += k2;
                    coord[2] += k3;
                }
                // if there is some neighbor that is in bounds, then take the ratio
                if(_iter_local_n < lvl_res3) {
                    nbr = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl;   // global index of neighbor
                    if(threadIdx.x < input_channels*max_batch_sizes_per_block) {
                        int cur_batch_id = threadIdx.x / input_channels + current_batch_block_;
                        inp_[threadIdx.x] = input[nbr*batch_size*input_channels + cur_batch_id*input_channels + threadIdx.x%input_channels];
                    }
                    __syncthreads();
                    // add it to weights
                    for(int i=threadIdx.x; i<iosize; i+=THREADS) {
                        int cout = i % output_channels;
                        int cin  = i / output_channels;
                        // add dL/dy[n, b, out] * dL/dx[n, b, in]
                        for(int _b=0; _b<max_batch_sizes_per_block; _b++) {
                            grad_weight_[i] += grad_out_[_b*output_channels + cout] * inp_[_b*input_channels + cin];
                        } 
                    }
                }
                // copy grad bias
                if(grad_bias && threadIdx.x < output_channels*max_batch_sizes_per_block) {
                    for(int _b=0; _b<max_batch_sizes_per_block; _b++) {
                        grad_bias_[threadIdx.x] += grad_out_[_b*output_channels + threadIdx.x];
                    }
                }
                __syncthreads();
            }
            local_n++;
        }
    }
    else {
        // updivide the number of entries to a multiple of `max_ns_per_block` because we want all threads to sync
        int lvl_size_div_up = ((lvl_size + max_ns_per_block - 1) / max_ns_per_block) * max_ns_per_block;
        while(local_n < lvl_size_div_up) {
            // this is the number of blocks that are remaining (if there is non divisible by max_ns_per_block), we want to keep track of them
            if(threadIdx.x == 0) {
                current_n_block_ = local_n;
            }
            __syncthreads();
            // this variable captures the maximum number of n's to consider
            int num_n_in_block = min(max_ns_per_block, lvl_res3 - current_n_block_);
            // fetch dL/dy for all n
            if(threadIdx.x < output_channels * batch_size * num_n_in_block) {
                int cur_n_id = threadIdx.x / (output_channels*batch_size) + current_n_block_;
                int batch_id = (threadIdx.x / output_channels) % batch_size;
                grad_out_[threadIdx.x] = grad_output[(cur_n_id+offset_lvl)*(batch_size*output_channels) + batch_id*output_channels + threadIdx.x%output_channels];
            }
            valid_n_thread_[threadIdx.x] = 0;   // everything is invalid by default
            inp_[threadIdx.x] = 0;
            __syncthreads();         
            // now fetch the inputs from any locations that contributed to this y_n  (x s.t. h(x) = local_n)
            if(local_n < lvl_size) {
                int _iter_local_n = local_n;
                int nbr;
                int coord[3];
                unravel_index(_iter_local_n, lvl_res, coord);  // find neighbor
                coord[0] += k1;
                coord[1] += k2;
                coord[2] += k3;
                while(_iter_local_n < lvl_res3 && out_of_bounds(coord, lvl_res)) {
                    _iter_local_n += hashmap_size;
                    unravel_index(_iter_local_n, lvl_res, coord);
                    coord[0] += k1;
                    coord[1] += k2;
                    coord[2] += k3;
                }
                // we found a valid index
                if(_iter_local_n < lvl_res3) {
                    valid_n_thread_[threadIdx.x] = 1;
                    nbr = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl;   // global index of neighbor
                    // only first `input_channels` will fetch this
                    if(c_idx < input_channels) {
                        inp_[threadIdx.x] = input[nbr*batch_size*input_channels + b_start*input_channels + c_idx];
                    }
                }
            }
            __syncthreads();
            // dL/dy are stored in the first ``batch_size * n * output_channels`` blocks of shared memory
            // however, dL/dx are stored in the ``THREADS`` blocks, padded with `channelmax - c_in` memory 

            // add to weights 
            // if number of io is large then split the computation across weight indices
            for(int i=threadIdx.x; i<iosize; i+=THREADS) {
                int cout = i % output_channels;
                int cin  = i / output_channels;
                // for this weight [cin, cout], iterate over batch_size * n
                // for each dL/dy, see if the `dy/dx` is valid too
                for(int _bn=0; _bn<batch_size*num_n_in_block; _bn++) {
                    if(valid_n_thread_[_bn*channelmax + cin])
                        grad_weight_[i] += grad_out_[_bn*output_channels + cout] * inp_[_bn*channelmax + cin];
                } 
            }
            // TODO: add an `else` case where we have a large batch size

            // copy grad bias
            if(grad_bias && threadIdx.x < output_channels) {
                for(int _b=0; _b<batch_size*num_n_in_block; _b++) {
                    grad_bias_[threadIdx.x] += grad_out_[_b*output_channels + threadIdx.x];
                }
            }
            // move to next set of n's
            local_n += max_ns_per_block;
        }
    }            
    __syncthreads();
    // // write values for this level
    for(int i=threadIdx.x; i<iosize; i+=THREADS) {
        grad_weight[level*(kernel_volume*iosize) + kernel_idx*iosize + i] = grad_weight_[i];
    }
    if(grad_bias && threadIdx.x < output_channels) {
        grad_bias[level*output_channels + threadIdx.x] = grad_bias_[threadIdx.x];
    }
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

    abstract_conv3d_forward_kernel_v3<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    const bool inp_requires_grad,
    const scalar_t* __restrict__ input,
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
    if(inp_requires_grad) {
        const dim3 blocks(num_embeddings/32, 1, K1*K2*K3);
        // call gradient w.r.t. input
        abstract_conv3d_backward_input_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output, grad_input, 
            input, offsets, resolutions, weights, bias,
            batch_size, num_embeddings, input_channels, output_channels,
            K1, K2, K3, num_levels, hashmap_size
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    int weightblocks = K1*K2*K3*num_levels;
    abstract_conv3d_backward_weight_kernel<<<weightblocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_output, grad_weights, grad_bias,
        input, offsets, resolutions, weights, bias,
        batch_size, num_embeddings, input_channels, output_channels,
        K1, K2, K3, num_levels, hashmap_size
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

torch::Tensor abstract_conv3d_forward(torch::Tensor input, torch::Tensor output, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size) {
    /* 
    
    input: input hash entries    (N, B, C_in)
    output: output hash entries  (N, B, C_out)
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
    const int num_embeddings = input.size(0);
    const int batch_size = input.size(1);
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
        bool inp_requires_grad, torch::Tensor input, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size) {
    /* Kernel for backward pass */
    // define extra variables
    const int num_embeddings = input.size(0);
    const int batch_size = input.size(1);
    const int input_channels = input.size(2);
    const int output_channels = grad_output.size(2);
    // kernel sizes
    const int k1 = weight.size(1);
    const int k2 = weight.size(2);
    const int k3 = weight.size(3);
    // call kernels
    AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "abstract_conv3d_backward_wrapper", ([&] {
        abstract_conv3d_backward_wrapper<scalar_t>(
            grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(), grad_weight.data_ptr<scalar_t>(), grad_bias.has_value() ? grad_bias.value().data_ptr<scalar_t>() : nullptr,
            inp_requires_grad, input.data_ptr<scalar_t>(), offsets.data_ptr<int>(), resolutions.data_ptr<int>(), weight.data_ptr<scalar_t>(), \
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr, batch_size, num_embeddings, input_channels, output_channels, k1, k2, k3, num_levels, hashmap_size);
    }));
    // return ret;
    return {grad_input, grad_weight, grad_bias};
}