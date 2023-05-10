/*

v1: <B, N, C_out> blocks, very slow

v2: <B, N, K1,K2,K3> blocks, hopefully faster with lot of coalescing
*/
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <torch/torch.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cooperative_groups.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 512
#define CONST_DIV 64
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define div_up(x, n) ((x)+(n)-1)/(n)
// assume that x is positive and n is a power of 2
#define modpow2(x, n) ((x)&(n-1))    
#define divpow2(x, n) ((x)>>(31 - __clz(n)))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ __forceinline__ int compute_ravel_hash(const int *coord, const int resolution, const int hashmap_size) {
    // compute hash function for tiled grid
    int index;
    // int stride = 1;
    // #pragma unroll
    // for(int i=0; i<3; i++) {
    //     index += coord[i]*stride;
    //     stride *= resolution;
    // }
    index = coord[0] + resolution*(coord[1] + resolution*coord[2]);
    return modpow2(index, hashmap_size);
}

__device__ __forceinline__ void unravel_index(const int index, const int resolution, int* coord) {
    // unravel index 
    int res2 = resolution*resolution;
    coord[2] = index / res2;
    coord[1] = (index%res2) / resolution;
    coord[0] = index % resolution;
}

__device__ __forceinline__ bool out_of_bounds(const int* coord, const int resolution) {
    // check if the coordinate is out of bounds
    return coord[0] < 0 || coord[0] >= resolution || coord[1] < 0 || coord[1] >= resolution || coord[2] < 0 || coord[2] >= resolution;
}

__device__ __forceinline__ int get_level(const int* __restrict__  offsets, const int tableoffset, const int num_levels) {
    // given the tableoffset in the big hash table, get its level
    for(int i=1; i<num_levels; i++) {
        if(tableoffset < offsets[i]) {
            return i-1;
        }
    }
    return num_levels-1;
}

__device__ __forceinline__ int compute_diff_hash(const int k1, const int k2, const int k3, const int lvl_res, const int hashmap_size) {
    int index = k1 + lvl_res*(k2 + lvl_res*k3);
    // return index>=0?index:(index%hashmap_size + hashmap_size);    // if positive, return as is, if negative, its the remainder, so add hash map to it
    if(index >= 0)
        return index;
    return (hashmap_size - (modpow2(-index, hashmap_size)));
}

template <typename scalar_t>
__global__ void abstract_conv3d_forward_kernel_v4(
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
    const int num_outputs = batch_size*num_embeddings*output_channels;
    const int kernel_volume = K1*K2*K3;
    const int iosize = input_channels*output_channels;

    // pull this into shared memory
    __shared__ int resolutions_shared[32];
    __shared__ int offsets_shared[32];

    if(threadIdx.x < num_levels) {
        resolutions_shared[threadIdx.x] = resolutions[threadIdx.x];
    }
    if(threadIdx.x < num_levels+1) {
        offsets_shared[threadIdx.x] = offsets[threadIdx.x];
    }
    __syncthreads();

    // // to capture level bias in case
    int level_bias = -1;
    scalar_t bias_saved = 0;

    CUDA_KERNEL_LOOP(index, num_outputs) {
        // get n, batch index, and output channel index
        // int c_out = index % output_channels;
        int c_out = modpow2(index, output_channels);    // assume channels are powers of 2 
        int ibyout = divpow2(index, output_channels);
        int b_idx = modpow2(ibyout, batch_size);
        int n_idx = divpow2(ibyout, batch_size);
        // get level information
        int level = get_level(offsets_shared, n_idx, num_levels);
        int offset_lvl = offsets_shared[level];
        int local_n = n_idx - offset_lvl;
        int lvl_res = resolutions_shared[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        // if this is a tail end, skip t
        if(local_n >= lvl_res3)
            continue;
        // now we have n, b, c --> time to get y[n, b, cout]
        scalar_t res = 0;
        // const scalar_t* weight_start = weights + level*kernel_volume*iosize;
        int weight_index = level*kernel_volume*iosize;
        // cache this result in the beginning
        int coordstart[3];
        unravel_index(local_n, lvl_res, coordstart);

        for(int k1idx=0; k1idx<K1; k1idx++) {
            int k1 = k1idx - K1/2;
            for(int k2idx=0; k2idx<K2; k2idx++) {
                int k2 = k2idx - K2/2;
                for(int k3idx=0; k3idx<K3; k3idx++) {
                    int k3 = k3idx - K3/2;
                    // get neighboring index
                    int x_index;
                    int coord[3];
                    #pragma unroll
                    for(int i=0; i<3; i++)
                        coord[i] = coordstart[i];

                    // resolve x index
                    if(lvl_res3 > hashmap_size) {   // this is a big resolution, simply compute the (x + dx) % T
                        x_index = modpow2((local_n + compute_diff_hash(k1, k2, k3, lvl_res, hashmap_size)), hashmap_size) + offset_lvl;
                    }
                    else {  // only one point corresponds to this n, find it
                        coord[0] += k1; coord[1] += k2; coord[2] += k3;
                        if(out_of_bounds(coord, lvl_res))
                            x_index = -1;
                        else
                            x_index = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl;
                    }
                    // compute if x_index != -1
                    if(x_index == -1) {
                        weight_index += iosize;
                    } 
                    else {
                        // loop in for all the x's
                        int input_index = input_channels*(x_index*batch_size + b_idx);
                        for(int c=0; c<input_channels; c++) {
                            res += weights[weight_index + c_out] * input[input_index + c];
                            weight_index += output_channels;
                        }
                    }
                }
            }
        }

        if(bias) {
            if(level_bias != level) {
                level_bias = level;
                bias_saved = bias[level*output_channels + c_out];
            }
            res += bias_saved;
        }
        // write
        output[index] = res;
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
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size
) {
    // For each block, fetch the embedding id, block number and output
    int num = batch_size*num_embeddings*input_channels;
    int kernel_volume = K1*K2*K3;
    int iosize = input_channels*output_channels;

    // pull this into shared memory
    __shared__ int resolutions_shared[32];
    __shared__ int offsets_shared[32];
    if(threadIdx.x < num_levels) {
        resolutions_shared[threadIdx.x] = resolutions[threadIdx.x];
    }
    if(threadIdx.x < num_levels+1) {
        offsets_shared[threadIdx.x] = offsets[threadIdx.x];
    }
    __syncthreads();

    CUDA_KERNEL_LOOP(index, num) {
        // get n, batch index, and output channel index
        int c_in = modpow2(index, input_channels);    // assume channels are powers of 2
        int ibyin = divpow2(index, input_channels);  
        int b_idx = modpow2(ibyin, batch_size); 
        int n_idx = divpow2(ibyin, batch_size);
        // get level information
        int level = get_level(offsets_shared, n_idx, num_levels);
        int offset_lvl = offsets_shared[level];
        int local_n = n_idx - offset_lvl;
        int lvl_res = resolutions_shared[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        // if this is a tail end, skip t
        if(local_n >= lvl_res3)
            continue;
        // now we have n, b, c_in --> time to get x'[n, b, cin]
        scalar_t res = 0;
        int weight_index = level*kernel_volume*iosize;
        int startcoord[3];
        unravel_index(local_n, lvl_res, startcoord);
        // loop over all the weights
        for(int k1idx=0; k1idx<K1; k1idx++) {
            int k1 = k1idx - K1/2;
            for(int k2idx=0; k2idx<K2; k2idx++) {
                int k2 = k2idx - K2/2;
                for(int k3idx=0; k3idx<K3; k3idx++) {
                    // for (k1, k2, k3), get weight
                    int k3 = k3idx - K3/2;
                    int y_index;
                    int coord[3];
                    #pragma unroll
                    for(int i=0; i<3; i++)
                        coord[i] = startcoord[i];

                    if(lvl_res3 > hashmap_size) {   // this is a big resolution, simply compute the (x + dx) % T
                        y_index = modpow2((local_n + compute_diff_hash(-k1, -k2, -k3, lvl_res, hashmap_size)), hashmap_size) + offset_lvl;
                    }
                    else {  // only one point corresponds to this n, find it
                        coord[0] -= k1; coord[1] -= k2; coord[2] -= k3;
                        if(out_of_bounds(coord, lvl_res))
                            y_index = -1;
                        else
                            y_index = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl;
                    }
                    // compute if x_index != -1
                    if(y_index == -1) {
                        weight_index += iosize;
                    }
                    else {
                        // loop in for all the x's
                        int grad_out_idx = output_channels*(y_index*batch_size + b_idx);
                        for(int c=0; c<output_channels; c++) {
                            res += weights[weight_index + c_in] * grad_output[grad_out_idx + c];
                            weight_index += input_channels;
                        }                    
                    }
                }
            }
        }
        grad_input[index] = res;
    }
}

/** Second version of computing gradient of weight kernel using the block scheme of y[n, b, c]
 * 
 * The idea is that for y[n, b, c], we should compute w[l, *, *, *, *, c] where l ---> n
 *  
 * dw[l, k1, k2, k3, cin, cout] = sum_n sum_b dy[n, b, cout] * x[(n + k), b, cin]
 * the first and second sums are divided into a temp index (which will be summed over) 
 * this hopefully achieves a good balance between the atomicAdd operation and block utilization
*/
template <typename scalar_t>
__global__ void abstract_conv3d_backward_weight_kernel_v2(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__  grad_weights_tmp,
    const scalar_t* __restrict__ input, 
    const int* __restrict__ offsets,
    const int* __restrict__ resolutions,
    const scalar_t* __restrict__ weights,
    const int* __restrict__ offsets_tmp,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels, const int hashmap_size
) {
    // loop over y, and add gradient  
    int num = batch_size * num_embeddings * output_channels;
    int iosize = input_channels * output_channels;
    int kernel_size = K1*K2*K3;

    CUDA_KERNEL_LOOP(index, num) { 
        // get n, batch index, and output channel index
        int c_out = modpow2(index, output_channels);
        int ibyout = divpow2(index, output_channels);
        int b_idx = modpow2(ibyout, batch_size);
        int n_idx = divpow2(ibyout, batch_size);
        // get level information
        int level = get_level(offsets, n_idx, num_levels);
        int offset_lvl = offsets[level];
        int local_n = n_idx - offset_lvl;
        int lvl_res = resolutions[level];
        int lvl_res3 = lvl_res*lvl_res*lvl_res;
        // if this is a tail end, skip t
        if(local_n >= lvl_res3)
            continue;
        // temp index for weight to be put in
        int grad_offset_start = offsets_tmp[level];
        int grad_offset_end = offsets_tmp[level+1];
        int grad_wt_offset = local_n % (grad_offset_end - grad_offset_start) + grad_offset_start;
        // get dy[n, b, c]
        scalar_t yval = grad_output[index];
        scalar_t grad_res;

        // store x-index
        int xindex;
        int startcoord[3];
        unravel_index(local_n, lvl_res, startcoord);
        // weight index to add to
        int wt_idx = grad_wt_offset*kernel_size*iosize + c_out;

        int kernel_idx=-1;
        for(int k1idx=0; k1idx<K1; k1idx++) {
            int k1 = k1idx - K1/2;
            for(int k2idx=0; k2idx<K2; k2idx++) {
                int k2 = k2idx - K2/2;
                for(int k3idx=0; k3idx<K3; k3idx++) {
                    int k3 = k3idx - K3/2;
                    int coord[3];
                    #pragma unroll
                    for(int i=0; i<3; i++)
                        coord[i] = startcoord[i];

                    if(lvl_res3 > hashmap_size) {   // this is a big resolution, simply compute the (x + dx) % T
                        xindex = modpow2((compute_diff_hash(k1, k2, k3, lvl_res, hashmap_size) + local_n), hashmap_size) + offset_lvl;
                    }
                    else {  // only one point corresponds to this n, find it
                        coord[0] += k1; coord[1] += k2; coord[2] += k3;
                        if(out_of_bounds(coord, lvl_res))
                            xindex = -1;
                        else
                            xindex = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl;
                    }
                    // compute if x_index != -1
                    kernel_idx++;
                    if(xindex == -1) {
                        wt_idx += iosize;
                        continue;
                    }
                    int inp_idx = input_channels*(xindex*batch_size + b_idx);
                    for(int c = 0; c < input_channels; c++) {
                        grad_res = yval * input[inp_idx];
                        inp_idx++;
                        atomicAdd(grad_weights_tmp + wt_idx, grad_res);
                        wt_idx+=output_channels;
                    } 
                    // weight idx += iosize has been done here
                }
            }
        }
        // for(int kernel_idx=0; kernel_idx < kernel_size; kernel_idx++) {
        //     // initialize grad weight
        //     // get kernel index, this is affected by
        //     int k1 = (kernel_idx/(K2*K3)) - K1/2;
        //     int k2 = ((kernel_idx % (K2*K3))/K3) - K2/2;
        //     int k3 = (kernel_idx%K3) - K3/2;
        //     // get neighboring index in x
        //     int xindex;
        //     int coord[3];
        //     unravel_index(local_n, lvl_res, coord);
        //     if(lvl_res3 > hashmap_size) {   // this is a big resolution, simply compute the (x + dx) % T
        //         xindex = (compute_diff_hash(k1, k2, k3, lvl_res, hashmap_size) + local_n) % hashmap_size + offset_lvl;
        //     }
        //     else {  // only one point corresponds to this n, find it
        //         coord[0] += k1; coord[1] += k2; coord[2] += k3;
        //         if(out_of_bounds(coord, lvl_res))
        //             xindex = -1;
        //         else
        //             xindex = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl;
        //     }
        //     // compute if x_index != -1
        //     if(xindex == -1)
        //         continue;
        //     // find x[xindex, b, c_in]
        //     for(int c = 0; c < input_channels; c++) {
        //         grad_res = yval * input[xindex*(batch_size*input_channels) + b_idx*input_channels + c];
        //         atomicAdd(grad_weights_tmp + (grad_wt_offset*kernel_size*iosize + kernel_idx*iosize + c*output_channels + c_out), grad_res);
        //     }
        // }
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
    // works good!
    // const uint32_t blocks = min((num_embeddings*batch_size*output_channels + THREADS - 1) / THREADS, 1<<30 - 1);
    const uint32_t blocks = min(div_up(num_embeddings*batch_size*output_channels, THREADS), 1<<30 - 1);
    abstract_conv3d_forward_kernel_v4<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        input, output, offsets, resolutions, weights, bias,
        batch_size, num_embeddings, input_channels, output_channels,
        K1, K2, K3, num_levels, hashmap_size
    );
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());
}


template <typename scalar_t>
void abstract_conv3d_backward_wrapper_v2(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_weights_tmp,
    const bool inp_requires_grad,
    const bool weight_requires_grad,
    const scalar_t* __restrict__ input,
    const int* __restrict__ offsets,
    const int* __restrict__ resolutions,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ weights_permute,
    const int* __restrict__ offsets_tmp,
    const int batch_size,
    const int num_embeddings,
    const int input_channels,
    const int output_channels,
    const int K1, const int K2, const int K3,
    const int num_levels,
    const int hashmap_size
) {
    if(inp_requires_grad) {
        const uint32_t blocks = min(div_up(num_embeddings*batch_size*input_channels, THREADS), 1<<30 - 1);
        // call gradient w.r.t. input
        abstract_conv3d_backward_input_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output, grad_input, 
            input, offsets, resolutions, weights_permute, 
            batch_size, num_embeddings, input_channels, output_channels,
            K1, K2, K3, num_levels, hashmap_size
        );
        // gpuErrchk(cudaPeekAtLastError());
        // gpuErrchk(cudaDeviceSynchronize());
    }
    if(weight_requires_grad) {
        const uint32_t blocks = min(div_up(num_embeddings*batch_size*output_channels, THREADS), 1<<30 - 1);
        abstract_conv3d_backward_weight_kernel_v2<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output, grad_weights_tmp, 
            input, offsets, resolutions, weights, offsets_tmp, 
            batch_size, num_embeddings, input_channels, output_channels,
            K1, K2, K3, num_levels, hashmap_size
        );
        // gpuErrchk(cudaPeekAtLastError());
        // gpuErrchk(cudaDeviceSynchronize());
    }
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
        bool inp_requires_grad, bool weight_requires_grad, torch::Tensor input, torch::Tensor offsets, torch::Tensor resolutions,
        torch::Tensor weight, at::optional<torch::Tensor> bias, int num_levels, int hashmap_size) {
    /* 
    ****** VERSION 1 ******
    Kernel for backward pass 
    */
    // define extra variables
    const int num_embeddings = input.size(0);
    const int batch_size = input.size(1);
    const int input_channels = input.size(2);
    const int output_channels = grad_output.size(2);
    // kernel sizes
    const int k1 = weight.size(1);
    const int k2 = weight.size(2);
    const int k3 = weight.size(3);

    /* 
    ****** VERSION 2 ******
    Use a temporary storage for storing gradient of weight
    gradient of bias is easy, simply loop over `num_levels` and take sum of that corresponding chunk
    */

    // compute factor to put temp weights in
    int cmin = min(input_channels, output_channels);
    int temp_weight_downsample_factor = div_up(k1*k2*k3*cmin, batch_size);

    bool grad_bias_exists = grad_bias.has_value();
    torch::Tensor grad_weight_tmp, offsets_tmp;
    offsets_tmp = offsets.clone();
    int total_tmp_sum = 0;   // number of entries in temp grad table
    torch::Tensor weight_permute;  // permuted weights for coalescing

    if(weight_requires_grad) {
        // compute bias level, and get tmp weight and offsets
        int offset_start = offsets.index({0}).item<int>();
        for(int i=0; i<num_levels; i++) {
            int offset_end = offsets.index({i+1}).item<int>();
            int lvl_res = resolutions.index({i}).item<int>();
            int level_size = offset_end - offset_start;
            if(grad_bias_exists)
                grad_bias.value().index({i}) = grad_output.slice(0, offset_start, offset_start+level_size).sum(0).sum(0);
            // for next level
            offset_start = offset_end;
            // set the offset for tmp weight too
            int tmp_level_size = div_up(level_size, temp_weight_downsample_factor);
            total_tmp_sum += tmp_level_size;   // total_sum = sum of all sizes for all levels upto i
            offsets_tmp.index({i+1}) = total_tmp_sum;
        }
        grad_weight_tmp = torch::zeros({total_tmp_sum, k1, k2, k3, input_channels, output_channels}, torch::TensorOptions().dtype(grad_weight.dtype()).layout(grad_weight.layout()).device(grad_weight.device().type(), grad_weight.device().index()));
    }
    else {
        grad_weight_tmp = torch::zeros({1}, torch::TensorOptions().dtype(grad_weight.dtype()).layout(grad_weight.layout()).device(grad_weight.device().type(), grad_weight.device().index()));
    }

    // if input requires grad, permute the weights channels dimensions 
    if(inp_requires_grad) {
        weight_permute = weight.permute({0, 1, 2, 3, 5, 4}).contiguous();
    }
    else {
        weight_permute = weight;
    }

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "abstract_conv3d_backward_wrapper", ([&] {
            abstract_conv3d_backward_wrapper_v2<scalar_t>(
                grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(), grad_weight_tmp.data_ptr<scalar_t>(), 
                inp_requires_grad, weight_requires_grad, input.data_ptr<scalar_t>(), offsets.data_ptr<int>(), 
                resolutions.data_ptr<int>(), weight.data_ptr<scalar_t>(), 
                weight_permute.data_ptr<scalar_t>(),
                offsets_tmp.data_ptr<int>(),
                batch_size, num_embeddings, input_channels, output_channels, k1, k2, k3, num_levels, hashmap_size);
    }));

    // TODO: Run aggregation over weight pointer
    if(weight_requires_grad) {
        int offset_start = offsets_tmp.index({0}).item<int>();
        int offset_end;
        for(int i=0; i<num_levels; i++)  {
            offset_end = offsets_tmp.index({i+1}).item<int>();
            grad_weight.index({i}) = grad_weight_tmp.slice(0, offset_start, offset_end).sum(0);
            offset_start = offset_end; // update the new starting point
        }
    }

    return {grad_input, grad_weight, grad_bias};
}
