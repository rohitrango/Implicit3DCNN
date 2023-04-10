// Attempt to parallelize the convolution forward over batches and coalesce the input/output channels in
// the threads
// However, this is very slow for small batch sizes, and doesnt pass unittest
// But for bigger batch sizes (~16), the time is virtually the same as for batch size = 1
// TODO: Make this pass unittests
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
    // For each block, fetch the embedding id, block number and output
    int n = blockIdx.x;
    int kernel_idx = blockIdx.z;
    // unravel the kernel_index  (offset by half the kernel size)
    int kernel_volume = K1*K2*K3;
    int k1 = (kernel_idx/(K2*K3)) - K1/2;
    int k2 = ((kernel_idx % (K2*K3))/K3) - K2/2;
    int k3 = (kernel_idx%K3) - K3/2;

    int iosize = input_channels*output_channels;
    int prev_level = -1;

    int max_channels = max(input_channels, output_channels);
    int max_batch_sizes_per_block = THREADS/max_channels;

    // this is the composition of (channel_in * channel_out) number (for more coalesced memory access)
    // get the level of the table
    while(n < num_embeddings) {
        // shared memory
        __shared__ scalar_t weight_[64 * 64];
        __shared__ scalar_t res_[THREADS];
        __shared__ scalar_t inp_[THREADS];
        __shared__ scalar_t bias_[64];

        // get level related stuff
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
        // find x-coordinate if there is
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
        if(local_n >= lvl_res3) {
            n += gridDim.x;
            continue;
        }
        // check for level and load weights if needed
        if(prev_level != level) {
            // load weights
            for(int i=threadIdx.x; i<iosize; i+=THREADS)
                weight_[i] = weights[level*(kernel_volume*iosize) + kernel_idx*iosize + i];
            // load bias if exists and first kernel
            if(threadIdx.x < output_channels) {
                if((bias != NULL) && kernel_idx==0) 
                    bias_[threadIdx.x] = bias[level*output_channels + threadIdx.x];
                else
                    bias_[threadIdx.x] = 0;
            }
        }
        __syncthreads();
        prev_level = level;
        // get neighbor index
        int nbr = compute_ravel_hash(coord, lvl_res, hashmap_size) + offset_lvl;
        // __shared__ int b_start;
        int batch_size_div_up = ((batch_size + max_batch_sizes_per_block - 1)/max_batch_sizes_per_block) * max_batch_sizes_per_block;
        // fetch the input, looping over batch sizes
        // for(int b=threadIdx.x/max_channels; b<batch_size_div_up; b+=max_batch_sizes_per_block) {
        for(int b=0; b<batch_size_div_up; b+=max_batch_sizes_per_block) {
            // if(threadIdx.x == 0) 
                // b_start = b;
            // __syncthreads();
            int batch_remaining_this_loop = min(max_batch_sizes_per_block, batch_size - b);
            // now that threads are synced, feed in the input
            if(threadIdx.x < batch_remaining_this_loop*input_channels) {
                inp_[threadIdx.x] = input[nbr*(batch_size*input_channels) + b*input_channels + threadIdx.x];
            }
            __syncthreads();
            // multiply this batch with weights to get output batch
            if(threadIdx.x < batch_remaining_this_loop*output_channels) {
                int b_idx = threadIdx.x/output_channels;
                int c_out = threadIdx.x%output_channels;
                res_[threadIdx.x] = 0;
                for(int c_in=0; c_in<input_channels; c_in++) {
                    res_[threadIdx.x] += weight_[c_in*output_channels + c_out] * inp_[b_idx*input_channels + c_in];
                }
                res_[threadIdx.x] += bias_[c_out];
                // append it to output
                atomicAdd(&output[n*(batch_size*output_channels) + b*output_channels + threadIdx.x], res_[threadIdx.x]);
            }
            __syncthreads();
        }
        n += gridDim.x;
    }
}

// TODO: This version is super slow (idk why). Best to keep the same version as forward 
// for the backward pass
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