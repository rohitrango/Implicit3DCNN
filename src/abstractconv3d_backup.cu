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