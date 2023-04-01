'''
Implementation of 3D conv layer for abstract grids
'''
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from backend import _backend
import time

class _abstract_conv3d(Function):
    @staticmethod
    def forward(ctx, input, offsets, resolutions, weight, bias, num_levels, hashmap_size):
        ''' Forward pass

        :input: [batch_size, num_embeddings, channels_in]
        :weight: [num_levels, kernel_size, kernel_size, kernel_size, channels_in, channels_out]
        :bias: [num_levels, channels_out]
        '''
        batch_size, num_embedding = input.shape[:2]
        channels_out = weight.shape[-1]
        output = torch.zeros((batch_size, num_embedding, channels_out), device=input.device, dtype=input.dtype)
        # Save backward tensors and return output
        output = _backend.abstract_conv3d_forward(input, output, offsets, resolutions, weight, bias, num_levels, hashmap_size)
        ctx.save_for_backward(input, output, offsets, resolutions, weight, bias, torch.tensor(num_levels), torch.tensor(hashmap_size))
        return output
    
    @staticmethod
    def backward(ctx, grad_outputs):
        ''' Backward pass

        :grad_outputs: [batch_size, num_embeddings, channels_out]
        '''
        input, output, offsets, resolutions, weight, bias, num_levels, hashmap_size = ctx.saved_tensors
        num_levels = int(num_levels.item())
        hashmap_size = int(hashmap_size.item())
        # get outputs
        input_grad = torch.zeros_like(input, dtype=input.dtype, device=input.device)
        weight_grad = torch.zeros_like(weight, dtype=weight.dtype, device=weight.device)
        bias_grad   = torch.zeros_like(bias, dtype=bias.dtype, device=bias.device) if bias is not None else None
        input_grad, weight_grad, bias_grad = _backend.abstract_conv3d_backward(grad_outputs, input_grad, weight_grad, bias_grad, \
                                                                               input, output, offsets, resolutions, weight, bias, num_levels, hashmap_size)
        # backward pass
        return input_grad, None, None, weight_grad, bias_grad, None, None


class AbstractConv3D(nn.Module):
    ''' Actual nn Module that implements the 3D convolution layer'''
    def __init__(self, channels_in, channels_out, resolutions, offsets, kernel_size, bias=True, num_levels=16,
                 log_hashmap_size=19):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        # hash encoding params
        self.num_levels = num_levels
        self.hashmap_size = int(2**log_hashmap_size)
        # load weights
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 3, "kernel_size should be int or tuple/list of length 3"
        ## list of grid sizes/resolutions, and offsets, should be int32
        self.register_buffer('resolutions', resolutions.clone().detach().int())
        self.register_buffer('offsets', offsets.clone().detach().int())
        ### load weights now
        self.register_parameter('weight', nn.Parameter(0.1*torch.randn(num_levels, *kernel_size, channels_in, channels_out)))   # keep channels_in at the end to 
        self.register_parameter('bias', nn.Parameter(0.01*torch.rand(num_levels, channels_out)) if bias else None)

    def forward(self, input):
        ''' forward pass 
        input: (B, N, C_in)
        '''
        return _abstract_conv3d.apply(input, self.offsets, self.resolutions, self.weight, self.bias, self.num_levels, self.hashmap_size)

## Check this
if __name__ == '__main__':
    import gridencoder as ge
    L = 19
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[None] * 10  # [1, N, 2]
    # embed = embed.expand(4, -1, -1).contiguous()
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    print(embed.shape, resolutions.shape, offsets.shape)

    layer = AbstractConv3D(2, 4, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    a = time.time()
    output = layer(embed)
    print(time.time() - a)
    print(output.min(), output.max(), output.shape)
    layer2 = AbstractConv3D(4, 32, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()

    a = time.time()
    output = layer2(output)
    print(time.time() - a)
    print(output.min(), output.max(), output.shape)

    # layer = AbstractConv3D(32, 2, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    # a = time.time()
    # output = layer(output)
    # print(time.time() - a)
    # print(output.min(), output.max(), output.shape)
