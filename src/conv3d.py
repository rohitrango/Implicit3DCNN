'''
Implementation of 3D conv layer for abstract grids
'''
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from backend import _backend

class _abstract_conv3d(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias):
        ctx.save_for_backward(input, offset, weight, bias)
        output = _backend.abstract_conv3d_forward(input, offset, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return None


class AbstractConv3D(nn.Module):
    ''' Actual nn Module that implements the 3D convolution layer'''
    def __init__(self, channels_in, channels_out, resolutions, offsets, kernel_size, bias=True, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        # load extra params just in case        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # load weights
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 3, "Input a "
        # self.resolutions = resolutions   # list of grid sizes/resolutions
        # self.offsets = offsets           # list of offsets, should be int32
        self.register_buffer('resolutions', torch.tensor(resolutions, dtype=torch.int32))
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.int32))
        # load weights now
        self.register_parameter('weight', nn.Parameter(torch.Tensor(channels_out, channels_in, *kernel_size)))
        self.register_parameter('bias', nn.Parameter(torch.Tensor(channels_out)) if bias else None)


    def forward(self, input):
        return _abstract_conv3d.apply(input, self.weight, self.weight, self.bias)

## Check this
if __name__ == '__main__':
    layer = AbstractConv3D(3, 16, [16, 32, 64], [0, 0, 0], 3).cuda()
    input = torch.rand(1, 3, 16, 16, 16).cuda()
    output = layer(input)
    print(output.shape)
    print(output - input)