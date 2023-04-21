'''
Implementation of 3D abstract upsampling layer for abstract grids
'''
import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as F
from torch.autograd import Function
from networks.backend import _backend_context
from gridencoder.grid import grid_encode
import time
from tqdm import tqdm
import numpy as np

class _abstract_context(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, offsets, resolutions, num_levels, hashmap_size):
        if input.requires_grad:
            ctx.save_for_backward(offsets, resolutions, torch.tensor(num_levels), torch.tensor(hashmap_size))
        output = torch.zeros_like(input, dtype=input.dtype, device=input.device)
        output = _backend_context.abstract_contextlayer_forward(input, output, offsets, resolutions, num_levels, hashmap_size)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        offsets, resolutions, num_levels, hashmap_size = ctx.saved_tensors
        num_levels, hashmap_size = int(num_levels.item()), int(hashmap_size.item())
        grad_input = torch.zeros_like(grad_outputs, dtype=grad_outputs.dtype, device=grad_outputs.device)
        grad_input = _backend_context.abstract_contextlayer_backward(grad_outputs, grad_input, offsets, resolutions, num_levels, hashmap_size)
        return grad_input, None, None, None, None

# Get function
abstractContextFunction = _abstract_context.apply

class AbstractContextLayer(nn.Module):
    ''' Couples a context function with an affine transformation (similar to a resnet layer but from the previous layer) '''
    def __init__(self, channels_in, channels_out, resolutions, offsets, affine=None, num_levels=16, log_hashmap_size=19):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.resolutions = resolutions
        self.offsets = offsets
        if channels_in != channels_out:
            if affine == False:
                print("WARNING: affine is set to False, but channels_in != channels_out. Setting affine to True.")
                affine = True
        self.affine = None
        if affine:
            self.affine = nn.Linear(channels_in, channels_out)
            nn.init.zeros_(self.affine.bias)
            nn.init.kaiming_uniform_(self.affine.weight, nonlinearity='leaky_relu')
        # hash encoding params
        self.num_levels = num_levels
        self.hashmap_size = int(2**log_hashmap_size) 
    
    def forward(self, x):
        ''' x: [N, B, C] '''
        y = abstractContextFunction(x, self.offsets, self.resolutions, self.num_levels, self.hashmap_size)
        if self.affine is not None:
            y = self.affine(y)
        return y

if __name__ == '__main__':
    import gridencoder as ge
    L = 19
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[:, None].contiguous() * 1e3  # [1, N, 2]
    embed = embed.detach()
    print(embed.min(), embed.max())
    # embed = embed.expand(4, -1, -1).contiguous()
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    print(embed.shape, resolutions.shape, offsets.shape)

    context = AbstractContextLayer(2, 2, resolutions=resolutions, offsets=offsets, affine=False, num_levels=16, log_hashmap_size=L).cuda()
    y = context(embed)
    a = time.time()
    y = context(embed)
    print(time.time() - a)
