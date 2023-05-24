'''
Implementation of LayerNorm for Implicit Hashtables
'''
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import time
from tqdm import tqdm
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, num_channels, resolutions, offsets, affine=True):
        ''' 
        :num_channels: number of channels in each layer
        :resolutions: list of resolutions of each layer (not needed but keep it for consistency with other layers)
        :offsets: list of starting offset of each layer 
        :affine: If True, have learnable mean and std layers
        '''
        super().__init__()
        self.num_channels = num_channels
        self.num_levels = len(resolutions)
        self.offsets = offsets.cpu()
        self.affine = affine
        if affine:
            self.mean = nn.Parameter(torch.zeros(self.num_levels, 1, 1, self.num_channels))
            self.logstd = nn.Parameter(torch.zeros(self.num_levels, 1, 1, self.num_channels))
    
    def forward(self, x):
        ''' given hashtable x of size (N, B, C), use layernorm '''
        y = torch.zeros_like(x)
        for i in range(self.num_levels):
            xchunk = x[self.offsets[i]:self.offsets[i+1]]
            xchunk = (xchunk - torch.mean(xchunk, 0, keepdim=True)) / (torch.std(xchunk, 0, keepdim=True) + 1e-10)
            if self.affine:
                xchunk = xchunk * torch.exp(self.logstd[i]) + self.mean[i]
            y[self.offsets[i]:self.offsets[i+1]] = xchunk
        return y

if __name__ == '__main__':
    import gridencoder as ge
    from time import time
    L = 19
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[:, None].contiguous() * 1e3  # [1, N, 2]
    embed = embed.detach()

    ln = LayerNorm(2, encoder.resolutions, encoder.offsets).cuda()
    a = time()
    out = ln(embed)
    loss = out.mean()
    loss.backward()
    print(time() - a)
    print(out.shape)