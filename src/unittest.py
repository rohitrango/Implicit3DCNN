import torch
from torch import nn
import time
from conv3d import AbstractConv3D
from torch.nn import functional as F

if __name__ == '__main__':
    import gridencoder as ge
    L = 19
    batch = 1
    # randomly initialize a grid encoder
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[None] * 10000 # [1, N, 2]
    embed = embed.expand(batch, -1, -1).contiguous()
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    
    # define layer with zero bias
    layer = AbstractConv3D(2, 32, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    # layer.bias.data.zero_()
    print(embed.shape)
    a = time.time()
    output = layer(embed)
    print(time.time() - a)

    # compare with a regular convolution
    for i in range(encoder.num_levels):
        embed_lvl = embed[:, offsets[i]:offsets[i+1]]
        _, S, _ = embed_lvl.shape
        r = resolutions[i] 
        if S != r**3:
            break
        # reshape this
        embed_lvl = embed_lvl.reshape(batch, r, r, r, 2).permute(0, 4, 3, 2, 1).contiguous()  # [1, 2, r, r, r]
        conv_lvl = layer.weight[i].permute(4, 3, 0, 1, 2)
        # get conv output
        conv_out = F.conv3d(embed_lvl, conv_lvl, bias=layer.bias[i], stride=1, padding=1).permute(0, 4, 3, 2, 1).contiguous()  # [1, r, r, r, 4]
        # conv_out = conv_out[:, 1:-1, 1:-1, 1:-1, :].contiguous()
        # compare with output 
        out = output[:, offsets[i]:offsets[i+1]].reshape(batch, r, r, r, -1).contiguous()
        # out = out[:, 1:-1, 1:-1, 1:-1, :].contiguous()
        print(conv_out.shape, out.shape, embed_lvl.shape)
        print("abs-diff ", (conv_out - out).abs().max().item(), "conv abs mean value ", conv_out.abs().mean().item())
        # print(conv_out.abs().max())
        diff = (conv_out - out).abs()
        conv_out = torch.clamp(conv_out.abs(), 1e-1)
        # print([x.item() for x in [conv_out.min(), conv_out.max(), diff.min(), diff.max()]])
        print("relative-diff", (diff/conv_out).max().item(), (diff/conv_out).mean().item())
        print()
        