#!/usr/bin/env python
import torch
from torch import nn
import time
from conv3d import AbstractConv3D, abstractContextFunction
from torch.nn import functional as F
import gridencoder as ge
import numpy as np
import matplotlib.pyplot as plt

def profiler_check():
    ''' Check forward pass '''
    L = 19
    batch = 1
    # randomly initialize a grid encoder
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[:, None] * 1e3
    embed = embed.expand(-1, batch, -1).contiguous().detach()   # [N, 1, 2]
    embed.requires_grad = True
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    # print(resolutions)
    # print(offsets)
    # define layer with zero bias
    layer = AbstractConv3D(2, 32, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    print(embed.shape)
    a = time.time()
    output = layer(embed)  # [B, N, out]
    print(time.time() - a)
    (output**2).sum().backward()

def forward_pass_check():
    ''' Check forward pass '''
    L = 19
    batch = 1
    # randomly initialize a grid encoder
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[:, None] * 1e3 # [N, 1, 2]
    embed = embed.expand(-1, batch, -1).contiguous()   # [N, B, 2]
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    # print(resolutions)
    # print(offsets)
    
    # define layer with zero bias
    layer = AbstractConv3D(2, 32, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    print(embed.shape)
    a = time.time()
    output = layer(embed)  # [B, N, out]
    print(time.time() - a)

    output = output.permute(1, 0, 2).contiguous()
    embed = embed.permute(1, 0, 2).contiguous()

    # compare with a regular convolution
    for i in range(encoder.num_levels):
        r = resolutions[i] 
        if(r**3 > encoder.max_params):
            break
        # embed level
        embed_lvl = embed[:, offsets[i]:offsets[i]+r**3]
        _, S, _ = embed_lvl.shape
        if S != r**3:
            break
        # reshape this
        embed_lvl = embed_lvl.reshape(batch, r, r, r, 2).permute(0, 4, 3, 2, 1).contiguous()  # [1, 2, r, r, r]
        conv_lvl = layer.weight[i].permute(4, 3, 0, 1, 2)
        # get conv output
        conv_out = F.conv3d(embed_lvl, conv_lvl, bias=layer.bias[i], stride=1, padding=1).permute(0, 4, 3, 2, 1).contiguous()  # [1, r, r, r, 4]
        # conv_out = conv_out[:, 1:-1, 1:-1, 1:-1, :].contiguous()
        # compare with output 
        out = output[:, offsets[i]:offsets[i]+r**3].reshape(batch, r, r, r, -1).contiguous()
        # out = out[:, 1:-1, 1:-1, 1:-1, :].contiguous()
        print(conv_out.shape, out.shape, embed_lvl.shape)
        print("abs-diff ", (conv_out - out).abs().mean().item(), "conv abs mean value ", conv_out.abs().mean().item())
        # print(conv_out.abs().max())
        diff = (conv_out - out).abs()
        conv_out = torch.clamp(conv_out.abs(), 1e-1)
        # print([x.item() for x in [conv_out.min(), conv_out.max(), diff.min(), diff.max()]])
        print("relative-diff", (diff/conv_out).max().item(), (diff/conv_out).mean().item())
        print()
        
def backward_pass_check():
    ''' check if the backward pass for weights and biases is correct '''
    L = 19
    batch = 1
    inp_lvl = 2
    out_lvl = 4
    # randomly initialize a grid encoder
    encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L, level_dim=inp_lvl).cuda()
    embed = encoder.embeddings[:, None] * 1e4
    embed = embed.expand(-1, batch, -1).contiguous().detach()
    embed.requires_grad = True
    ## store resolutions and offsets
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    
    # define layer with zero bias
    layer = AbstractConv3D(inp_lvl, out_lvl, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=L).cuda()
    out = layer(embed)
    (out**2).sum().backward() 

    # conv layer
    our_w_grad = layer.weight.grad.data
    our_b_grad = layer.bias.grad.data
    embed_grad = embed.grad.data
    # reset grad
    # layer.weight.grad = None
    # layer.bias.grad = None

    # compare with a regular convolution
    for i in range(encoder.num_levels):
        r = resolutions[i] 
        if r**3 > encoder.max_params:
            break
        # get level
        embed_lvl = embed[offsets[i]:offsets[i]+r**3, :].permute(1, 0, 2).contiguous() # [B, r^3, 2]
        _, S, _ = embed_lvl.shape
        if S != r**3:
            break
        # reshape this
        embed_lvl = embed_lvl.reshape(batch, r, r, r, inp_lvl).permute(0, 4, 3, 2, 1).contiguous()  # [1, 2, r, r, r]
        embed_lvl = embed_lvl.data
        embed_lvl.requires_grad = True
        # get conv
        conv_lvl = layer.weight[i].permute(4, 3, 0, 1, 2).data.clone()   # [cout, cin, k, k, k]
        conv_lvl.requires_grad_(True)
        conv_bias = layer.bias[i].data.clone()
        conv_bias.requires_grad_(True)
        # get conv output
        conv_out = F.conv3d(embed_lvl, conv_lvl, bias=conv_bias, stride=1, padding=1).permute(0, 4, 3, 2, 1).contiguous()  # [1, r, r, r, 4]
        (conv_out**2).sum().backward()

        with torch.no_grad():
            diffwt = conv_lvl.grad.permute(2, 3, 4, 1, 0).contiguous() - our_w_grad[i]
            diffbias = conv_bias.grad - our_b_grad[i]
            embedlvlgradflat = embed_lvl.grad.permute(4, 3, 2, 0, 1).contiguous().reshape(-1, batch, inp_lvl)
            diffembed = embed_grad[offsets[i]:offsets[i]+r**3, :] - embedlvlgradflat
            print(embedlvlgradflat.shape, embed_grad[offsets[i]:offsets[i]+r**3, :].shape)
            print("diff wt grad: {:04f}, diff bias grad: {:04f}, diff embed grad: {:04f}\nabs wt grad: {:04f}, abs bias grad: {:04f}, abs embed grad: {:04f}".format(\
                diffwt.abs().mean().item(), diffbias.abs().mean().item(), diffembed.abs().mean().item(),
                conv_lvl.grad.abs().mean().item(), conv_bias.grad.abs().mean().item(), embed_lvl.grad.abs().mean().item()))
            print()

def forward_context_check():
    ''' Check forward pass for context layer '''
    L = 19
    batch = 1
    # randomly initialize a grid encoder
    encoder = ge.GridEncoder(per_level_scale=2, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    # encoder = ge.GridEncoder(desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=L).cuda()
    embed = encoder.embeddings[:, None] * 1e3 # [N, 1, 2]
    embed = embed.expand(-1, batch, -1).contiguous().detach()   # [N, B, 2]
    resolutions = encoder.resolutions
    offsets = encoder.offsets
    # define layer with zero bias
    print(embed.shape)
    embed.requires_grad = True
    a = time.time()
    output = abstractContextFunction(embed, offsets, resolutions, 16, 2**L)
    print(time.time() - a)
    (output**2).sum().backward()
    pred_grad = embed.grad.data  # store it

    ## compare it with interpolation
    for lvl in range(1, 16):
        r = resolutions[lvl] 
        if r**3 > encoder.max_params:
            break
        # get level from previous level
        rprev = resolutions[lvl-1]
        embed_lvl_inp = embed[offsets[lvl-1]:offsets[lvl-1]+rprev**3, :].permute(1, 0, 2).contiguous().detach()   # [B, rprev^3, C]
        embed_lvl_inp.requires_grad = True
        # convert it and pass it
        embed_lvl = embed_lvl_inp.reshape(batch, rprev, rprev, rprev, 2).permute(0, 4, 3, 2, 1).contiguous()  # [1, 2, r, r, r]
        embed_lvl = F.interpolate(embed_lvl, size=r, mode='nearest').permute(0, 4, 3, 2, 1).reshape(batch, r**3, 2).permute(1, 0, 2).contiguous()
        # get output level (which is interpolated version of previous level)
        output_lvl = output[offsets[lvl]:offsets[lvl]+r**3]
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(embed_lvl[0, 0, 0, :, :].detach().cpu().numpy(), cmap='jet')
        # axs[1].imshow(output_lvl[0, 0, 0, :, :].detach().cpu().numpy(), cmap='jet')
        # fig.savefig("{lvl}.png".format(lvl=lvl))
        # fig.clf()
        meandiff = (output_lvl - embed_lvl).abs().mean().item()
        # print(output_lvl.shape, embed_lvl.shape)
        print("error: {} avg output value: {}".format(meandiff, output_lvl.abs().mean().item()))
        # take gradients
        (embed_lvl**2).sum().backward()
        grad_lvl = embed_lvl_inp.grad.data.permute(1, 0, 2)
        pred_grad_lvl = pred_grad[offsets[lvl-1]:offsets[lvl-1]+rprev**3]
        # print(grad_lvl.shape, pred_grad_lvl.shape)
        meangraddiff = (grad_lvl - pred_grad_lvl).abs().mean().item()
        print("grad error: {}, mean grad value: {}".format(meangraddiff, grad_lvl.abs().mean().item()))
        print()
        




if __name__ == '__main__':
    print("Profiler check")
    profiler_check()
    # print("Forward pass check")
    # forward_pass_check()
    # print("\n\n\n")
    # print("Backward pass check")
    # backward_pass_check()
    # print("Forward context check.")
    # forward_context_check()