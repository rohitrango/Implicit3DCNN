#!/usr/bin/env python
import torch
from torch import nn
from time import time, sleep
from conv3d import AbstractConv3D
from contextlayer import abstractContextFunction
from torch.nn import functional as F
import gridencoder as ge
import numpy as np
import matplotlib.pyplot as plt

# given a batch size, generate a table
batch_size = 1

for input_channels in [2, 4, 8, 16, 32]:
    # Get inputs
    encoder = ge.GridEncoder(level_dim=2, desired_resolution=256, gridtype='tiled', align_corners=True, log2_hashmap_size=19).cuda()
    embed = encoder.embeddings[:, None] * 1e3 # [N, 1, 2]
    embed = embed.repeat(1, batch_size, input_channels//2).contiguous().detach()
    offsets, resolutions = encoder.offsets, encoder.resolutions
    # now iterate over outputs
    for output_channels in [2, 4, 8, 16, 32, 64]:
        conv = AbstractConv3D(input_channels, output_channels, resolutions, offsets, 3, bias=True, num_levels=16, log_hashmap_size=19).cuda()
        input_ = embed.clone().requires_grad_(True)
        # run forward
        a = time()
        out = conv(input_)
        fwd_time = time() - a
        # run backward
        loss = (out**2).sum()
        a = time()
        loss.backward()
        bwd_time = time() - a
        print("batch_size: {}, input channels: {}, output channels: {}, fwd time: {:.6f}, bwd time: {:.6f}".format(batch_size, input_channels, output_channels, fwd_time, bwd_time))
