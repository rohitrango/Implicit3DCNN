import torch
from torch import nn
from networks.contextlayer import AbstractContextLayer
from networks.conv3d import AbstractConv3D, HashRouterLayer

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, resolutions, offsets, layers=2, context=True, affine_context=True, 
                 kernel_size=3,
                 num_levels=16, log_hashmap_size=19, activation=nn.LeakyReLU()):
        super().__init__()
        self.context = None
        self.activation = activation
        if in_channels != out_channels:
            self.context = AbstractContextLayer(in_channels, out_channels, resolutions=resolutions, offsets=offsets, \
                                                 affine=True, num_levels=num_levels, log_hashmap_size=log_hashmap_size)
        elif context:
            self.context = AbstractContextLayer(in_channels, out_channels, resolutions=resolutions, offsets=offsets, \
                                                affine=affine_context, num_levels=num_levels, log_hashmap_size=log_hashmap_size)
        # define convolutions
        convs = []
        for _ in range(layers):
            convs.append(AbstractConv3D(in_channels, out_channels, resolutions=resolutions, offsets=offsets,
                                        kernel_size=kernel_size, num_levels=num_levels, log_hashmap_size=log_hashmap_size))
            in_channels = out_channels
        self.convs = nn.ModuleList(convs)
    
    def forward(self, input):
        # input = [N, B, C]
        x = input+0
        for mod in self.convs:
            x = mod(x)
            if self.activation is not None:
                x = self.activation(x)
        if self.context is not None:
            x = x + self.context(input)
        return x


class AbstractResNetBasic(nn.Module):
    def __init__(self, offsets, resolutions):
        super().__init__()
        resblocks = []
        # resblocks.append(Resblock(4, 8, resolutions, offsets))
        # resblocks.append(Resblock(8, 16, resolutions, offsets))
        # resblocks.append(Resblock(16, 16, resolutions, offsets))
        # resblocks.append(Resblock(16, 16, resolutions, offsets))
        # resblocks.append(Resblock(16, 8, resolutions, offsets))
        # self.resblocks = nn.ModuleList(resblocks)
        # # decoder
        # self.decoder = HashRouterLayer(resolutions, offsets, num_levels=16, log_hashmap_size=19,
        #                                   embed_channels=8, mlp_channels=[32, 32], out_channels=4)
        resblocks.append(Resblock(4, 8, resolutions, offsets))
        resblocks.append(Resblock(8, 8, resolutions, offsets))
        resblocks.append(Resblock(8, 8, resolutions, offsets))
        resblocks.append(Resblock(8, 8, resolutions, offsets))
        self.resblocks = nn.ModuleList(resblocks)
        self.decoder = HashRouterLayer(resolutions, offsets, num_levels=16, log_hashmap_size=19,
                                            embed_channels=8, mlp_channels=[], out_channels=4)

        self.resolutions = resolutions
        self.offsets = offsets
    
    def forward(self, embedding, x):
        # embedding = [M, B, C]
        # x = [N, B, C]
        for resblock in self.resblocks:
            embedding = resblock(embedding)
        y = self.decoder(x, embedding)
        return y 
    