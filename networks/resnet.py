import torch
from torch import nn
from networks.contextlayer import AbstractContextLayer
from networks.conv3d import AbstractConv3D, HashRouterLayer
from networks.layernorm import LayerNorm

class Resblock(nn.Module):
    '''
    General residual block containing convolutions+non-linearity followed by residual block 
    the residual block can either be a simple per-grid layer, or a context layer which interpolates
    features from the previous grid
    '''
    def __init__(self, in_channels, out_channels, resolutions, offsets, layers=2, context=True, affine_context=True, 
                 kernel_size=3,
                 num_levels=16, log_hashmap_size=19, activation=nn.LeakyReLU(negative_slope=0.1),
                 layernorm=False):
        super().__init__()
        self.context = None
        self.activation = activation
        self.layernorm = layernorm
        if context:
            self.context = AbstractContextLayer(in_channels, out_channels, resolutions=resolutions, offsets=offsets, \
                                    affine=(in_channels!=out_channels or affine_context), num_levels=num_levels, log_hashmap_size=log_hashmap_size)
        else:
            self.context = nn.Linear(in_channels, out_channels)
            nn.init.kaiming_uniform_(self.context.weight)
            nn.init.zeros_(self.context.bias)
        # define convolutions
        convs = []
        lns = []
        for _ in range(layers):
            convs.append(AbstractConv3D(in_channels, out_channels, resolutions=resolutions, offsets=offsets,
                                        kernel_size=kernel_size, num_levels=num_levels, log_hashmap_size=log_hashmap_size))
            if layernorm:
                lns.append(LayerNorm(out_channels, resolutions, offsets))
            in_channels = out_channels
        self.convs = nn.ModuleList(convs)
        if layernorm:
            self.lns = nn.ModuleList(lns)
    
    def forward(self, input):
        # input = [N, B, C]
        x = input+0
        for i, mod in enumerate(self.convs):
            x = mod(x)
            if self.activation is not None:
                x = self.activation(x)
            if self.layernorm:
                x = self.lns[i](x)
        if self.context is not None:
            x = x + self.context(input)
        return x


class AbstractGeneralResNet(nn.Module):
    ''' General class for using Abstract ResNet with Context module '''
    def __init__(self, input_channels, output_channels, offsets, resolutions, blocks, num_layers_per_block, activation, context=True, layernorm=False):
        super().__init__()
        resblocks = []
        for out_channels, n_layers in zip(blocks, num_layers_per_block):
            resblocks.append(Resblock(input_channels, out_channels, resolutions, offsets, n_layers, activation=activation, context=context, layernorm=layernorm))
            resblocks.append(activation)
            input_channels = out_channels
        # add a final MLP layer to make channel size to if does not exist (because GridEncoder doesnt support higher grid sizes)
        if input_channels > 8:
            resblocks.append(nn.Linear(input_channels, 8))
            resblocks.append(activation)
            input_channels = 8

        self.resblocks = nn.ModuleList(resblocks)
        # TODO: Make decoder parameters more flexible
        self.decoder = HashRouterLayer(resolutions, offsets, num_levels=16, log_hashmap_size=19,
                                    embed_channels=input_channels, mlp_channels=[], out_channels=output_channels)

    def forward(self, embedding, x):
        # embedding = [M, B, C]
        # x = [N, B, C]
        for resblock in self.resblocks:
            embedding = resblock(embedding)
        y = self.decoder(x, embedding)
        return y 

def AbstractContextResNet(input_channels, output_channels, offsets, resolutions, blocks, num_layers_per_block, activation, layernorm):
    ''' a function that calls the general resnet with context variable set to True '''
    return AbstractGeneralResNet(input_channels, output_channels, offsets, resolutions, blocks, num_layers_per_block, activation, context=True, layernorm=layernorm)

def AbstractResNet(input_channels, output_channels, offsets, resolutions, blocks, num_layers_per_block, activation, layernorm):
    ''' just a function that calls the general resnet with context variable set to False '''
    return AbstractGeneralResNet(input_channels, output_channels, offsets, resolutions, blocks, num_layers_per_block, activation, context=False, layernorm=layernorm)


class ConvBlocks(nn.Module):
    '''
    Just a concatenation of conv blocks followed by non-linear activation
    '''
    def __init__(self, input_channels, output_channels, offsets, resolutions, blocks, activation):
        super().__init__()
        convblocks = []
        for out_channels in blocks:
            convblocks.append(AbstractConv3D(input_channels, out_channels, resolutions, offsets))
            convblocks.append(activation)
            input_channels = out_channels
        if input_channels > 8:
            convblocks.append(nn.Linear(input_channels, 8))
            convblocks.append(activation)
            input_channels = 8
            
        self.convs = nn.Sequential(convblocks)
        ## decoder
        self.decoder = HashRouterLayer(resolutions, offsets, num_levels=16, log_hashmap_size=19,
                                    embed_channels=input_channels, mlp_channels=[], out_channels=output_channels)
    
    def forward(self, embedding, x):
        # embedding = [M, B, C]
        # x = [N, B, C]
        embedding = self.convs(embedding)
        y = self.decoder(x, embedding)
        return y 
