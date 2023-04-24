import numpy as np
from networks.resnet import AbstractContextResNet, AbstractResNet, ConvBlocks
import torch
from torch import nn

def z_score_normalize(image):
    return (image - image.mean()) / image.std()

def uniform_normalize(image):
    return (image - image.min()) / (image.max() - image.min()) * 2 - 1

def get_activation_fn(cfg):
    # given network activation, get the apt activation function
    name = cfg.ACTIVATION
    if name == 'LeakyReLU':
        return nn.LeakyReLU(cfg.ACTIVATION_PARAM)
    elif name == 'ReLU':
        return nn.ReLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")

## network related utils
def init_network(cfg, offsets, resolutions):
    # Check the network parameters
    cfgnet = cfg.NETWORK
    name = cfgnet.NAME
    actv = get_activation_fn(cfgnet)
    # get input and output channels
    input_channels = cfgnet.INPUT_CHANNELS
    output_channels = cfgnet.OUTPUT_CHANNELS

    if name == 'AbstractContextResNet':
        net = AbstractContextResNet(input_channels, output_channels, offsets, resolutions, blocks=cfgnet.BLOCK_CHANNELS, 
                                    num_layers_per_block=cfgnet.BLOCK_NUM_LAYERS, activation=actv)
    elif name == 'AbstractResNet':
        net = AbstractResNet(input_channels, output_channels, offsets, resolutions, cfgnet.BLOCK_CHANNELS,
                             num_layers_per_block=cfgnet.BLOCK_NUM_LAYERS, activation=actv)
    elif name == "ConvBlocks":
        net = ConvBlocks(input_channels, output_channels, offsets, resolutions, cfgnet.BLOCK_CHANNELS,
                         activation=actv)
    else:
        raise ValueError(f"Unknown network type: {name}")
    print(net)
    return net

def get_optimizer(cfg, net):
    # TODO: Change this
    optim = torch.optim.Adam(net.parameters(), lr=cfg.TRAIN.BASE_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optim

def get_scheduler(cfg, optim):
    # TODO: Change this
    sch = torch.optim.lr_scheduler.PolynomialLR(optim, total_iters=cfg.TRAIN.EPOCHS, power=0.9)
    return sch

def crop_collate_fn(data):
    # data is a list of dicts, crop the coordinates and segmentations
    sizes = [d['xyz'].shape[0] for d in data]
    minsize = min(sizes)
    for d in data:
        d['xyz'] = d['xyz'][:minsize]
        d['segm'] = d['segm'][:minsize]
    ret = dict()
    for k in data[0].keys():
        ret[k] = torch.stack([d[k] for d in data])
    return ret

def format_raw_gt_to_brats(segm):
    # convert the raw segmentation to brats format
    # enhancing tumor, tumor core, whole tumor
    et = (segm == 3).float()
    tc = et + (segm == 1)
    wt = tc + (segm == 2)
    return et, tc, wt