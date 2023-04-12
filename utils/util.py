import numpy as np
from networks.resnet import AbstractResNetBasic
import torch

def z_score_normalize(image):
    return (image - image.mean()) / image.std()

def uniform_normalize(image):
    return (image - image.min()) / (image.max() - image.min()) * 2 - 1

## network related utils
def init_network(cfg, offsets, resolutions):
    # TODO: Change this
    net = AbstractResNetBasic(offsets, resolutions)
    return net

def get_optimizer(cfg, net):
    # TODO: Change this
    optim = torch.optim.Adam(net.parameters(), lr=cfg.TRAIN.BASE_LR)
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
