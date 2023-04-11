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
