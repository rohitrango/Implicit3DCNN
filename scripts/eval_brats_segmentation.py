from configs.config import get_cfg_defaults
import torch
from torch import nn
import argparse
from dataloaders import BRATS2021EncoderSegDataset
from utils import init_network, get_optimizer, get_scheduler
from gridencoder import GridEncoder
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from utils.losses import dice_loss_with_logits, dice_score_val, dice_loss_with_logits_batched, dice_score_binary
from utils.util import crop_collate_fn, format_raw_gt_to_brats
import tensorboardX
import os
from torch.utils.data import DataLoader
from os import path as osp
from scripts.train_brats_segmentation import eval_validation_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--tta', action='store_true', help="Test time augmentation")
    parser.add_argument('--tta_samples', type=int, default=10, help="Number of test time augmentation samples")
    parser.add_argument('--save_preds', action='store_true', help='Save predictions to disk')
    args = parser.parse_args()

    exp_path = osp.join('experiments', args.exp_name)
    cfg_path = osp.join(exp_path, 'config.yaml')

    if not os.path.exists(exp_path):
        raise ValueError(f'Experiment path {exp_path} does not exist')
        
    # read the config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    print(cfg)

    # get validation dataset
    val_dataset = BRATS2021EncoderSegDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=False, val_fold=cfg.VAL.FOLD)
    # compute offsets and resolutions
    dummy = GridEncoder(level_dim=cfg.ENCODE.LEVEL_DIM, desired_resolution=cfg.ENCODE.DESIRED_RESOLUTION, gridtype='tiled', align_corners=True, log2_hashmap_size=19)
    resolutions = dummy.resolutions.cuda()
    offsets = dummy.offsets.cuda()
    del dummy

    # init network and stuff
    network = init_network(cfg, offsets, resolutions).cuda()
    network.load_state_dict(torch.load(osp.join(exp_path, 'best_model.pth'))['network'], strict=True)

    metrics = dict()
    eval_validation_data(cfg, network, None, val_dataset, best_metrics=metrics, epoch=None, writer=None, stop_at=None, \
                         tta=args.tta, tta_samples=args.tta_samples, save_preds=args.save_preds)
