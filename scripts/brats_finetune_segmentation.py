
from configs.config import get_cfg_defaults
import torch
from torch import nn
import argparse
from dataloaders import BRATSFinetuneDataset
from utils import init_network, get_optimizer, get_scheduler
from gridencoder import GridEncoder
from tqdm.auto import tqdm
import numpy as np
from torch.nn import functional as F
from utils.losses import dice_loss_with_logits, dice_score_val, dice_loss_with_logits_batched, dice_score_binary, focal_loss_with_logits
from utils.util import crop_collate_fn, format_raw_gt_to_brats
import tensorboardX
import os
from torch.utils.data import DataLoader
from os import path as osp
from scripts.train_brats_segmentation import eval_validation_data
from networks.unet_refine import ResidualUNet3D

def generate_patch_indices(H, W, D, patch_size):
    """
    Generate starting indices for patches of size patch_size x patch_size x patch_size
    that cover the entire volume of size H x W x D.
    """
    indices = []
    for h in range(0, H, patch_size):
        for w in range(0, W, patch_size):
            for d in range(0, D, patch_size):
                h_end = h + patch_size if h + patch_size <= H else H
                w_end = w + patch_size if w + patch_size <= W else W
                d_end = d + patch_size if d + patch_size <= D else D
                indices.append((h_end-patch_size, w_end-patch_size, d_end-patch_size))
    
    return indices

def train_finetune_net(unet_model, encoder_net, train_dataset, val_dataset, num_epochs, patch_size, cfg, lr):
    '''
    unet_model: unet model to train
    encoder_net: (trained) encoder network to use
    train_dataset: training dataset
    val_dataset: validation dataset
    num_epochs: number of epochs to train
    patch_size: patch size to train on
    cfg: config from encoder (to double check anything)
    lr: learning rate
    '''
    coords = torch.meshgrid(torch.linspace(-1, 1, 240), torch.linspace(-1, 1, 240), torch.linspace(-1, 1, 155), indexing='ij')
    coords = torch.stack(coords, dim=-1).cuda().reshape(-1, 3)[:, None].contiguous()  # [N, 1, 3]
    optim = torch.optim.Adam(unet_model.parameters(), lr=lr)
    patch_indices = generate_patch_indices(240, 240, 155, patch_size)
    # train
    writer = tensorboardX.SummaryWriter(log_dir='experiments/' + args.exp_name + "/finetune")
    for epoch in range(num_epochs):
        pbar = tqdm(np.random.permutation(len(train_dataset)))
        for idx in pbar:
            datum = train_dataset[idx]
            # get the coarse segmentation
            with torch.no_grad():
                seg_coarse_logits = encoder_net(datum['embeddings'].cuda(), coords).reshape(240, 240, 155, -1).permute(3, 0, 1, 2)
                seg_coarse_logits = seg_coarse_logits[1:].contiguous()  # [3, 240, 240, 155]
                # prepare other outputs
                seg_coarse_p = torch.sigmoid(seg_coarse_logits)
                seg_idx = datum['seg'].cuda()  # [H, W, D]
                seg_gt = torch.stack(format_raw_gt_to_brats(seg_idx), dim=0)
                images = datum['images'].cuda()
                # perform some augmentation here?

            # get patches
            for (hs, ws, ds) in patch_indices:
                optim.zero_grad()
                imagepatch = images[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size]
                coarse_patch = seg_coarse_p[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size]
                input_patch = torch.cat([imagepatch, coarse_patch], dim=0)[None] # [1, 7, 7, 7]
                out_logit = unet_model(input_patch) + 0.1*seg_coarse_logits[None, :, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] # [1, 3, P, P, P]
                # loss 
                seg_gt_patch = seg_gt[None, :, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] # [1, 3, P, P, P]
                seg_idx_patch = seg_idx[None, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] # [1, P, P, P]
                focal_loss = focal_loss_with_logits(out_logit, seg_gt_patch,  gamma=cfg.SEG.FOCAL_GAMMA)
                dice_loss = dice_loss_with_logits_batched(out_logit.reshape(1, 3, -1).permute(0, 2, 1), seg_idx_patch.reshape(1, -1), 'sigmoid', ignore_idx=0)
                loss = dice_loss + cfg.SEG.WEIGHT_FOCAL * focal_loss
                loss.backward()
                optim.step()
                pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}, Dice: {1-dice_loss.item():.4f}, Focal: {focal_loss.item():.4f}")
                writer.add_scalar('train/loss', loss.item(), epoch)
                writer.add_scalar('train/dice', 1-dice_loss.item(), epoch)
                writer.add_scalar('train/focal', focal_loss.item(), epoch)

            # perform some augmentation on the image? 

        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()

    # get experiment config and model
    exp_path = osp.join('experiments', args.exp_name)
    cfg_path = osp.join(exp_path, 'config.yaml')

    if not os.path.exists(exp_path):
        raise ValueError(f'Experiment path {exp_path} does not exist')
    # read the config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    print(cfg)

    # load network
    dummy = GridEncoder(level_dim=cfg.ENCODE.LEVEL_DIM, desired_resolution=cfg.ENCODE.DESIRED_RESOLUTION, gridtype='tiled', align_corners=True, log2_hashmap_size=19)
    resolutions = dummy.resolutions.cuda()
    offsets = dummy.offsets.cuda()
    del dummy
    # init network and stuff
    encoder_net = init_network(cfg, offsets, resolutions).cuda()
    encoder_net.load_state_dict(torch.load(osp.join(exp_path, 'best_model.pth'))['network'], strict=True)
    encoder_net.eval()

    # get new training and validation dataset
    # train_dataset = BRATSFinetuneDataset()
    shuffle_seed = cfg.VAL.RANDOM_SHUFFLE_SEED if cfg.VAL.RANDOM_SHUFFLE else None 
    train_dataset = BRATSFinetuneDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=True, \
                                               val_fold=cfg.VAL.FOLD, num_folds=cfg.VAL.MAX_FOLDS, \
                                                scale_range=cfg.DATASET.SCALE_RANGE, shuffle_seed=shuffle_seed)
    # validation dataset
    val_dataset   = BRATSFinetuneDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=False, \
                                               val_fold=cfg.VAL.FOLD, scale_range=cfg.DATASET.SCALE_RANGE, \
                                                num_folds=cfg.VAL.MAX_FOLDS, shuffle_seed=shuffle_seed)
    # initialize unet (4 image channels + 3 segmentation channels)
    unet_model = ResidualUNet3D(4+3, 3, num_groups=2, num_levels=2, f_maps=32).cuda()
    train_finetune_net(unet_model, encoder_net, train_dataset, val_dataset, args.num_epochs, args.patch_size, cfg, args.learning_rate)
    