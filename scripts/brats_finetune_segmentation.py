
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

@torch.no_grad()
def eval_validation_data_finetune(encoder_net, unet_model, val_dataset, patch_indices, coords, \
                                      writer, epoch, best_metrics, cfg):
    patch_size = cfg.FINETUNE.PATCH_SIZE
    # evaluate the given model on validation set
    pbar = tqdm(val_dataset)
    dice_scores = [[], [], []]
    for datum in pbar:
        seg_coarse_logits = encoder_net(datum['embeddings'].cuda(), coords).reshape(240, 240, 155, -1).permute(3, 0, 1, 2)
        seg_coarse_logits = seg_coarse_logits[1:].contiguous()  # [3, H, W, D]
        # prepare other outputs
        seg_coarse_p = torch.sigmoid(seg_coarse_logits)
        seg_idx = datum['seg'].cuda()  # [H, W, D]
        seg_gt = torch.stack(format_raw_gt_to_brats(seg_idx), dim=0)  # [3, H, W, D]
        images = datum['images'].cuda()

        outputs_collated = torch.zeros(3, 240, 240, 155).cuda()
        count_collated = torch.zeros(1, 240, 240, 155).cuda()
        # get patches
        for (hs, ws, ds) in patch_indices:
            imagepatch = images[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size]
            coarse_patch = seg_coarse_p[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size]
            input_patch = torch.cat([imagepatch, coarse_patch], dim=0)[None] # [1, 7, H, W, D]
            # get output logits
            out_logit = unet_model(input_patch)
            if cfg.FINETUNE.ADD_RESIDUAL_END:
                out_logit = out_logit + seg_coarse_logits[None, :, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] # [1, 3, P, P, P]
            # out_logit = seg_coarse_logits[None, :, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size]
            # loss
            out_patch = torch.sigmoid(out_logit)
            outputs_collated[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] += out_patch[0]
            count_collated[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] += 1
        # get average
        outputs_collated /= count_collated
        # print(outputs_collated.min(), outputs_collated.max())
        # print(count_collated.min(), count_collated.max())
        outputs_collated = (outputs_collated >= 0.5).float()
        # get dice scores
        for i in range(3):
            dice_scores[i].append(dice_score_binary(outputs_collated[i], seg_gt[i]).item())
        pbar.set_description(", ".join([str(x[-1]) for x in dice_scores]))
    # print mean dice scores
    is_best = False
    #dice_scores = [torch.stack(x).mean().item() for x in dice_scores]
    dice_scores = [np.mean(x) for x in dice_scores]
    dice_score = np.mean(dice_scores)
    best_dice_so_far = best_metrics.get('dice', 0)
    print("Mean dice score for epoch {}: {}".format(epoch, dice_scores))
    if dice_score > best_dice_so_far:
        best_metrics['dice'] = np.mean(dice_scores)
        is_best = True
    writer.add_scalar('val/dice', dice_score, epoch)
    return is_best

def train_finetune_net(unet_model, encoder_net, train_dataset, val_dataset, cfg):
    '''
    unet_model: unet model to train
    encoder_net: (trained) encoder network to use
    train_dataset: training dataset
    val_dataset: validation dataset
    cfg: config from encoder (to double check anything)
    '''
    num_epochs = cfg.FINETUNE.NUM_EPOCHS
    patch_size = cfg.FINETUNE.PATCH_SIZE

    best_metrics = dict()
    # set up coordinates, optimizer, scheduler
    coords = torch.meshgrid(torch.linspace(-1, 1, 240), torch.linspace(-1, 1, 240), torch.linspace(-1, 1, 155), indexing='ij')
    coords = torch.stack(coords, dim=-1).cuda().reshape(-1, 3)[:, None].contiguous()  # [N, 1, 3]
    optim = torch.optim.Adam(unet_model.parameters(), lr=cfg.FINETUNE.LR, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs, 0.9)
    patch_indices = generate_patch_indices(240, 240, 155, patch_size)
    # train
    writer = tensorboardX.SummaryWriter(log_dir='experiments/' + args.exp_name + "/finetune")
    it = 0
    for epoch in range(num_epochs):
        pbar = tqdm(np.random.permutation(len(train_dataset)))
        try:
            for idx in pbar:
                datum = train_dataset[idx]
                # get the coarse segmentation
                with torch.no_grad():
                    seg_coarse_logits = encoder_net(datum['embeddings'].cuda(), coords).reshape(240, 240, 155, -1).permute(3, 0, 1, 2)
                    seg_coarse_logits = seg_coarse_logits[1:].contiguous()  # [3, H, W, D]
                    # prepare other outputs
                    seg_coarse_p = torch.sigmoid(seg_coarse_logits)
                    seg_idx = datum['seg'].cuda()  # [H, W, D]
                    seg_gt = torch.stack(format_raw_gt_to_brats(seg_idx), dim=0)  # [3, H, W, D]
                    images = datum['images'].cuda()
                    # perform some augmentation here?
                # get patches
                for (hs, ws, ds) in patch_indices:
                    optim.zero_grad()
                    imagepatch = images[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size]
                    coarse_patch = seg_coarse_p[:, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size]
                    input_patch = torch.cat([imagepatch, coarse_patch], dim=0)[None] # [1, 7, H, W, D]
                    # get seg_gt (which is binary) and seg_idx (which is int)
                    seg_gt_patch = seg_gt[None, :, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] # [1, 3, P, P, P]
                    seg_idx_patch = seg_idx[None, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] # [1, P, P, P]
                    # transform patch
                    dims = []
                    for i in range(3):
                        if np.random.rand() > 0.5:
                            dims.append(i+2)
                    if dims != []:
                        input_patch = torch.flip(input_patch, dims=dims)
                        seg_gt_patch = torch.flip(seg_gt_patch, dims=dims)
                        seg_idx_patch = torch.flip(seg_idx_patch, dims=[x-1 for x in dims])
                    # get output logits
                    out_logit = unet_model(input_patch)
                    if cfg.FINETUNE.ADD_RESIDUAL_END:
                        out_logit = out_logit + seg_coarse_logits[None, :, hs:hs+patch_size, ws:ws+patch_size, ds:ds+patch_size] # [1, 3, P, P, P]
                    # loss
                    focal_loss = focal_loss_with_logits(out_logit, seg_gt_patch, gamma=cfg.SEG.FOCAL_GAMMA)
                    dice_loss = dice_loss_with_logits_batched(out_logit.reshape(1, 3, -1).permute(0, 2, 1), seg_idx_patch.reshape(1, -1), 'sigmoid', ignore_idx=-1)
                    loss = dice_loss + 0
                    if cfg.SEG.WEIGHT_FOCAL > 0:
                        loss += cfg.SEG.WEIGHT_FOCAL * focal_loss
                    loss.backward()
                    optim.step()
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}, Dice: {1-dice_loss.item():.4f}, Focal: {focal_loss.item():.4f}")
                    writer.add_scalar('train/loss', loss.item(), it)
                    writer.add_scalar('train/dice', 1-dice_loss.item(), it)
                    writer.add_scalar('train/focal', focal_loss.item(), it)
                    it += 1
        except:
            print("Skipping finetuning for epoch {}".format(epoch))

        # scheduler step
        lr_scheduler.step()
        # validation
        is_best = eval_validation_data_finetune(encoder_net, unet_model, val_dataset, patch_indices=patch_indices, coords=coords, \
                                      writer=writer, epoch=epoch, best_metrics=best_metrics, cfg=cfg)
        if is_best: 
            # save model
            torch.save({
                'network': unet_model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_metrics': best_metrics,
            }, osp.join('experiments', args.exp_name, 'best_model_finetune.pth'))
            print("Saved best model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # get experiment config and model
    exp_path = osp.join('experiments', args.exp_name)
    cfg_path = osp.join(exp_path, 'config.yaml')

    if not os.path.exists(exp_path):
        raise ValueError(f'Experiment path {exp_path} does not exist')
    # read the config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(args.opts)
    print(cfg)

    with open(osp.join(exp_path, 'config_finetune.yaml'), 'w') as f:
        f.write(cfg.dump())

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
    # ignoring the `is_3d` flag for this network
    # initialize unet (4 image channels + 3 segmentation channels)
    unet_model = ResidualUNet3D(4+3, 3, num_levels=2, f_maps=32).cuda()
    print(unet_model)
    train_finetune_net(unet_model, encoder_net, train_dataset, val_dataset, cfg)
    
