'''
Script to train the BRATs dataset
'''
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
from utils.losses import dice_loss_with_logits, dice_score_val, dice_loss_with_logits_batched
from utils.util import crop_collate_fn
import tensorboardX
import os
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True, help='Name of experiment')
parser.add_argument('--cfg_file', type=str, default='configs/brats_basic_seg.yaml', help='Path to config file')
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)

@torch.no_grad()
def eval_validation_data(cfg, network, val_dataset, epoch=None, writer=None):
    '''
    Check for validation performance here
    '''
    names = ['whole_tumor', 'tumor_core', 'enhancing_tumor']
    try:
        dice_scores = [[], [], []]
        gt_segms = []
        pred_segms = []
        for idx in tqdm(range(len(val_dataset))):
            if idx == 400:
                break
            datum = val_dataset[idx]
            embed = datum['embeddings'].cuda()
            coords = datum['xyz'].cuda() / (datum['dims'][None].cuda() - 1) * 2 - 1
            coords = coords[:, None]
            gt_segm = datum['segm'].cuda()
            logits = network(embed, coords)[:, 0]
            pred_segms.append(logits)
            gt_segms.append(gt_segm)
            # compute the dice
            if idx % 8 == 7:
                gt_segms = torch.cat(gt_segms, dim=0)  
                pred_segms = torch.cat(pred_segms, dim=0)
                pred_segms = torch.argmax(pred_segms, dim=-1) # [N]
                # get dice scores (for whole tumor)
                pred_wt = (pred_segms > 0)
                gt_wt = (gt_segms > 0)
                dice_scores[0].append(dice_score_val(pred_wt, gt_wt, num_classes=2, ignore_class=0)[0].item())
                # get dice scores (for tumor core)
                pred_wt = (pred_segms == 1)+(pred_segms == 3)
                gt_wt = (gt_segms == 1)+(gt_segms == 3)
                dice_scores[1].append(dice_score_val(pred_wt, gt_wt, num_classes=2, ignore_class=0)[0].item())
                # get dice scores (for enhancing tumor)
                pred_wt = (pred_segms == 3)
                gt_wt = (gt_segms == 3)
                dice_scores[2].append(dice_score_val(pred_wt, gt_wt, num_classes=2, ignore_class=0)[0].item())
                # dices = dice_score_val(pred_segms, gt_segms, num_classes=4, ignore_class=0)
                # for i, d in enumerate(dices):
                #     dice_scores[i].append(d.item())
                # reset
                pred_segms = []
                gt_segms = []

        print("validation results for epoch=", epoch)
        for i, name in enumerate(names):
            print("Dice {} mean={:04f}, std={:04f}".format(name, np.mean(dice_scores[i]), np.std(dice_scores[i])))
            valepoch = -1 if epoch is None else epoch
            writer.add_scalar(f'val/dice_mean_{name}', np.mean(dice_scores[i]), valepoch)

    except KeyboardInterrupt:
        print("Skipping validation")
    

if __name__ == '__main__':
    # parse args and get config
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    print(cfg)

    if os.path.exists('experiments/' + args.exp_name):
        print("Experiment {args.exp_name} already exists. Please delete it first.")
        exit(1)
    os.makedirs('experiments/' + args.exp_name)
    writer = tensorboardX.SummaryWriter(log_dir='experiments/' + args.exp_name)

    # set up datasets
    train_dataset = BRATS2021EncoderSegDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=True, val_fold=cfg.VAL.FOLD)
    val_dataset   = BRATS2021EncoderSegDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=False, val_fold=cfg.VAL.FOLD)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, 
                                  pin_memory=True)

    # compute offsets and resolutions
    dummy = GridEncoder(level_dim=4, desired_resolution=196, gridtype='tiled', align_corners=True, log2_hashmap_size=19)
    resolutions = dummy.resolutions.cuda()
    offsets = dummy.offsets.cuda()
    del dummy

    # init network and stuff
    network = init_network(cfg, offsets, resolutions).cuda()
    optim = get_optimizer(cfg, network)
    lr_scheduler = get_scheduler(cfg, optim)

    eval_validation_data(cfg, network, val_dataset, epoch=None, writer=writer)
    # train
    for epoch in range(cfg.TRAIN.EPOCHS):
        # perm = tqdm(np.random.permutation(len(train_dataset)))
        perm = tqdm(train_dataloader)
        for i, datum in enumerate(perm):
            optim.zero_grad()
            # get data
            # datum = train_dataset[idx]
            embed = datum['embeddings'].cuda() # [B, N, 1, C]
            embed = embed.squeeze(2).permute(1, 0, 2).contiguous()  # [N, B, C]
            # coords = [B, N, 3], dims = [B, 3]
            coords = datum['xyz'].cuda() / (datum['dims'][:, None].cuda() - 1) * 2 - 1
            coords = coords.permute(1, 0, 2).contiguous()  # [N, B, 3]
            # coords = coords[:, None]
            gt_segm = datum['segm'].cuda()   # [B, N]
            # weights = datum['weights'].cuda().mean(0)  # [B, C] -> [C]
            # forward
            logits = network(embed, coords)  # [N, B, out]
            logits = logits.permute(1, 0, 2) # [B, N, out]
            ce_loss = F.cross_entropy(logits.permute(0, 2, 1), gt_segm.long()) 
            dice_loss = dice_loss_with_logits_batched(logits, gt_segm)
            loss = cfg.SEG.WEIGHT_DICE * dice_loss + cfg.SEG.WEIGHT_CE * ce_loss
            loss.backward()
            optim.step()
            perm.set_description(f'Epoch:{epoch} Loss:{loss.item():.4f} CE:{ce_loss.item():.4f} Dice:{dice_loss.item():.4f}')
            writer.add_scalar('train/loss', loss.item(), epoch*len(train_dataset)+i)
            writer.add_scalar('train/ce_loss', ce_loss.item(), epoch*len(train_dataset)+i)
            writer.add_scalar('train/dice_loss', dice_loss.item(), epoch*len(train_dataset)+i)
        lr_scheduler.step()
        eval_validation_data(cfg, network, val_dataset, epoch, writer=writer)
