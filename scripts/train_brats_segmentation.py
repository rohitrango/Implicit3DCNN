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
from utils.losses import dice_loss_with_logits, dice_score_val

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, default='configs/brats_basic_seg.yaml', help='Path to config file')
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)

@torch.no_grad()
def eval_validation_data(cfg, network, val_dataset, epoch=None):
    '''
    Check for validation performance here
    '''
    dice_scores = [[], [], []]
    gt_segms = []
    pred_segms = []
    for idx in tqdm(range(len(val_dataset))):
        if idx == 80:
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
            dices = dice_score_val(pred_segms, gt_segms, num_classes=4, ignore_class=0)
            for i, d in enumerate(dices):
                dice_scores[i].append(d.item())
            # reset
            pred_segms = []
            gt_segms = []

    print("validation results for epoch=", epoch)
    for i in range(3):
        print("Dice mean={:04f}, std={:04f}".format(np.mean(dice_scores[i]), np.std(dice_scores[i])))
    

if __name__ == '__main__':
    # parse args and get config
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    print(cfg)

    # set up datasets
    train_dataset = BRATS2021EncoderSegDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=True, val_fold=cfg.VAL.FOLD)
    val_dataset   = BRATS2021EncoderSegDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=False, val_fold=cfg.VAL.FOLD)

    # compute offsets and resolutions
    dummy = GridEncoder(level_dim=4, desired_resolution=196, gridtype='tiled', align_corners=True, log2_hashmap_size=19)
    resolutions = dummy.resolutions.cuda()
    offsets = dummy.offsets.cuda()
    del dummy

    # init network and stuff
    network = init_network(cfg, offsets, resolutions).cuda()
    optim = get_optimizer(cfg, network)
    lr_scheduler = get_scheduler(cfg, optim)

    # Sanity check first
    eval_validation_data(cfg, network, val_dataset)

    # train
    for epoch in range(cfg.TRAIN.EPOCHS):
        perm = tqdm(np.random.permutation(len(train_dataset)))
        for idx in perm:
            optim.zero_grad()
            # get data
            datum = train_dataset[idx]
            embed = datum['embeddings'].cuda()
            coords = datum['xyz'].cuda() / (datum['dims'][None].cuda() - 1) * 2 - 1
            coords = coords[:, None]
            gt_segm = datum['segm'].cuda()
            weights = datum['weights'].cuda()
            # forward
            logits = network(embed, coords)[:, 0]
            ce_loss = F.cross_entropy(logits, gt_segm.long(), weight=weights)
            dice_loss = dice_loss_with_logits(logits, gt_segm)
            loss = cfg.SEG.WEIGHT_DICE * dice_loss + cfg.SEG.WEIGHT_CE * ce_loss
            loss.backward()
            optim.step()
            perm.set_description(f'Epoch:{epoch} Loss:{loss.item():.4f} CE:{ce_loss.item():.4f} Dice:{dice_loss.item():.4f}')
        lr_scheduler.step()
        eval_validation_data(cfg, network, val_dataset, epoch)
