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
from tqdm.auto import tqdm
import numpy as np
from torch.nn import functional as F
from utils.losses import dice_loss_with_logits, dice_score_val, dice_loss_with_logits_batched, dice_score_binary
from utils.util import crop_collate_fn, format_raw_gt_to_brats
import tensorboardX
import os
from torch.utils.data import DataLoader
from os import path as osp

@torch.no_grad()
def eval_validation_data(cfg, network, optim, val_dataset, best_metrics=None, epoch=None, writer=None, stop_at=None):
    '''
    Check for validation performance here
    '''
    names = ['whole_tumor', 'tumor_core', 'enhancing_tumor']
    try:
        dice_scores = [[], [], []]
        gt_segms = []
        pred_segms = []
        for idx in tqdm(range(len(val_dataset))):
            if stop_at is not None and idx >= stop_at:
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
                # get dice scores (for whole tumor)
                if cfg.TRAIN.BRATS_SEGM_MODE == 'raw':
                    pred_segms = torch.argmax(pred_segms, dim=-1) # [N]
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
                else:
                    # print(pred_segms.shape, gt_segms.shape)
                    pred_segms = (torch.sigmoid(pred_segms)>=0.5).float()
                    gts_brats = format_raw_gt_to_brats(gt_segms)
                    for i in range(3):
                        _d = dice_score_binary(pred_segms[..., i+1], gts_brats[i]).item()
                        # print(i, _d)
                        dice_scores[2-i].append(_d)
                    # dice_scores[2].append(dice_score_binary(pred_segms[..., 1], gts_brats[0]).item())  # ET
                    # dice_scores[1].append(dice_score_binary(pred_segms[..., 2], gts_brats[1]).item())  # TC
                    # dice_scores[0].append(dice_score_binary(pred_segms[..., 3], gts_brats[2]).item())  # WT
                # reset
                pred_segms = []
                gt_segms = []

        print("validation results for epoch=", epoch)
        for i, name in enumerate(names):
            print("Dice {} mean={:04f}, std={:04f}".format(name, np.mean(dice_scores[i]), np.std(dice_scores[i])))
            valepoch = -1 if epoch is None else epoch
            if writer is not None:
                writer.add_scalar(f'val/dice_mean_{name}', np.mean(dice_scores[i]), valepoch)

        # Check if these are the best metrics
        if epoch is not None:
            this_dice = np.mean([np.mean(dice_scores[i]) for i in range(len(names))])
            # see if the mean dice is better than what we have!
            if this_dice > np.mean([best_metrics.get(n, -np.inf) for n in names]):
                for i, n in enumerate(names):
                    best_metrics[n] = np.mean(dice_scores[i])
                # save model
                torch.save({'epoch': epoch,
                            'network': network.state_dict(),  
                            'optim': optim.state_dict(),
                            'metrics': best_metrics
                            }, osp.join(cfg.EXP_NAME, "best_model.pth"))
                print("Saved best model.\n")

    # check for a keyboard interrupt to skip
    except KeyboardInterrupt:
        print("Skipping validation")
    
    return best_metrics
    
if __name__ == '__main__':
    # Load parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help='Name of experiment')
    parser.add_argument('--cfg_file', type=str, default='configs/brats_basic_seg.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    # parse args and get config
    args = parser.parse_args()
    cfg = get_cfg_defaults()

    if args.resume:
        # Just take the config from the experiment folder
        exp_dir = osp.join('experiments/', args.exp_name)
        args.cfg_file = osp.join(exp_dir, 'config.yaml')
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)
        print(cfg)
        cfg.EXP_NAME = exp_dir
        print("Resuming from checkpoint...")

    else:
        # Create a new experiment folder
        cfg.merge_from_file(args.cfg_file)
        cfg.merge_from_list(args.opts)
        print(cfg)
        cfg.EXP_NAME = osp.join('experiments/', args.exp_name)
        if os.path.exists('experiments/' + args.exp_name):
            print(f"Experiment {args.exp_name} already exists. Please delete it first.")
            exit(1)
        os.makedirs('experiments/' + args.exp_name)
        # save the config
        with open(osp.join('experiments/', args.exp_name, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())
        print("Starting new experiment...")

    writer = tensorboardX.SummaryWriter(log_dir='experiments/' + args.exp_name)

    # set up datasets
    train_dataset = BRATS2021EncoderSegDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=True, val_fold=cfg.VAL.FOLD)
    val_dataset   = BRATS2021EncoderSegDataset(cfg.DATASET.TRAIN_ENCODED_DIR, cfg.DATASET.TRAIN_SEG_DIR, train=False, val_fold=cfg.VAL.FOLD)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, 
                                  pin_memory=True)

    # compute offsets and resolutions
    dummy = GridEncoder(level_dim=cfg.ENCODE.LEVEL_DIM, desired_resolution=cfg.ENCODE.DESIRED_RESOLUTION, gridtype='tiled', align_corners=True, log2_hashmap_size=19)
    resolutions = dummy.resolutions.cuda()
    offsets = dummy.offsets.cuda()
    del dummy

    # init network and stuff
    network = init_network(cfg, offsets, resolutions).cuda()
    optim = get_optimizer(cfg, network)

    # load checkpoint if needed
    start_epoch = 0
    # keep track of best metrics
    best_metrics = dict()
    if args.resume:
        saved = torch.load(osp.join(cfg.EXP_NAME, 'best_model.pth'))
        network.load_state_dict(saved['network'])
        optim.load_state_dict(saved['optim'])
        start_epoch = saved['epoch'] + 1
        best_metrics = saved['metrics']
        print(f"Resuming from epoch {start_epoch}.")
    
    # Load scheduler
    lr_scheduler = get_scheduler(cfg, optim, start_epoch)
    
    # extra loss config here
    use_ce_loss = cfg.SEG.WEIGHT_CE > 0
    # Eval in the beginning once
    # eval_validation_data(cfg, network, val_dataset, best_metrics=None, epoch=None, writer=writer, stop_at=400)
    # train
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # perm = tqdm(np.random.permutation(len(train_dataset)))
        perm = tqdm(train_dataloader)
        lr = lr_scheduler.get_last_lr()[0]
        try:
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
                # check for what mode are we training in
                ce_loss = torch.tensor(0, device=logits.device)
                if cfg.TRAIN.BRATS_SEGM_MODE == 'raw':
                    # check if we should use CE loss
                    if use_ce_loss:
                        ce_loss = F.cross_entropy(logits.permute(0, 2, 1), gt_segm.long()) 
                    dice_loss = dice_loss_with_logits_batched(logits, gt_segm, cfg.TRAIN.LOGIT_TRANSFORM, ignore_idx=0)
                else:
                    gt_segm_bratsformat = format_raw_gt_to_brats(gt_segm)
                    # check if we should use CE loss
                    if use_ce_loss:
                        ce_losses = [F.binary_cross_entropy_with_logits(logits[..., i+1], gt_segm_bratsformat[i]) for i in range(3)]
                        ce_loss = 0
                        for ce in ce_losses:
                            ce_loss += ce
                        ce_loss /= 3
                    dice_loss = dice_loss_with_logits_batched(logits, gt_segm_bratsformat, 'sigmoid', ignore_idx=0)

                loss = cfg.SEG.WEIGHT_DICE * dice_loss + cfg.SEG.WEIGHT_CE * ce_loss
                loss.backward()
                optim.step()
                perm.set_description(f'Epoch:{epoch} lr: {lr:.4f} Loss:{loss.item():.4f} CE:{ce_loss.item():.4f} Dice:{dice_loss.item():.4f}')
                writer.add_scalar('train/loss', loss.item(), epoch*len(train_dataloader)+i)
                writer.add_scalar('train/ce_loss', ce_loss.item(), epoch*len(train_dataloader)+i)
                writer.add_scalar('train/dice_loss', dice_loss.item(), epoch*len(train_dataloader)+i)
        except KeyboardInterrupt:
            print(f"Skipping rest of training for epoch {epoch}.")
        lr_scheduler.step()
        # val and save 
        if (epoch+1) % cfg.VAL.EVAL_EVERY == 0 or epoch == cfg.TRAIN.EPOCHS-1:
            best_metrics = eval_validation_data(cfg, network, optim, val_dataset, best_metrics, epoch, writer=writer, stop_at=cfg.VAL.STOP_AT)
