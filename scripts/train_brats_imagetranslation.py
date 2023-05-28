'''
Training script to perform image translation
'''
from configs.config import get_cfg_defaults
import torch
from torch import nn
import argparse
from dataloaders import BRATS2021ImageTranslationDataset
from utils import init_network, get_optimizer, get_scheduler
from gridencoder import GridEncoder
from tqdm.auto import tqdm
import numpy as np
from torch.nn import functional as F
from utils.losses import dice_loss_with_logits, dice_score_val, dice_loss_with_logits_batched, dice_score_binary
from utils.util import crop_collate_fn, format_raw_gt_to_brats, split_dataset
import tensorboardX
import os
from torch.utils.data import DataLoader
from os import path as osp
import nibabel as nib

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

    # keep track of best metrics (to save models)
    best_loss = np.inf
    best_psnr = 0

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
    dataset = BRATS2021ImageTranslationDataset(cfg.DATASET.TRAIN_SEG_DIR, cfg.DATASET.TRAIN_ENCODED_DIR, input_modalities=cfg.DATASET.INPUT_MODALITIES,
                                                        output_modality=cfg.DATASET.OUTPUT_MODALITY, sample_mode='sample')
    shuffle_seed = cfg.VAL.RANDOM_SHUFFLE_SEED if cfg.VAL.RANDOM_SHUFFLE else None
    print("Using shuffle seed = {}".format(shuffle_seed))
    train_dataset, val_dataset = split_dataset(dataset, cfg.VAL.FOLD, cfg.VAL.MAX_FOLDS, shuffle_seed)
    val_dataset.sample_mode = 'full'

    # set up dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)

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
        try:
            saved = torch.load(osp.join(cfg.EXP_NAME, 'best_model.pth'))
            network.load_state_dict(saved['network'])
            optim.load_state_dict(saved['optim'])
            start_epoch = saved['epoch'] + 1
            best_metrics = saved['metrics']
            print(f"Resuming from epoch {start_epoch}.")
        except:
            start_epoch = 0
            best_metrics = dict()
            print(f"Didn't find checkpoint, starting from epoch 0")
    
    # Load scheduler
    lr_scheduler = get_scheduler(cfg, optim, start_epoch)
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # start training
        pbar = tqdm(train_dataloader)
        lr = lr_scheduler.get_last_lr()[0]
        try:
            for i, datum in enumerate(pbar):
                optim.zero_grad()
                # get data
                embed = datum['encoder'].cuda() # [B, N, C]
                embed = embed.squeeze(2).permute(1, 0, 2).contiguous()  # [N, B, C]
                # coords = [B, N, 3], dims = [B, 3]
                coords = datum['xyz'].cuda() / (datum['dims'][:, None].cuda() - 1) * 2 - 1
                coords = coords.permute(1, 0, 2).contiguous()  # [N, B, 3]
                gt_img = datum['image'].cuda()   # [B, N]
                # forward
                preds = network(embed, coords)  # [N, B, out]
                preds = preds.permute(1, 0, 2).squeeze(2) # [B, N, out]
                loss = F.mse_loss(preds, gt_img)
                loss.backward()
                optim.step()
                pbar.set_description(f'Epoch:{epoch} lr: {lr:.4f} Loss:{loss.item():.4f}')
                writer.add_scalar('train/loss', loss.item(), epoch*len(train_dataloader)+i)
        except KeyboardInterrupt:
            print(f"Skipping rest of training for epoch {epoch}.")

        lr_scheduler.step()

        # val and save 
        if (epoch+1) % cfg.VAL.EVAL_EVERY == 0 or epoch == cfg.TRAIN.EPOCHS-1:
            # validation metrics
            total_psnr = []
            total_loss = []
            with torch.no_grad():
                pbar = tqdm(val_dataloader)
                for i, datum in enumerate(pbar):
                    optim.zero_grad()
                    # get data
                    embed = datum['encoder'].cuda() # [B, N, C]
                    embed = embed.squeeze(2).permute(1, 0, 2).contiguous()  # [N, B, C]
                    # coords = [B, N, 3], dims = [B, 3]
                    coords = datum['xyz'].cuda() / (datum['dims'][:, None].cuda() - 1) * 2 - 1
                    coords = coords.permute(1, 0, 2).contiguous()  # [N, B, 3]
                    gt_img = datum['image'].cuda()   # [B, N]
                    # forward
                    preds = network(embed, coords)  # [N, B, out]
                    preds = preds.permute(1, 0, 2).squeeze(2) # [B, N, out]
                    loss = F.mse_loss(preds, gt_img)
                    psnr = 10 * torch.log10(4 / loss)  # image range is from -1 to 1
                    loss.backward()
                    optim.step()
                    pbar.set_description(f'Epoch:{epoch} lr: {lr:.4f} Loss:{loss.item():.4f}, PSNR:{psnr.item():.4f}')
                    total_psnr.append(psnr.item())
                    total_loss.append(loss.item())
                # save metrics
                mean_loss = np.mean(total_loss)
                mean_psnr = np.mean(total_psnr)
                writer.add_scalar('val/loss', mean_loss, epoch)
                writer.add_scalar('val/psnr', mean_psnr, epoch)
                # save the model
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_psnr = mean_psnr
                    torch.save({'epoch': epoch,
                                'network': network.state_dict(),  
                                'optim': optim.state_dict(),
                                'metrics': {'loss': mean_loss, 'psnr': mean_psnr},
                                }, osp.join(cfg.EXP_NAME, "best_model.pth"))
                    print(f"Saved best model with loss {best_loss:.4f} and psnr {best_psnr:.4f} at epoch {epoch}.") 
