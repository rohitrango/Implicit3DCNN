'''
Evaluation script to evaluate performance of encoder + finetuning on image synthesis
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
from networks.unet_refine import ResidualUNet3D, UNet2D
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    # Load parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help='Name of experiment')
    # parse args and get config
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg_file = osp.join('experiments/', args.exp_name, 'config_finetune.yaml')
    cfg.merge_from_file(cfg_file)
    print(cfg)
    cfg.EXP_NAME = osp.join('experiments/', args.exp_name)
    # set up datasets
    dataset = BRATS2021ImageTranslationDataset(cfg.DATASET.TRAIN_SEG_DIR, cfg.DATASET.TRAIN_ENCODED_DIR, input_modalities=cfg.DATASET.INPUT_MODALITIES,
                                                        output_modality=cfg.DATASET.OUTPUT_MODALITY, sample_mode='full', include_images=True)
    shuffle_seed = cfg.VAL.RANDOM_SHUFFLE_SEED if cfg.VAL.RANDOM_SHUFFLE else None
    print("Using shuffle seed = {}".format(shuffle_seed))
    train_dataset, val_dataset = split_dataset(dataset, cfg.VAL.FOLD, cfg.VAL.MAX_FOLDS, shuffle_seed)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
    # compute offsets and resolutions
    dummy = GridEncoder(level_dim=cfg.ENCODE.LEVEL_DIM, desired_resolution=cfg.ENCODE.DESIRED_RESOLUTION, gridtype='tiled', align_corners=True, log2_hashmap_size=19)
    resolutions = dummy.resolutions.cuda()
    offsets = dummy.offsets.cuda()
    del dummy
    # init network and finetuning unet
    network = init_network(cfg, offsets, resolutions).cuda()
    network.load_state_dict(torch.load(osp.join(cfg.EXP_NAME, 'best_model.pth'))['network'])
    network.eval()
    # init finetuning unet
    input_channels = len(cfg.DATASET.INPUT_MODALITIES)
    # finetune model takes inputs + predicted coarse segmentation
    if cfg.FINETUNE.IS_3D:
        unet_model = ResidualUNet3D(input_channels + 1, 1, num_levels=2, f_maps=32).cuda()
    else:
        unet_model = UNet2D(input_channels + 1, 1, num_levels=2, f_maps=32).cuda()
    unet_model.load_state_dict(torch.load(osp.join(cfg.EXP_NAME, 'best_model_finetune.pth'))['network'], strict=True)
    unet_model.eval()
    
    psnrs = []
    ssims = []
    # validation metrics
    unet_model.eval()
    with torch.no_grad():
        pbar = tqdm(val_dataloader)
        for i, datum in enumerate(pbar):
            # get data
            embed = datum['encoder'].cuda() # [B, N, C]
            embed = embed.squeeze(2).permute(1, 0, 2).contiguous()  # [N, B, C]
            # coords = [B, N, 3], dims = [B, 3]
            coords = datum['xyz'].cuda() / (datum['dims'][:, None].cuda() - 1) * 2 - 1
            coords = coords.permute(1, 0, 2).contiguous()  # [N, B, 3]
            gt_img = datum['image'].cuda()   # [B, N]
            # forward
            preds = network(embed, coords)  # [N, B, out]
            preds = preds.permute(1, 0, 2).squeeze(2) # [B, N]
            H, W, D = datum['dims'][0].int()
            preds = preds.reshape(-1, 1, H, W, D)
            # get input images
            input_images = torch.cat([datum['input_images'].cuda(), preds], dim=1)  # [B, N+1, H, W, D]
            # get fine predictions depending on what mode is used
            pred_fine_3d = None
            if cfg.FINETUNE.IS_3D:
                raise NotImplementedError
            else:
                pred_fine_3d = []
                gt_image_3d = gt_img.reshape(-1, 1, H, W, D)
                for i in range(D):
                    preds_fine = unet_model(input_images[..., i])
                    if cfg.FINETUNE.ADD_RESIDUAL_END:
                        preds_fine = preds_fine + preds[..., i]
                    # preds_fine = preds[..., i]
                    pred_fine_3d.append(preds_fine)
                pred_fine_3d = torch.stack(pred_fine_3d, dim=-1)   # [B, 1, H, W, D]
                # compute metrics
                # print(pred_fine_3d.shape, gt_image_3d.shape, pred_fine_3d.min(), pred_fine_3d.max(), gt_image_3d.min(), gt_image_3d.max())
                mse = F.mse_loss(pred_fine_3d, gt_image_3d)
                psnr = 10 * torch.log10(4 / mse)
                ssim_v = ssim(pred_fine_3d[0,0].cpu().numpy(), gt_image_3d[0,0].cpu().numpy(), data_range=2)
                psnrs.append(psnr.item())
                ssims.append(ssim_v)
                pbar.set_description("PSNR: {:.4f}, SSIM: {:.4f}".format(np.mean(psnrs), np.mean(ssims)))
    
    print("PSNR: {:.4f}, std: {:.4f}".format(np.mean(psnrs), np.std(psnrs)))                 
    print("SSIM: {:.4f}, std: {:.4f}".format(np.mean(ssims), np.std(ssims)))
