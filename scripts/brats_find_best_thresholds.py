''' Find the best thresholds for the BRATS dataset.'''
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
from utils.util import crop_collate_fn, format_raw_gt_to_brats
import tensorboardX
import os
from torch.utils.data import DataLoader
from os import path as osp
from scripts.train_brats_segmentation import eval_validation_data
from glob import glob
import nibabel as nib
from utils.losses import dice_loss_with_logits, dice_score_val, dice_loss_with_logits_batched, dice_score_binary, focal_loss_with_logits
import multiprocessing
import pickle as pkl

def worker(predfilename, gtfilename, thresholds):
    ''' load the files and compute dice scores for all thresholds '''
    gt_vol = nib.load(gtfilename).get_fdata()
    et, tc, wt = format_raw_gt_to_brats(torch.from_numpy(gt_vol).long().squeeze(-1))
    pred_vol = torch.from_numpy(nib.load(predfilename).get_fdata()).float()
    metrics = {
        "ET": [],
        "TC": [],
        "WT": []
    }
    for thres in thresholds:
        pred_thres = (pred_vol >= thres).float()
        pred_et, pred_tc, pred_wt = [pred_thres[..., i] for i in range(3)]
        dice_et, dice_tc, dice_wt = [dice_score_binary(pred, gt).item() for pred, gt in zip([pred_et, pred_tc, pred_wt], [et, tc, wt])]
        metrics['ET'].append(dice_et)
        metrics['TC'].append(dice_tc)
        metrics['WT'].append(dice_wt)
    print(metrics, predfilename)
    for k, v in metrics.items():
        metrics[k] = np.array(v)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--num_processes', type=int, default=16, help="Number of processes to use")
    args = parser.parse_args()
    # find the best threshold
    preds = sorted(glob("./" + args.exp_name + "/pred*.nii.gz"))
    gt = sorted(glob("./" + args.exp_name + "/gt*.nii.gz"))
    thresholds = np.arange(0.05, 1, 0.05)
    print(thresholds)
    predgt = list(zip(preds, gt))

    with multiprocessing.Pool(args.num_processes) as pool:
        metrics_all = pool.starmap(worker, [(pred, gt, thresholds) for pred, gt in predgt])
    # save metrics
    with open("./" + args.exp_name + "/threshold_metrics.pkl", "wb") as f:
        pkl.dump(metrics_all, f)

    metrics = {}
    for k in metrics_all[0].keys():
        metrics[k] = np.array([m[k] for m in metrics_all])  # [N, T]
        metrics[k] = metrics[k].mean(0)
    
    for k in metrics.keys():
        print(f"Best threshold for {k} is {thresholds[np.argmax(metrics[k])]} with dice score {metrics[k].max()}.")
    # for _pred, _gt in zip(preds, gt):
    #     metrics = worker(_pred, _gt, thresholds)
    #     print(metrics)
    #     break