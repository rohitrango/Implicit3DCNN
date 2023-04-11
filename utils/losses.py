import torch
from torch import nn
from torch.nn import functional as F

def dice_loss_with_logits(logits, segm, ignore_idx=0):
    # compute dice score for each logit
    N, D = logits.shape
    prob = F.softmax(logits, dim=-1)
    loss, count = 0, 0
    for d in range(D):
        if d == ignore_idx:
            continue
        gt_d = (segm == d).float()
        num = (2 * prob[:, d] * gt_d).mean() + 1e-8
        den = (prob[:, d] + gt_d).mean() + 1e-5
        loss = loss + (1 - num/den)
        count += 1
    # average it out
    return loss/count

def dice_score_val(pred, gt, num_classes=4, ignore_class=0):
    ret = []
    for i in range(num_classes):
        if i == ignore_class:
            continue
        num = 2 * (pred == i).float() * (gt == i).float()
        den = (pred == i).float() + (gt == i).float() + 1e-5
        ret.append(num.mean() / den.mean())
    return ret