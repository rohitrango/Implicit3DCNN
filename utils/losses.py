import torch
from torch import nn
from torch.nn import functional as F

def dice_loss_with_logits_batched(logits, segm, transform='softmax', ignore_idx=0):
    '''
    :logits: [B, N, D]
    :segm: [B, N]
    '''
    B1, N1, D = logits.shape
    if isinstance(segm, torch.Tensor):
        B2, N2 = segm.shape
        assert B1 == B2 and N1 == N2

    if transform == 'softmax':
        prob = F.softmax(logits, dim=-1)
    else:
        prob = torch.sigmoid(logits)

    loss, count = 0, 0
    for d in range(D):
        if d == ignore_idx:
            continue
        if isinstance(segm, torch.Tensor):
            gt_d = (segm == d).float()
        elif isinstance(segm, (list, tuple)):
            gt_d = segm[count]
        else:
            raise ValueError('segm must be a tensor or a list/tuple of tensors')

        num = (2 * prob[:, :, d] * gt_d).mean(1) + 1e-5
        den = (prob[:, :, d]**2 + gt_d).mean(1) + 1e-5
        loss = loss + (1 - num/den).mean()
        count += 1
    return loss/count


def dice_loss_with_logits(logits, segm, ignore_idx=0):
    # compute dice score for each logit
    N, D = logits.shape
    prob = F.softmax(logits, dim=-1)
    loss, count = 0, 0
    for d in range(D):
        if d == ignore_idx:
            continue
        gt_d = (segm == d).float()
        num = (2 * prob[:, d] * gt_d).mean() + 1e-5
        den = (prob[:, d]**2 + gt_d).mean() + 1e-5
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

def dice_score_binary(pred, gt):
    num = (2.0 * pred * gt).mean()
    den = (pred + gt + 1e-5).mean()
    return num/den

def focal_loss_with_logits(logits, gt, gamma=2.0):
    ''' 
    given logits and ground truth, compute focal loss 
    assumed no alpha
    '''
    p = torch.sigmoid(logits)
    ce_term = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
    p_t = p * gt + (1 - p) * (1 - gt)
    loss = ce_term * ((1 - p_t)**gamma)
    return loss.mean()
