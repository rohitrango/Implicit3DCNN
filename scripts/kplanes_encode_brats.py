''' 
Script to encode the BRATS dataset into the KPlanes encoder
'''
import argparse
from dataloaders import BRATS2021KPlanesDataset
from gridencoder import GridEncoder
import torch
from torch import nn
from tqdm import tqdm
from os import path as osp
import numpy as np
from torch.nn import functional as F
import gc
from configs.config import get_cfg_defaults
import os
from networks.kplanes.encoder import ImageKPlaneAttnEncoder
from collections import namedtuple
import tensorboardX as tb

class ResNet(nn.Module):
    def __init__(self, inc, out) -> None:
        super().__init__()
        self.ff = nn.Linear(inc, out)
        self.ln = nn.LayerNorm(inc)
        self.act = nn.GELU()
        self.res = nn.Linear(inc, out) if inc != out else nn.Identity()

    def forward(self, x):
        return self.act(self.ff(self.ln(x))) + self.res(x)

def choose_octet(img, octet):
    ''' given a binary mask, choose the octet '''
    H, W, D = img.shape[2:]
    i, j, k = octet
    startx, endx = (0, H//2) if i == 0 else (H//2, H)
    starty, endy = (0, W//2) if j == 0 else (W//2, W)
    startz, endz = (0, D//2) if k == 0 else (D//2, D)
    return img[:, :, startx:endx, starty:endy, startz:endz]

def _to_cpu(state_dict):
    # helper to convert the entire state dict into CPU (except things like int, float, str, etc)
    if isinstance(state_dict, torch.Tensor):
        return state_dict.cpu()
    elif isinstance(state_dict, (int, float, str)) or state_dict is None:
        return state_dict
    elif isinstance(state_dict, (tuple, list)):
        return [_to_cpu(x) for x in state_dict]

    for k, v in state_dict.items():
        if isinstance(v, dict):
            state_dict[k] = _to_cpu(v)
        elif isinstance(v, list):
            state_dict[k] = [_to_cpu(x) for x in v]
        else:
            state_dict[k] = _to_cpu(v)
    return state_dict

parser = argparse.ArgumentParser(description='Encode the BRATS dataset into our representation')
parser.add_argument('--cfg_file', type=str, required=True)
parser.add_argument('--root_dir', type=str, help='Path to the BRATS directory', default="/data/rohitrango/BRATS2021/training/")
parser.add_argument('--output_dir', type=str, required=False, \
                    help='Path to the output directory', default="/data/rohitrango/Implicit3DCNNTasks/kplanes-brats2021/")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--combine_method', type=str, default='add')
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    writer = tb.SummaryWriter('logs/')

    ### Load the dataset
    num_points=cfg.ENCODE.NUM_POINTS
    dataset = BRATS2021KPlanesDataset(root_dir=args.root_dir, augment=True, 
                               multimodal=cfg.ENCODE.MULTIMODAL, mlabel=cfg.ENCODE.MLABEL)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # create model config
    ModelCfgObj = namedtuple('modelcfg',
                          ['input_channels', 'hidden_init', 'downsample_init', 'dropout',
                             'num_attn_layers', 'num_heads', 'attn_out_channels'])
    inp_channels = 4 if cfg.ENCODE.MULTIMODAL else 1
    # hidden_init = 256 if cfg.ENCODE.MULTIMODAL else 128
    hidden_init = 512
    N_c = 128
    downsample_init = 4
    modelcfg = ModelCfgObj(inp_channels, hidden_init, downsample_init, 0.1, 3, 8, [256, 128, N_c])
    encoder = ImageKPlaneAttnEncoder(modelcfg).cuda()
    # load decoder
    decoder = nn.Sequential(
        # ResNet(N_c*3, 256),
        # ResNet(256, 256),
        # nn.LayerNorm(256),
        # nn.Linear(256, 4 if cfg.ENCODE.MULTIMODAL else 1)
        nn.Linear(N_c*3, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 4 if cfg.ENCODE.MULTIMODAL else 1)
    ).cuda()
    # net = nn.ModuleList([encoder, decoder])
    # optim = torch.optim.AdamW(net.parameters(), lr=1e-4)
    optim = torch.optim.AdamW(encoder.parameters(), lr=1e-4)
    optim_d = torch.optim.Adam(decoder.parameters(), lr=3e-4)
    power = 0.7
    sch = torch.optim.lr_scheduler.PolynomialLR(optim, args.num_epochs, power, verbose=True)
    sch_d = torch.optim.lr_scheduler.PolynomialLR(optim_d, args.num_epochs, power, verbose=True)

    # keep track of best loss and best PSNR
    best_loss = np.inf
    best_psnr = 0

    for epoch in range(args.num_epochs):
        losses_thisepoch = []
        psnr_thisepoch = []
        pbar = tqdm(dataloader)
        for datum in pbar:
            # load the data (normalized already)
            optim.zero_grad()
            optim_d.zero_grad()

            img = datum['images'].cuda()  # [B, C, H, W, D]  
            B, _, H, W, D = img.shape
            f_xy, f_yz, f_xz = encoder(img)    # [B, C, H, W], [B, C, W, D], [B, C, H, D]
            # sample random points and sample from the image and encoder
            # x = torch.rand(B, num_points)*2 - 1
            # y = torch.rand(B, num_points)*2 - 1
            # z = torch.rand(B, num_points)*2 - 1
            # x, y, z = x.cuda(), y.cuda(), z.cuda()
            ## encoded features
            # enc_fxy = F.grid_sample(f_xy, torch.stack([x, y], dim=-1).unsqueeze(1), align_corners=True)  # [B, C, 1, num_points]
            # enc_fyz = F.grid_sample(f_yz, torch.stack([y, z], dim=-1).unsqueeze(1), align_corners=True)  # [B, C, 1, num_points]
            # enc_fzx = F.grid_sample(f_xz, torch.stack([x, z], dim=-1).unsqueeze(1), align_corners=True)
            # if args.combine_method == 'multiply':
            #     enc_f = enc_fxy * enc_fyz * enc_fzx
            # elif args.combine_method == 'add':
            #     enc_f = enc_fxy + enc_fyz + enc_fzx  
            # else:
            #     raise NotImplementedError("Combine method not implemented")

            # Get full encoder image
            di = downsample_init
            enc_img = torch.cat([f_xy.unsqueeze(4).repeat(1, 1, 1, 1, D//di), \
                                 f_yz.unsqueeze(2).repeat(1, 1, H//di, 1, 1), f_xz.unsqueeze(3).repeat(1, 1, 1, W//di, 1)], dim=1)  # [B, C, H, W, D]
            
            # enc_img = f_xy.unsqueeze(4) * f_yz.unsqueeze(2) * f_xz.unsqueeze(3)  # [B, C, H, W, D]
            # enc_img = f_xy.unsqueeze(4) + f_yz.unsqueeze(2) + f_xz.unsqueeze(3)  # [B, C, H, W, D]
            # choose an octet
            octet = np.random.randint(8)
            octet = [octet // 4, (octet // 2) % 2, octet % 2]
            enc_img = choose_octet(enc_img, octet)
            enc_img = F.interpolate(enc_img, (H//2, W//2, D//2), mode='trilinear', align_corners=False)  # [B, C, H, W, D]
            # enc_img = F.interpolate(enc_img, (H, W, D), mode='trilinear', align_corners=True)  # [B, C, H, W, D]
            enc_img = enc_img.permute(0, 2, 3, 4, 1)  # [B, H, W, D, C]
            enc_img = decoder(enc_img)  # [B, H, W, D, C]
            enc_img = enc_img.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]
            gt_img = choose_octet(img, octet)
            # print(enc_img.shape, img.shape)
            # enc_f = enc_f.squeeze(2).permute(0, 2, 1)  # [B, num_points, C]
            # img_pred = decoder(enc_f).permute(0, 2, 1)  # [B, num_points, C]
            # get sampled image intensities
            # img_sampled = F.grid_sample(img, torch.stack([x, y, z], dim=-1)[:, None, None], align_corners=True)  # [B, C, 1, 1, num_points]
            # img_sampled = img_sampled[:, :, 0, 0, :].permute(0, 2, 1) # [B, num_points, C]
            # loss = F.mse_loss(img_pred, img_sampled)
            loss = F.mse_loss(gt_img, enc_img)
            psnr = 10 * torch.log10(4/loss).item()
            # step
            loss.backward()
            pbar.set_description("Epoch: {}/{}, Loss: {:06f}, PSNR: {:06f}".format(epoch, args.num_epochs, loss.item(), psnr))
            optim.step()
            optim_d.step()
            # append to list
            losses_thisepoch.append(loss.item())
            psnr_thisepoch.append(psnr)
            # write to tensorboard
            writer.add_scalar('Loss', loss.item(), epoch*len(dataloader) + pbar.n)
            writer.add_scalar('PSNR', psnr, epoch*len(dataloader) + pbar.n)

        sch.step()
        sch_d.step()
        # save the model
        print("Epoch statistics:")
        print("Epoch {}/{}".format(epoch, args.num_epochs))
        print("Mean loss: {}".format(np.mean(losses_thisepoch)))
        print("Mean PSNR: {}".format(np.mean(psnr_thisepoch)))
        if np.mean(psnr_thisepoch) > best_psnr:
            print("Saving model")
            best_psnr = np.mean(psnr_thisepoch)
            suffix = "" if cfg.ENCODE.MULTIMODAL else "{}".format(cfg.ENCODE.MLABEL)
            torch.save(_to_cpu(encoder.state_dict()), osp.join(args.output_dir, "encoder{}.pth".format(suffix)))
            torch.save(_to_cpu(decoder.state_dict()), osp.join(args.output_dir, "decoder{}.pth".format(suffix)))
