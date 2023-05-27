''' 
Script to encode the BRATS dataset into our representation
'''
import argparse
from dataloaders import BRATS2021Dataset
from networks.tensorf import TensoRFEncoder
import torch
from torch import nn
from tqdm import tqdm
from os import path as osp
import numpy as np
from torch.nn import functional as F
import gc
from configs.config import get_cfg_defaults
import os
from scripts.encode_brats import _to_cpu

parser = argparse.ArgumentParser(description='Encode the BRATS dataset into our representation')
parser.add_argument('--cfg_file', type=str, required=True)
parser.add_argument('--root_dir', type=str, help='Path to the BRATS directory', default="/data/rohitrango/BRATS2021/training/")
parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory', default="/data/rohitrango/Implicit3DCNNTasks/brats2021-tensorf/")
parser.add_argument('--skip_stage1', action='store_true', help='Skip stage 1 and load the decoder from the output directory')
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    # multimodal configs
    multimodal = cfg.ENCODE.MULTIMODAL
    mlabel = cfg.ENCODE.MLABEL

    # encoder params
    image_size = (256, 256, 256)
    num_features = 32
    agg_method = 'cat'
    agg_factor = 3 if agg_method == 'cat' else 1

    if osp.exists(args.output_dir) and multimodal:
        print(f"Path {args.output_dir} exists and multimodal mode is on.")
        exit(0)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = BRATS2021Dataset(root_dir=args.root_dir, augment=False, num_points=cfg.ENCODE.NUM_POINTS, multimodal=cfg.ENCODE.MULTIMODAL, mlabel=cfg.ENCODE.MLABEL,
                                winsorize=cfg.ENCODE.WINSORIZE_PERCENTILE)
    encoder = TensoRFEncoder(image_size, num_features, agg_method=agg_method).cuda()
    encoder_params = []
    print(f"Using multimodal = {multimodal} with mlabel = {mlabel}")
    # save a copy of state dict for each image
    if multimodal:
        decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(agg_factor * num_features, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4)
        ).cuda()
    else:
        decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(agg_factor * num_features, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        ).cuda()

    # initialize encoder and decoder optims
    encoder_lr = 5e-3
    decoder_lr = 1e-3
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=encoder_lr) 
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=decoder_lr, weight_decay=1e-6)
    encoder_optim_params = dict()
    if not args.skip_stage1:
        for i in tqdm(range(cfg.ENCODE.STAGE1_TRAIN_IMAGES)):
            encoder_params.append(TensoRFEncoder(image_size, num_features, agg_method=agg_method).state_dict())
        # run this
        for epoch in range(cfg.ENCODE.NUM_EPOCHS_STAGE1):
            idxperm = np.random.permutation(cfg.ENCODE.STAGE1_TRAIN_IMAGES) 
            pbar = tqdm(idxperm)
            loss_avg = []
            for idx in pbar:
                encoder.load_state_dict(encoder_params[idx])
                if encoder_optim_params.get(idx) is None:
                    encoder_optim_params[idx] = torch.optim.Adam(encoder.parameters(), lr=encoder_lr).state_dict()
                encoder_optim.load_state_dict(encoder_optim_params[idx])
                # zero grad
                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                # load the data
                data = dataset[idx]
                xyzfloat = (data['xyz']*2.0+1)/data['dims'][None] - 1
                enc = encoder(xyzfloat[None].float().cuda())[0]
                pred_intensities = decoder(enc) 
                # loss and backward
                loss = F.mse_loss(pred_intensities, data['imgpoints'].cuda())
                loss.backward()
                loss_avg.append(loss.item())
                pbar.set_description("Epoch: {}, Loss: {:06f}, Running loss: {:06f}".format(epoch, loss.item(), np.mean(loss_avg)))
                encoder_optim.step()
                decoder_optim.step()
                # save it to cpu memory
                encoder_params[idx] = _to_cpu(encoder.state_dict())
                encoder_optim_params[idx] = _to_cpu(encoder_optim.state_dict())
        
        # save decoder
        torch.save(decoder.state_dict(), osp.join(args.output_dir, "decoder.pth" if multimodal else f"decoder{mlabel}.pth"))
        torch.save(decoder_optim.state_dict(), osp.join(args.output_dir, "decoder_optim.pth" if multimodal else f"decoder_optim{mlabel}.pth"))
        print("Saved decoder, preparing for stage 2...")
        decoder_optim.zero_grad()
    else:
        print("Loading decoder from output directory...")
        decoder.load_state_dict(torch.load(osp.join(args.output_dir, "decoder.pth" if multimodal else f"decoder{mlabel}.pth")), strict=True)

    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    # now learn each image individually
    del encoder_optim_params, encoder_params, encoder, encoder_optim
    del decoder_optim
    gc.collect()

    ## Stage 2
    dataset = BRATS2021Dataset(root_dir=args.root_dir, augment=False, num_points=cfg.ENCODE.NUM_POINTS, multimodal=cfg.ENCODE.MULTIMODAL, mlabel=cfg.ENCODE.MLABEL, sample='full')
    pbar = tqdm(range(len(dataset)))
    for idx in pbar:
        encoder = TensoRFEncoder(image_size, num_features, agg_method=agg_method).cuda()
        encoder_optim = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
        # retrieve the data
        datum = dataset[idx]
        xyzfloat = (datum['xyz']*2.0+1)/datum['dims'][None] - 1
        xyzfloat = xyzfloat.float().cuda()
        image = datum['imgpoints'].cuda()
        total_points = image.shape[0]
        subj = datum['subj']
        for i in range(cfg.ENCODE.NUM_EPOCHS_STAGE2):
            encoder_optim.zero_grad()
            minibatch = np.random.randint(total_points, size=(cfg.ENCODE.NUM_POINTS))
            xyzminibatch = xyzfloat[minibatch][None]
            imageminibatch = image[minibatch]
            # load the data
            pred_minibatch = decoder(encoder(xyzminibatch)[0])
            # loss and backward
            loss = F.mse_loss(pred_minibatch, imageminibatch)
            loss.backward()
            pbar.set_description("subj: {} iter: {}/{}, Loss: {:06f}".format(subj, i, cfg.ENCODE.NUM_EPOCHS_STAGE2, loss.item()))
            encoder_optim.step()
        # finished training, save the encoder
        torch.save(encoder.state_dict(), osp.join(args.output_dir, "encoder_{}{}.pth".format(subj, "" if multimodal else f"_{mlabel}")))
