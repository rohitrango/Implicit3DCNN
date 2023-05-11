'''
Code to generate segmentation volumes for the BRATS2021 validation set
Values are hardcoded for now, change them later

Run as follows:
    python brats_validationset_generateseg.py --folds 8 --cur_fold 0-7 --experiment_names <list of experiment names> \
        --mode [create_embedding | create_segmentation]
'''
import torch
from torch import nn
from queue import Queue
from gridencoder import GridEncoder
from dataloaders import BRATS2021Dataset
from glob import glob
import nibabel as nib
from multiprocessing import Process
from utils import uniform_normalize
from time import sleep
from os import path as osp
from tqdm import tqdm
import argparse
import gc
from utils import init_network
from configs.config import get_cfg_defaults
import numpy as np

ROOT_DIR = "/data/rohitrango/BRATS2021/val/"
ENCODER_DIR = "/data/rohitrango/Implicit3DCNNTasks/brats2021_unimodal"
OUT_DIR = "/data/rohitrango/Implicit3DCNNTasks/brats2021_unimodal_val/"
NUM_PTS = 200000
EPOCHS = 2500

@torch.no_grad()
def create_seg(fold_id, max_folds, experiment_names):
    '''
    Given a fold id and max folds, split the work between the encoders, load the networks,
    and create the segmentation volumes
    '''
    print("Running create_seg with fold {}/{}".format(fold_id, max_folds))
    encoder_paths = list(sorted(glob(OUT_DIR + "/*.pth") ))[fold_id::max_folds]
    device = torch.device("cuda")
    ## load experiments
    enc = GridEncoder(level_dim=2, desired_resolution=196).to(device)
    offsets = enc.offsets.cuda()
    resolutions = enc.resolutions.cuda()
    networks = []
    for exp in experiment_names:
        exp_path = osp.join('experiments', exp)
        cfg_path = osp.join(exp_path, 'config.yaml')
        cfg = get_cfg_defaults()
        cfg.merge_from_file(cfg_path)
        network = init_network(cfg, offsets, resolutions).cuda()
        network.load_state_dict(torch.load(osp.join(exp_path, 'best_model.pth'))['network'], strict=True)
        network.eval()
        network.requires_grad_(False)
        networks.append(network)
        print("Loaded network from {}".format(exp_path))
    num_networks = len(networks)
    
    H, W, D = 240, 240, 155

    # load encoders
    for enc in encoder_paths:
        # for this encoder, run all the networks through this encoding, and visualize the segmentation
        ########################
        allseg = 0
        encoder = torch.load(enc).cuda()[:, None].contiguous()  # [N, 1, C]
        encoder /= 0.05
        xyz = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), torch.linspace(-1, 1, D), indexing='ij')
        xyz = torch.stack(xyz, dim=-1).cuda()  # [H, W, D, 3]
        xyzflat = xyz.reshape(-1, 1, 3)  # [H*W*D, 1, 3]
        for net in networks:
            allseg = allseg + torch.sigmoid(net(encoder, xyzflat).reshape(H, W, D, -1))[..., 1:]  # [H, W, D, C], discard the 0th index which is the background
        allseg = allseg / num_networks
        # save it
        nib.save(nib.Nifti1Image(allseg.cpu().numpy(), np.eye(4)), enc.replace(".pth", ".nii.gz"))
        print("Saved segmentation to {}".format(enc.replace(".pth", ".nii.gz")))



def worker(fold_id, max_folds, experiment_names):
    '''
    Given a fold id and max folds, split the work, load the unimodal decoders (hardcoded paths) 
    corresponding images, encode them, save the embeddings

    `create_seg` function will handle creating the outputs
    '''
    # load decoders
    print("Running fold {} in process {}".format(fold_id, max_folds))
    dirqueue = list(sorted(glob(ROOT_DIR + "/*") ))[fold_id::max_folds]
    device = torch.device("cuda")
    decoders = [nn.Sequential(
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        ).to(device) for _ in range(4)]
    for i in range(4):
        decoders[i].load_state_dict(torch.load(osp.join(ENCODER_DIR, f"decoder{i}.pth")))
        decoders[i].eval()
        decoders[i].requires_grad_(False)
    for q in dirqueue:
        files = sorted(glob(q + "/*"))
        print("Processing {:s}".format(q))
        images = [torch.from_numpy(nib.load(f).get_fdata()).float().to(device) for f in files]
        images = [uniform_normalize(img) for img in images]
        encoders = [GridEncoder(level_dim=2, desired_resolution=196).to(device) for _ in range(4)]
        optims = [torch.optim.Adam(enc.parameters(), lr=1e-3) for enc in encoders]
        # for each image, learn encoder
        for imgid, image in enumerate(images):
            H, W, D = image.shape
            pbar = tqdm(range(EPOCHS))
            for iter in pbar:
                x = torch.randint(0, H, (NUM_PTS, 1), device=device)
                y = torch.randint(0, W, (NUM_PTS, 1), device=device)
                z = torch.randint(0, D, (NUM_PTS, 1), device=device)
                coords = torch.cat((2.0*x/(H-1) - 1, 2.0*y/(W-1) - 1, 2.0*z/(D-1) - 1), dim=1).float()
                img = image[x[:,0], y[:,0], z[:,0]].view(-1, 1)
                pred_img = decoders[imgid](encoders[imgid](coords))
                loss = ((pred_img - img)**2).mean()
                optims[imgid].zero_grad()
                loss.backward()
                optims[imgid].step()
                pbar.set_description("Image {:d} Iter {:d} Loss {:.4f}".format(imgid, iter, loss.item()))
        # save embedding
        for optim in optims:
            del optim
        embedding = torch.cat([enc.embeddings for enc in encoders], dim=1)  # [N, C]
        outpath = osp.join(OUT_DIR, q.split('/')[-1] + '.pth')
        torch.save(embedding, outpath)
        print("Saved to {}".format(outpath))
        gc.collect()
        # now run inference on all experiment models
        # with torch.no_grad():
        #     xyz = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), torch.linspace(-1, 1, D), indexing='ij')
        #     xyz = torch.stack(xyz, dim=-1).to(device)  # [H, W, D, 3]



if __name__ == '__main__':
    # turns out serial implementation is the fastest in this case
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_folds", type=int, default=1)
    parser.add_argument("--cur_fold", type=int, default=0)
    parser.add_argument('--experiment_names', type=str, required=False, nargs='+', help='Name of the experiments to use')
    parser.add_argument('--mode', type=str, default="create_embedding", help='Mode of operation (create_embedding, create_segmentation)')

    args = parser.parse_args()
    cur_fold, folds = args.cur_fold, args.max_folds
    assert cur_fold < folds and cur_fold >= 0
    if args.mode == "create_embedding":
        worker(cur_fold, folds, args.experiment_names)
    elif args.mode == "create_segmentation":
        create_seg(cur_fold, folds, args.experiment_names)
    else:
        raise ValueError("Invalid mode {}".format(args.mode))
