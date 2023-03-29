''' Replication of the notebook `Initial experiments with BRATs` '''
import gridencoder as ge
import torch
from torch import nn
from torch.nn import functional as F
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import os
import numpy as np
from os import path as osp
from matplotlib import pyplot as plt

brats_path = '/data/BRATS2021/'
paths = sorted(glob(osp.join(brats_path, "*")))
paths = list(filter(lambda x: osp.isdir(x), paths))

def load_nifti(filepath):
    # print(f"Loading {filepath}")
    image = sitk.ReadImage(filepath)
    image = sitk.GetArrayFromImage(image)
    return image

def sizeof_net(network):
    ''' get size of network in MB '''
    res = 0
    for p in network.parameters():
        res += p.data.nelement() * p.data.element_size()
    return res / 1024**2

class ImplicitWrapper(nn.Module):
    def __init__(self, num_levels=16, desired_resolution=256, level_dim=2, base_resolution=16, \
                 log2_hashmap_size=19, decoder_num_hiddenlayers=1, decoder_num_nodes=256,
                 decoder_num_outputs=1,
                 num_encoders=1,
                ):
        super().__init__()
        self.encoder = [ge.GridEncoder(num_levels=num_levels,
                                      level_dim=level_dim,
                                     base_resolution=base_resolution,
                                     desired_resolution=desired_resolution,
                                     align_corners=True,
                                     gridtype='tiled',
                                     log2_hashmap_size=log2_hashmap_size) for _ in range(num_encoders)]
        self.encoder = nn.ModuleList(self.encoder)
        
        self.decoder = [nn.Linear(num_levels*level_dim, decoder_num_nodes), nn.LeakyReLU()]
        for i in range(decoder_num_hiddenlayers):
            self.decoder.append(nn.Linear(decoder_num_nodes, decoder_num_nodes))
            self.decoder.append(nn.LeakyReLU())
        self.decoder.append(nn.Linear(decoder_num_nodes, decoder_num_outputs))
        self.decoder = nn.Sequential(*self.decoder)
    
    def forward(self, x, idx=0):
        return self.decoder(self.encoder[idx](x))

def training_loop_multi_image(net, batch_size, n_iters, num_images, lr=1e-4, diff_optims=False, silent=False):
    # load all images
    imgs = [load_nifti(t1files[i]) for i in range(num_images)]
    H, W, D = imgs[0].shape
    maxDim = max([H, W, D])
    maxVal = max([np.max(i) for i in imgs])
    # center = np.array([H, W, D],  dtype=np.int32)//2            # [3, ]
    # centercuda = torch.cuda.FloatTensor(center)[:, None]        # [3, 1]
    # [H-1, W-1, D-1]
    dims = np.array([H, W, D]) - 1.0                        
    dimscuda = torch.cuda.FloatTensor(dims)[:, None]
    print(f"Training with {num_images} image(s).")
    
    # see if adam updates mess up with the encoders
    if diff_optims:
        optim_encoder = [torch.optim.Adam(net.encoder[i].parameters(), lr=lr) for i in range(num_images)]
        optim_decoder = torch.optim.Adam(net.decoder.parameters(), lr=lr)
    else:
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        
    ## training loop
    if silent:
        pbar = range(int(n_iters * num_images))
    else:
        pbar = tqdm(range(int(n_iters * num_images)))
    for i in pbar:
        if diff_optims:
            [o.zero_grad() for o in optim_encoder]
            optim_decoder.zero_grad()
        else:
            optim.zero_grad()
        # sample points randomly
        x, y, z = [np.random.randint(T, size=(1, batch_size)) for T in [H, W, D]]
        img_idx = np.random.randint(num_images)
        # img_idx = i % num_images
        
        # sample points
        val = imgs[img_idx][x[0], y[0], z[0]][..., None]*2.0/maxVal - 1   # [B, 1]
        coord = np.concatenate([x, y, z], axis=0).transpose(1, 0) # [B, 3]

        coord = coord/dims*2.0 - 1  
        # coord = coord - center[None]
        # coord = coord * 2 / maxDim

        # coord = coord / maxDim * 2 - 1
        # put to tensors
        val = torch.cuda.FloatTensor(val)
        coord = torch.cuda.FloatTensor(coord)
        pred = net(coord, img_idx)
        # get loss
        loss = F.mse_loss(val, pred)
        loss.backward()
        # check for different optims
        if diff_optims:
            optim_decoder.step()
            optim_encoder[img_idx].step()
        else:
            optim.step()

    if diff_optims:
        for x in optim_encoder:
            x.zero_grad()
            del x
        optim_decoder.zero_grad()
        del optim_decoder
    else:
        optim.zero_grad()
        del optim
    torch.cuda.empty_cache()
    psnrs = []
    with torch.no_grad():
        x, y, z = torch.meshgrid(torch.arange(H, device='cuda'), torch.arange(W, device='cuda'), torch.arange(D, device='cuda'), indexing='ij')
        coord = torch.cat([t.reshape(1, -1) for t in [x, y, z]], dim=0)  # [3, B]
        # coord = coord/maxDim*2-1

        # coord = coord - centercuda
        # coord = coord * 2 / maxDim

        coord = coord/dimscuda*2.0 - 1
        # permute
        coord = coord.permute(1, 0)  # [B, 3]
        B = coord.shape[0]
        net.eval()
        for i in range(num_images):
            allpred = []
            for it in range(0, B, 10000):
                end = min(B, it+10000)
                minicoord = coord[it:end]
                pred = net(minicoord, i)
                allpred.append(pred.data.cpu().numpy())
                # pred = pred.reshape(imgs[i].shape)
                # pred = pred.data.cpu().numpy()
            pred = np.concatenate(allpred, axis=0).reshape(imgs[i].shape) 
            gt   = imgs[i]*2.0/maxVal-1
            psnr = 10*np.log10(4/((gt - pred)**2).mean())
            psnrs.append(psnr)
    return psnrs

if __name__ == '__main__':
    # get all t1 files
    t1files = list(map(lambda x: glob(osp.join(x, '*_t1.nii.gz'))[0], paths))
    all_psnrs = []
    for num_images in range(1, 26):
        net = ImplicitWrapper(num_encoders=num_images).cuda()
        psnrs = training_loop_multi_image(net, 30000, 5000, num_images, diff_optims=True, silent=False)
        all_psnrs.append(psnrs)
        print(np.around(np.mean(psnrs), 4))
        print(np.around(psnrs, 4))
        del net
        torch.cuda.empty_cache()
    # Plot them PSNRs
    #figs, axs = plt.subplots(1, 2, figsize=(10, 5))
    #axs[0].plot(np.arange(len(all_psnrs))+1, [np.mean(x) for x in all_psnrs])
    #axs[0].scatter(np.arange(len(all_psnrs))+1, [np.mean(x) for x in all_psnrs])
    #axs[0].set_xlabel('#images')
    #axs[0].set_ylabel('PSNR')
    #axs[0].set_title('decoder layers=1')
    #plt.show()
    print(np.around(psnrs, 3))
