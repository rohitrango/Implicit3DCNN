import torch
from torch.utils.data import Dataset
from glob import glob
from os import path as osp
import nibabel as nib
from utils import z_score_normalize, uniform_normalize
import numpy as np

class BRATS2021Dataset(Dataset):
    '''
    Images are of size (155, 240, 240)
    sample modes are 'random', 'full' (for random and full sampling of points), 
                    'randomseg', 'fullseg' (for random and full sampling of points and the segmentation mask)
    '''
    def __init__(self, root_dir, augment=True, sample='random', num_points=25000):
        super().__init__()
        self.root_dir = root_dir
        self.augment = augment
        self.sample = sample
        self.dirs = sorted(glob(osp.join(root_dir, 'BraTS2021_*')))
        self.num_points = num_points
        self.files = dict()
        self.seg   = dict()
        for dir in self.dirs:
            allniftis = sorted(glob(osp.join(dir, '*.nii.gz')))
            segnifti = [n for n in allniftis if 'seg' in n][0]
            inputsnifti = [n for n in allniftis if 'seg' not in n]
            self.files[dir] = inputsnifti
            self.seg[dir] = segnifti
    
    def __len__(self):
        return len(self.dirs) if not self.augment else 8*len(self.dirs)    # randomly flip given coordinates
    
    def __getitem__(self, index):
        fullidx = index
        if self.augment:
            index = index % len(self.dirs)
            augidx = index // len(self.dirs)
        else:
            augidx = None
        # load image
        images = [torch.from_numpy(nib.load(f).get_fdata()).float() for f in self.files[self.dirs[index]]]
        images = [uniform_normalize(img) for img in images]
        # images = [z_score_normalize(img) for img in images]
        H, W, D = images[0].shape
        seg = None
        # check for segmentation
        if 'seg' in self.sample:
            seg = torch.from_numpy(nib.load(self.seg[self.dirs[index]]).get_fdata()).int()
            seg[seg == 4] = 3

        if augidx is not None:
            iflip, jflip, kflip = augidx // 4, (augidx // 2) % 2, augidx % 2
            axes = []
            if iflip:
                axes.append(0)
            if jflip:
                axes.append(1)
            if kflip:
                axes.append(2)
            if axes != []:
                images = [torch.flip(img, dims=axes) for img in images]
                if seg is not None:
                    seg = torch.flip(seg, dims=axes)

        # now sample from them
        imgpoints, segpoints = None, None
        if 'random' in self.sample:
            xyz = [torch.randint(0, dim, size=(self.num_points,)) for dim in [H, W, D]]
            imgpoints = torch.cat([img[xyz[0], xyz[1], xyz[2], None] for img in images], dim=1)  # [num_points, 4]
            if seg is not None:
                segpoints = seg[xyz[0], xyz[1], xyz[2]]
            xyz = torch.stack(xyz, dim=1).int()  # [num_points, 3]
        elif 'full' in self.sample:
            xyz = torch.meshgrid([torch.arange(0, dim,) for dim in [H, W, D]], indexing='ij')  # [H, W, D]
            xyz = torch.stack(xyz, dim=-1).view(-1, 3)  # [H*W*D, 3]
            imgpoints = torch.cat([img[..., None] for img in images], dim=-1).reshape(-1, 4)  # [H*W*D, 4]
            if seg is not None:
                segpoints = seg.reshape(-1)  # [H*W*D]
        else:
            raise ValueError('Invalid sample mode')

        return {
            'index': fullidx,
            'imgidx': index,
            'augidx': augidx,
            'imgpoints': imgpoints,
            'segpoints': segpoints,
            'xyz': xyz,
            'dims': torch.tensor([H, W, D]).int(),
        }
            

if __name__ == '__main__':
    dataset = BRATS2021Dataset('/data/BRATS2021/training/', sample='randomseg')
    print(len(dataset))
    ds = dataset[0]
    for k, v in ds.items():
        if type(v) == torch.Tensor:
            print(k, v.shape, v.dtype, v.device, v.min(), v.max())
        else:
            print(k, v)
    norm = ds['xyz']/(ds['dims'][None]-1)*2 - 1
    print(norm, norm.dtype)