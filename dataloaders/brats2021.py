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
    multimodal: should we try to encode multimodal inputs (if false, use `mlabel`th image )
    mlabel: if multimodal is false, which channels to use
    '''
    def __init__(self, root_dir, augment=True, sample='random', num_points=25000, multimodal=True, mlabel=0):
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
            try:
                segnifti = [n for n in allniftis if 'seg' in n][0]
            except:
                segnifti = None
            inputsnifti = [n for n in allniftis if 'seg' not in n]
            if not multimodal:
                inputsnifti = [inputsnifti[mlabel]]
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
        subj = self.dirs[index].split('/')[-1]
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
            imgpoints = torch.cat([img[xyz[0], xyz[1], xyz[2], None] for img in images], dim=1)  # [num_points, num_outs]
            if seg is not None:
                segpoints = seg[xyz[0], xyz[1], xyz[2]]
            xyz = torch.stack(xyz, dim=1).int()  # [num_points, 3]

        elif 'full' in self.sample:
            xyz = torch.meshgrid([torch.arange(0, dim,) for dim in [H, W, D]], indexing='ij')  # [H, W, D]
            xyz = torch.stack(xyz, dim=-1).view(-1, 3)  # [H*W*D, 3]
            imgpoints = torch.cat([img[xyz[..., 0], xyz[..., 1], xyz[..., 2], None] for img in images], dim=-1).reshape(-1, len(images))  # [H*W*D, 4]
            ### Check if meshgrid and full sampling are the same
            # imgpoint2 = torch.cat([img[..., None] for img in images], dim=-1).reshape(-1, 4)
            # print('avg', torch.abs(imgpoint2 - imgpoints).mean())
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
            'subj': subj,
            'dims': torch.tensor([H, W, D]).int(),
        }
            

class BRATS2021EncoderSegDataset(Dataset):
    ''' Dataset for loading the encoded features, coordinates and segmentation '''
    def __init__(self, encoded_root_dir, seg_root_dir, train=True, num_folds=5, val_fold=0):
        super().__init__()
        self.train = train
        self.encoded_root_dir = encoded_root_dir
        self.seg_root_dir = seg_root_dir
        # get all the encoders and segmentations  
        # (sorting should be consistent because we use the same naming convention for encoders and original segmentations)
        encoded_files = sorted(glob(osp.join(encoded_root_dir, 'encoder*.pth')))
        segm_files = sorted(glob(osp.join(seg_root_dir, '*/*seg*nii.gz')))
        assert len(encoded_files) == len(segm_files)
        # zip both files and split into folds
        both_files = list(zip(encoded_files, segm_files))
        N = len(both_files)
        # discard the val fold if train, else keep only the val fold
        both_files_split = np.array_split(both_files, num_folds)
        if train:
            del both_files_split[val_fold]
            both_files_split = [item for sublist in both_files_split for item in sublist]
        else:
            both_files_split = [item for item in both_files_split[val_fold]]
        # save it to files, and initialize weights
        self.both_files = both_files_split
        self.ce_weights = dict()
    
    def __len__(self):
        return len(self.both_files) * (1 if self.train else 8)   # divide it into 8 chunks of subsampled points for test, for training, select a random chunk

    def __getitem__(self, index):
        # get chunk and index size
        if self.train:
            chunk = np.random.randint(8)
        else:
            chunk = index % 8
            index = index // 8
        startx, starty, startz = (chunk // 4), ((chunk // 2) % 2), (chunk % 2)
        # get encoder and segmentation
        encoderfile, segmfile = self.both_files[index]
        data = dict(torch.load(encoderfile, map_location='cpu'))
        data['embeddings'] = data['embeddings'][:, None]   # [N, 1, C]
        data['embeddings'] /= 0.05
        if self.train:
            scale = np.random.rand()*0.4 + 0.8
            data['embeddings'] = data['embeddings'] * scale  # scale augmentation
        # load segmentation
        seg = torch.from_numpy(nib.load(segmfile).get_fdata()).int() 
        seg[seg == 4] = 3
        H, W, D = seg.shape
        # get count stats
        if self.ce_weights.get(index) is None:
            count = torch.bincount(seg.reshape(-1), minlength=4)
            weights = 1.0/(1 + count)
            weights = weights / weights.sum() * 4
            self.ce_weights[index] = weights
        # retrieve weights
        weights = self.ce_weights[index]
        # do this to collate more easily
        if self.train:
            H2, W2, D2 = [int(t - t%2) for t in [H, W, D]]
        else:
            H2, W2, D2 = H, W, D
        xyz = torch.meshgrid([torch.arange(s, dim, 2) for s, dim in [(startx, H2), (starty, W2), (startz, D2)]], indexing='ij')  # [H, W, D]
        xyz = torch.stack(xyz, dim=-1)              # [H2, W2, D2, 3]
        seg = seg[startx:H2:2, starty:W2:2, startz:D2:2]  # [H2, W2, D2]
        # add to data
        data['xyz'] = xyz.reshape(-1, 3).int()
        data['dims'] = torch.tensor([H, W, D]).int()
        data['segm'] = seg.reshape(-1)
        data['weights'] = weights
        return data


if __name__ == '__main__':
    ### Check for encoded dataset
    dataset = BRATS2021EncoderSegDataset('/data/Implicit3DCNNTasks/brats2021_unimodal', '/data/BRATS2021/training/', val_fold=1)
    for i in range(5):
        enc, seg = dataset.both_files[i]
        print(enc, seg)
    # print(len(dataset))
    # for idx in np.random.randint(len(dataset)//8, size=(20,)):
    #     datum = dataset[idx]
    #     enc, seg = dataset.both_files[idx]
    #     print(enc, seg)
    #     for k, v in datum.items():
    #         if(k == 'weights'):
    #             print(k, v)
    #             continue
    #         if type(v) == torch.Tensor:
    #             print(k, v.shape, v.dtype, v.device, v.min(), v.max())
    #         else:
    #             print(k, v)
    #     print()
    #     # print(idx, enc, seg)
    #     # print(idx, dataset[idx])

    ### Check for BRATS 2021 dataset
    # dataset = BRATS2021Dataset('/data/BRATS2021/training/', sample='random', multimodal=False, mlabel=2)
    # print(len(dataset))
    # ds = dataset[0]
    # for k, v in ds.items():
    #     if type(v) == torch.Tensor:
    #         print(k, v.shape, v.dtype, v.device, v.min(), v.max())
    #     else:
    #         print(k, v)
    # norm = ds['xyz']/(ds['dims'][None]-1)*2 - 1
    # print(norm, norm.dtype)
