import torch
from torch.utils.data import Dataset
from glob import glob
from os import path as osp
import nibabel as nib
from torch.utils.data.dataset import ConcatDataset, Dataset
from utils import z_score_normalize, uniform_normalize
import numpy as np
from typing import List
import os

class BRATS2021Dataset(Dataset):
    '''
    Images are of size (155, 240, 240)
    sample modes are 'random', 'full' (for random and full sampling of points), 
                    'randomseg', 'fullseg' (for random and full sampling of points and the segmentation mask)
    multimodal: should we try to encode multimodal inputs (if false, use `mlabel`th image )
    mlabel: if multimodal is false, which channels to use
    '''
    def __init__(self, root_dir, augment=True, sample='random', num_points=25000, multimodal=True, mlabel=0, winsorize=100.0):
        super().__init__()
        self.root_dir = root_dir
        self.winsorize = winsorize
        if winsorize < 100:
            print(f"Clamping images to {winsorize}th percentile")
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
        if self.winsorize < 100.0:
            images = [torch.clamp(img, 0, np.percentile(img, self.winsorize)) for img in images]
        images = [uniform_normalize(img) for img in images]
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


class BRATS2021ImageTranslationDataset(Dataset):
    ''' 
    This dataset takes the directories to the images, and the encoded paths, along with the set of 
    input modalities (can be a set), and the output modality (one modality only)

    The input modalities are simply extracted from the multimodal encoding, and the output modality is given
    as a pair of coordinates and the intensity values.     

    image_dir: directory to the images
    encoder_dir: directory to the encoders
    input_modalities: list of modalities to use as input  (0-3)
    output_modality: modality to use as output (0-3)
    sample_mode: 'full' or 'sample' (sample randomly from output image)
    winsorize: winsorize the output image to this percentile
    maximum_intensity: maximum intensity of the output image
    '''
    def __init__(self, image_dir:str, encoder_dir:str, input_modalities: List[int] = [0], output_modality: int = 1,
                 sample_mode: str = 'full', winsorize: float = 99.0, maximum_intensity = np.inf, num_samples: int = 100000) -> None:
        super().__init__()
        self.image_root_dir = image_dir
        self.encoder_root_dir = encoder_dir
        self.winsorize = winsorize
        self.sample_mode = sample_mode
        self.maximum_intensity = maximum_intensity
        self.num_samples = num_samples
        self.isfinite_max = np.isfinite(maximum_intensity)
        if output_modality in input_modalities:
            assert False, "Output modality cannot be in input modalities"
        if len(input_modalities) == 0:
            assert False, "Input modalities cannot be empty"
        self.input_modalities = input_modalities
        self.output_modality = output_modality
        # get all directories containing the files
        self.image_dirs = sorted(glob(os.path.join(image_dir, '*')))
        self.encoded_files = sorted(glob(os.path.join(encoder_dir, 'encoder*')))
        assert len(self.image_dirs) == len(self.encoded_files), "Number of images and encoders must be the same"
    
    def __len__(self,):
        return len(self.image_dirs)
    
    def __getitem__(self, index):
        # return self.encoded_files[index], self.image_dirs[index]
        encoder = torch.load(self.encoded_files[index], map_location='cpu')['embeddings'] / 0.05
        N, C = encoder.shape
        Cby4 = C // 4
        assert C % 4 == 0, "Number of channels must be a multiple of 4 (to extract each channel)"
        # get the input modalities
        inputs = [encoder[:, i*Cby4:(i+1)*Cby4] for i in self.input_modalities]
        inputs = torch.cat(inputs, dim=1)  # [N, Cby4*len(input_modalities)]

        out_image = sorted(glob(os.path.join(self.image_dirs[index], '*.nii.gz')))
        out_image = list(filter(lambda x: 'seg' not in x, out_image))
        assert len(out_image) == 4
        out_image = out_image[self.output_modality]
        out_image = torch.from_numpy(nib.load(out_image).get_fdata()).float()
        if self.winsorize < 100.0:
            out_image = torch.clamp(out_image, 0, np.percentile(out_image, self.winsorize))
        if self.isfinite_max:
            out_image[out_image > self.maximum_intensity] = self.maximum_intensity
        out_image = uniform_normalize(out_image)
        H, W, D = out_image.shape
        # check sampling strategy
        if self.sample_mode == 'full':
            x, y, z = torch.meshgrid(torch.arange(H), torch.arange(W), torch.arange(D), indexing='ij')
            x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
            imgpoints = out_image.reshape(-1)  # [H*W*D]
        elif self.sample_mode == 'sample':
            x, y, z = [torch.randint(0, dim, (self.num_samples,)) for dim in [H, W, D]]
            imgpoints = out_image[x, y, z]
        else:
            raise NotImplementedError("Sampling mode not implemented")
        # collate coordinates
        x, y, z = [t.float()/(dim-1)*2.0 - 1 for t, dim in zip([x, y, z], [H, W, D])]  # [H*W*D]
        xyz = torch.stack([x, y, z], dim=-1)  # [H*W*D, 3]
        return {
            'encoder': inputs, 
            'image': imgpoints,
            'xyz': xyz,
            'image_name': self.image_dirs[index],
            'encoder_name': self.encoded_files[index],
            'index': index,
        }


if __name__ == '__main__':

    ### Check for image translation dataset
    dataset = BRATS2021ImageTranslationDataset("/data/rohitrango/BRATS2021/training", \
                                               "/data/rohitrango/Implicit3DCNNTasks/brats2021_unimodal", 
                                                input_modalities=[0, 1], output_modality=2, sample_mode='sample')
    for i in range(5):
        data = dataset[i]
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape, v.min(0).values, v.max(0).values, v.abs().mean(0))
            else:
                print(k, v)
        print()

    ### Check for encoded dataset
    # dataset = BRATS2021EncoderSegDataset('/data/Implicit3DCNNTasks/brats2021_unimodal', '/data/BRATS2021/training/', val_fold=1)
    # for i in range(5):
    #     enc, seg = dataset.both_files[i]
    #     print(enc, seg)

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
