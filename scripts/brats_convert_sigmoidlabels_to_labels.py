''' Script that takes the results from `brats_validationset_generateseg.py` 
    (which are of size (3, H, W, D)) with per-channel sigmoid, and converts them into 
    a single-channel label volume of size (H, W, D) with values in {0, 1, 2, 3}

    This is the algorithm for converting raw segmentation to training format:
        # enhancing tumor, tumor core, whole tumor
        et = (segm == 3).float()
        tc = et + (segm == 1)
        wt = tc + (segm == 2)
        return et, tc, wt
    
    To reverse this, we do the following:
        # enhancing tumor, tumor core, whole tumor
        segm[0] = et = (segm==3)
        segm[1] = et + (segm==1) => (segm==1) = (segm[1] - segm[0])
        segm[2] = tc + (segm==2) => (segm==2) = (segm[2] - segm[1])
'''
import argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from os import path as osp

def bool_subtract(a, b):
    # returns a - b = a AND (NOT b)
    return np.logical_and(a, np.logical_not(b))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the input segmentation volumes')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output segmentation volumes')
    parser.add_argument('--input_glob', type=str, default="BraTS-GLI-[0-9]*-[0-9]*.nii.gz", help="Glob pattern for input segmentation volumes")
    # parser.add_argument('--input_glob', type=str, default="BraTS2021_[0-9]*.nii.gz", help="Glob pattern for input segmentation volumes")
    parser.add_argument('--thres', type=float, default=0.5)
    args = parser.parse_args()

    # TODO: Change this
    ROOT_IMG_DIR = "/data/rohitrango/BRATS2021/val/"

    # input segmentation volumes
    input_segmentations = sorted(glob(osp.join(args.input_dir, args.input_glob)))
    for inp in input_segmentations:
        # image name
        imgname = inp.split('/')[-1].split('.')[0]
        t1imgname = osp.join(ROOT_IMG_DIR, imgname, imgname + "-t1n.nii.gz")
        # print(t1imgname)
        img = sitk.ReadImage(inp)
        data = sitk.GetArrayFromImage(img) >= args.thres   # threshold at sigmoid threshold (=0.5)
        labelseg = np.zeros_like(data[0])
        labelseg[data[0] == 1] = 3
        labelseg[bool_subtract(data[1], data[0])] = 1
        labelseg[bool_subtract(data[2], data[1])] = 2
        # load actual image and get metadata
        t1img = sitk.ReadImage(t1imgname)
        # create simpleitk image
        labelimg = sitk.GetImageFromArray(labelseg.astype(np.uint8))
        labelimg.CopyInformation(t1img)
        # labelimg.SetSpacing(t1img.GetSpacing())
        # labelimg.SetDirection(t1img.GetDirection())
        # labelimg.SetOrigin(t1img.GetOrigin())
        outimgname = osp.join(args.output_dir, inp.split('/')[-1].replace('GLI', 'GLI-submit'))
        # outimgname = osp.join(args.output_dir, inp.split('/')[-1].replace("_", "-").replace(".nii.gz", "-{:03d}.nii.gz".format(args.timestamp)))
        sitk.WriteImage(labelimg, outimgname)
        print("Written to {}".format(outimgname))
        # modify data