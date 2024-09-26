''' Script that takes the results from `brats_validationset_generateseg_submission.py` 
    (which are of size (3, H, W, D)) with per-channel sigmoid/binary mask, and converts them into 
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
import nibabel as nib
from tqdm import tqdm
import os
import shutil

def bool_subtract(a, b):
    # returns a - b = a AND (NOT b)
    return np.logical_and(a, np.logical_not(b))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True, help='Directory containing the input segmentation volumes')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output segmentation volumes')
    parser.add_argument('--override', action='store_true', help='Override the output directory if it exists')
    parser.add_argument('--subtractive', action='store_true', help='Use subtractive method for conversion')
    # parser.add_argument('--input_glob', type=str, default="BraTS-GLI-[0-9]*-[0-9]*.nii.gz", help="Glob pattern for input segmentation volumes")
    # # parser.add_argument('--input_glob', type=str, default="BraTS2021_[0-9]*.nii.gz", help="Glob pattern for input segmentation volumes")
    # parser.add_argument('--thres', type=float, default=0.5)
    args = parser.parse_args()
    # convert relative rootdirs to absolute
    ROOT_PATH = "/data/rohitrango/Implicit3DCNNTasks/brats2021_ce_unimodal_val/"
    input_dirs = [osp.abspath(osp.join(ROOT_PATH, x)) for x in args.input_dirs]
    input_files = [sorted(glob(osp.join(x, "*.nii.gz"))) for x in input_dirs]
    # create output dir
    output_dir = osp.join(ROOT_PATH, args.output_dir)
    if osp.exists(output_dir):
        if args.override:
            shutil.rmtree(output_dir)
        else:
            print("Output directory already exists, exiting (use override option to delete) ...")
            exit(0)
    os.makedirs(output_dir)

    for files in tqdm(list(zip(*input_files))):
        input_data = [nib.load(f) for f in files]
        segms = [x.get_fdata() for x in input_data]
        data_types = set([x.header.get_data_dtype() for x in input_data])
        assert len(data_types) == 1, ("Data types of input files are not the same", data_types)
        mean_segm = (np.mean(segms, axis=0) >= 0.5)
        # create final segmentation
        final_segm = mean_segm[..., 0] * 0
        et, tc, wt = [mean_segm[..., i].astype(bool) for i in range(3)]
        if args.subtractive:
            tc = np.logical_and(tc, wt)
            et = np.logical_and(et, tc)
        final_segm[et] = 3
        final_segm[bool_subtract(tc, et)] = 1
        final_segm[bool_subtract(wt, np.logical_or(et, tc))] = 2
        # get source and save  
        source_hdr = input_data[0].header.copy()
        source_affine = input_data[0].affine.copy()
        seg_img = nib.Nifti1Image(final_segm.astype(np.uint8), source_affine, header=source_hdr)
        save_path = osp.join(output_dir, files[0].split("/")[-1])
        nib.save(seg_img, save_path)
        # load files