'''
Script to take the BRATS dataset images and perform histogram equalization on non-zero voxels
'''
import nibabel as nib
from glob import glob
import argparse
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import os
import skimage

def equalization(pair):
    ''' given pair of input and output file, perform histogram equalization on the input file and save it to the output file '''
    input_file, output_file = pair
    if 'seg' in input_file:
        # segmentation file, just create a symlink
        if not os.path.exists(output_file):
            os.symlink(input_file, output_file)
    else:
        # load the image 
        image_nii = nib.load(input_file)
        data = image_nii.get_fdata()
        # get min and max values
        min_val = np.min(data[data > 0])
        max_val = np.max(data)
        num_bins = int(max_val - min_val + 1)
        data[data < 0] = min_val
        #data_pos = data[data > 0]
        x, y, z = np.where(data > 0)
        data_pos = data[x, y, z]
        # perform histogram equalization
        data_norm = skimage.exposure.equalize_hist(data_pos, nbins=num_bins)
        new_data = np.zeros_like(data)
        new_data[x, y, z] = data_norm
        # save this to the output file
        nib.save(nib.Nifti1Image(new_data, image_nii.affine, image_nii.header), output_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform histogram equalization on BRATS dataset')
    parser.add_argument('--input_dir', type=str, default='/data/rohitrango/BRATS2021/', help='Path to the BRATS directory')
    parser.add_argument('--output_dir', type=str, default='/data/rohitrango/BRATS2021_CE/', help='Path to save')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers')
    args = parser.parse_args()

    # get the list of all the files
    input_files = glob(args.input_dir + 'training/*/*') + glob(args.input_dir + 'val/*/*')
    output_files = [x.replace(args.input_dir, args.output_dir) for x in input_files]
    # make dirs
    for o in output_files:
        os.makedirs(os.path.dirname(o), exist_ok=True)
    # now run the equalization
    with Pool(args.num_workers) as p:
        list(tqdm(p.imap(equalization, zip(input_files, output_files)), total=len(input_files)))
