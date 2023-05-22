''' 
this script runs over all the files in the PATHS provided, and computes the min and max intensities for each modality
'''
import nibabel as nib
from glob import glob
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    PATHS = ['/data/rohitrango/BRATS2021/training/', '/data/rohitrango/BRATS2021/val/']
    # minmax_data = dict(flair=[0, 0], t1=[0, 0], t1ce=[0, 0], t2=[0, 0])
    minmax_data = [[0, 0], [0, 0], [0, 0], [0, 0]]
    for path in PATHS:
        dirs = sorted(glob(path + 'BraTS2021_*'))
        pbar = tqdm(dirs)
        for dir in pbar:
            files = sorted(glob(dir + '/*.nii.gz'))
            files = [f for f in files if 'seg' not in f]
            if len(files) != 4:
                print('ERROR: ', dir)
                continue
            for i, f in enumerate(files):
                data = nib.load(f).get_fdata()
                clampval = np.percentile(data, 99)
                data[data > clampval] = clampval
                minmax_data[i][0] = min(minmax_data[i][0], data.min())
                minmax_data[i][1] = max(minmax_data[i][1], data.max())
            pbar.set_description(f"minmax_data: {minmax_data}")
    print(minmax_data)