'''
author: rohitrango

Script to generate toy examples where the volumes contain a ball at some locations.

- The segmentation task is to segment all the balls that have another ball at diagonally opposite location. 
- The classification task is to compute the XOR of all the location of the balls.
    The volume is divided into eight octants and the `location` of each octant is `xyz` in binary.
    So the first octant is numbered `000`(=0) and the last octant is numbered `111` (=7).
'''
import numpy as np
from scipy.ndimage import gaussian_filter
from os import path as osp
from tqdm import tqdm

save_path = "/data/Implicit3DCNNTasks/toyexample1"

if __name__ == '__main__':
    volume_size = 256
    sigma = 30

    # get labels file
    labels_file = open(osp.join(save_path, "labels.txt"), "w")
    for i in tqdm(range(300)):
        volume = np.zeros((volume_size, volume_size, volume_size))       
        loc = np.random.randint(4)  # we will multiply with 2 to get 8 different choices
        # find radius
        coord = np.random.randint(10, size=(3,))-5 + np.array([volume_size//4, volume_size//4, volume_size//4])
        if loc > 0:
            coord[loc-1] += volume_size//2
        # put the ball there
        x, y, z = coord
        volume[x, y, z] = (np.sqrt(2*np.pi)*sigma)**3
        mirror = np.random.randint(2)
        if mirror:
            coordnew = volume_size - coord
            x, y, z = coordnew
            volume[x, y, z] = (np.sqrt(2*np.pi)*sigma)**3
        # smear the ball
        volume = gaussian_filter(volume, sigma=sigma).astype(np.float32)
        # volume = volume / volume.max()
        segm = (volume >= 0.5).astype(np.uint8)
        if not mirror:
            segm = segm*0
        # classification
        label = loc + 4*mirror
        # save them
        np.save(osp.join(save_path, f"img_{i}.npy"), volume)
        np.save(osp.join(save_path, f"seg_{i}.npy"), segm)
        labels_file.write("{}\n".format(label))
    labels_file.close()
