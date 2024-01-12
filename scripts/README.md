# README Scripts

This document contains descriptions of the scripts in this directory, since I was bad with naming some of these files.

### BRATS dataset

1. `encode_brats.py`: Script to encode the BRATS dataset into the InstantNGP format. This has two parts:
    - In stage1, both the encoders and decoder is trained. Only a small representative number of volumes are trained.
    - In stage2, only the encoders are trained one by one.

    This uses the `BRATS2021Dataset` dataset to load pairs of `xyz` coordinates and intensities.
    This script is the entrypoint to learn any other tasks like segmentation on the BRATS dataset.

2. `train_brats_segmentation.py`: Script to train the InstantNGP on brats.