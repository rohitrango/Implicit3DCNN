CUDA_VISIBLE_DEVICES=4 python scripts/train_brats_segmentation.py --exp_name resnetv1.2-fold0-focal0-ln --cfg_file configs/brats_v1.2.yaml SEG.WEIGHT_FOCAL 0.0 NETWORK.USE_LAYERNORM True &
CUDA_VISIBLE_DEVICES=5 python scripts/train_brats_segmentation.py --exp_name resnetv1.2-fold0-focal0.1-ln --cfg_file configs/brats_v1.2.yaml SEG.WEIGHT_FOCAL 0.1 NETWORK.USE_LAYERNORM True &
CUDA_VISIBLE_DEVICES=6 python scripts/train_brats_segmentation.py --exp_name resnetv1.2-fold0-focal0.25-ln --cfg_file configs/brats_v1.2.yaml SEG.WEIGHT_FOCAL 0.25 NETWORK.USE_LAYERNORM True &
CUDA_VISIBLE_DEVICES=7 python scripts/train_brats_segmentation.py --exp_name resnetv1.2-fold0-focal1.0-ln --cfg_file configs/brats_v1.2.yaml SEG.WEIGHT_FOCAL 1.0 NETWORK.USE_LAYERNORM True &
