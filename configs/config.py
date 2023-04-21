from yacs.config import CfgNode as CN

_C = CN()
seg = _C.SEG = CN()  # segmentation config

## run segmentation config here
seg.WEIGHT_DICE = 1.0
seg.WEIGHT_CE = 0.2

train = _C.TRAIN = CN()  # training config
train.EPOCHS = 300
train.BASE_LR = 0.001
train.LR_SCHEDULER = 'poly'
train.BATCH_SIZE = 4
train.NUM_WORKERS = 4
train.BRATS_SEGM_MODE = 'brats'   # choices are 'raw' for the format mentioned in the dataset, 'brats' for the format used in the brats challenge
train.LOGIT_TRANSFORM = 'sigmoid'

val = _C.VAL = CN()  # validation config
val.FOLD = 0        

dataset = _C.DATASET = CN()  # dataset config
dataset.TRAIN_ENCODED_DIR = '/data/Implicit3DCNNTasks/brats2021/'
dataset.TRAIN_SEG_DIR = '/data/BRATS2021/training/'

## encoder parameters
enc = _C.ENCODE = CN()
enc.MULTIMODAL = True
enc.MLABEL = 0   # can be 0, 1, 2, 3
enc.NUM_EPOCHS_STAGE1 = 20
enc.NUM_EPOCHS_STAGE2 = 500
enc.STAGE1_TRAIN_IMAGES = 250
enc.NUM_POINTS = 100000
enc.SCORE_FN = "uniform"   # normalize using z-score or uniform
# encoder parameters
enc.LEVEL_DIM = 4
enc.DESIRED_RESOLUTION = 196

net = _C.NETWORK = CN()

def get_cfg_defaults():
    return _C.clone()