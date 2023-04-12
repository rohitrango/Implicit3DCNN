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

net = _C.NETWORK = CN()

def get_cfg_defaults():
    return _C.clone()