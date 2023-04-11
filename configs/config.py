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

val = _C.VAL = CN()  # validation config
val.FOLD = 0        

def get_cfg_defaults():
    return _C.clone()