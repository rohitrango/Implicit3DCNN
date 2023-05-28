from yacs.config import CfgNode as CN

_C = CN()
_C.EXP_NAME = ""     # can be used by the script
seg = _C.SEG = CN()  # segmentation config

## run segmentation config here
seg.WEIGHT_DICE = 1.0
seg.WEIGHT_CE = 0.0
seg.WEIGHT_FOCAL = 0.0
seg.FOCAL_GAMMA = 2.0

train = _C.TRAIN = CN()  # training config
train.EPOCHS = 300
train.BASE_LR = 0.001
train.LR_SCHEDULER = 'poly'
train.BATCH_SIZE = 4
train.NUM_WORKERS = 0
train.BRATS_SEGM_MODE = 'brats'   # choices are 'raw' for the format mentioned in the dataset, 'brats' for the format used in the brats challenge
train.LOGIT_TRANSFORM = 'sigmoid'
train.WEIGHT_DECAY = 0.0
train.OPTIMIZER = 'adam'

val = _C.VAL = CN()  # validation config
val.FOLD = 0        
val.MAX_FOLDS = 5   #
val.RANDOM_SHUFFLE = False
val.RANDOM_SHUFFLE_SEED = 498534  # scribbed away the keyboard for luck
val.EVAL_EVERY = 1
val.STOP_AT = 400

dataset = _C.DATASET = CN()  # dataset config
dataset.TRAIN_ENCODED_DIR = '/data/rohitrango/Implicit3DCNNTasks/brats2021_unimodal/'
dataset.TRAIN_SEG_DIR = '/data/rohitrango/BRATS2021/training/'

dataset.VAL_ENCODED_DIR = '/data/rohitrango/Implicit3DCNNTasks/brats2021_unimodal_val/'
dataset.VAL_SEG_DIR = '/data/rohitrango/BRATS2021/val/'
dataset.SCALE_RANGE = 0.2

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
enc.WINSORIZE_PERCENTILE = 100.0  # 100.0 means no winsorization

net = _C.NETWORK = CN()
net.NAME = "AbstractContextResNet"
net.BLOCK_CHANNELS = [16, 16, 16, 8]
net.BLOCK_NUM_LAYERS = [2, 2, 2, 2]
net.ACTIVATION = 'LeakyReLU'
net.ACTIVATION_PARAM = 0.05
net.INPUT_CHANNELS = 4
net.OUTPUT_CHANNELS = 4
net.USE_LAYERNORM = False

def get_cfg_defaults():
    return _C.clone()
