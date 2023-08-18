# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# NOTE: given the new config system
# (https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html),
# we will stop adding new functionalities to default CfgNode.

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.IMG_SIZE = [256, 256]
_C.DEBUG = False
_C.OUTPUT_DIR = ""
_C.SEED = 3407
_C.DETERMINISTIC = True
_C.RESUME_CKPT_PATH = None
_C.PRETRAINED_CKPT_PATH = None

_C.DATASETS = CN()
_C.DATASETS.STEP = 2
_C.DATASETS.WAVE_LENS = 28
_C.DATASETS.MASK_TYPE = "mask_3d"

_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.MASK_PATH = ""
_C.DATASETS.TRAIN.RANDOM_MASK = True 
_C.DATASETS.TRAIN.PATHS = []
_C.DATASETS.TRAIN.ITERATION = 1000
_C.DATASETS.TRAIN.CROP_SIZE = [256, 256]
_C.DATASETS.TRAIN.AUGMENT = True
_C.DATASETS.TRAIN.WITH_NOISE = True

_C.DATASETS.VAL = CN()
_C.DATASETS.VAL.MASK_PATH = ""
_C.DATASETS.VAL.PATH = ""


_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.PATH = None
_C.DATASETS.TEST.MASK_PATH = None

# DATALOADER
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 4
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.PIN_MEMORY = False


# MODEL
_C.MODEL = CN()
_C.MODEL.DENOISER = CN()
_C.MODEL.DENOISER.TYPE = "DERNN_LNLT"

_C.MODEL.DENOISER.DERNN_LNLT = CN()
_C.MODEL.DENOISER.DERNN_LNLT.LOCAL = True
_C.MODEL.DENOISER.DERNN_LNLT.NON_LOCAL = True
_C.MODEL.DENOISER.DERNN_LNLT.IN_DIM = 28
_C.MODEL.DENOISER.DERNN_LNLT.OUT_DIM = 28
_C.MODEL.DENOISER.DERNN_LNLT.DIM = 28
_C.MODEL.DENOISER.DERNN_LNLT.WINDOW_SIZE = [8, 8]
_C.MODEL.DENOISER.DERNN_LNLT.WINDOW_NUM = [8, 8]
_C.MODEL.DENOISER.DERNN_LNLT.NUM_BLOCKS = [1, 1, 1, 1, 1]
_C.MODEL.DENOISER.DERNN_LNLT.FFN_NAME = "Gated_Dconv_FeedForward"
_C.MODEL.DENOISER.DERNN_LNLT.FFN_EXPAND = 2.66
_C.MODEL.DENOISER.DERNN_LNLT.LAYERNORM_TYPE = "WithBias"
_C.MODEL.DENOISER.DERNN_LNLT.STAGE = 5
_C.MODEL.DENOISER.DERNN_LNLT.SHARE_PARAMS = True
_C.MODEL.DENOISER.DERNN_LNLT.WITH_DL = True
_C.MODEL.DENOISER.DERNN_LNLT.WITH_MU = True
_C.MODEL.DENOISER.DERNN_LNLT.WITH_NOISE_LEVEL = True




# EMA
_C.MODEL.EMA = CN()
_C.MODEL.EMA.ENABLE = True
_C.MODEL.EMA.DECAY = 0.999

# OPTIMIZER
_C.OPTIMIZER = CN()
_C.OPTIMIZER.MAX_EPOCH = 300
_C.OPTIMIZER.LR = 2e-4
_C.OPTIMIZER.GRAD_CLIP = True

_C.LOSSES = CN()
_C.LOSSES.L1_LOSS = True
_C.LOSSES.TV_LOSS = False
