import os
import sys
import time

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torch_ema import ExponentialMovingAverage

import cv2
import numpy as np
from scipy import io as sio
from tqdm import tqdm

from csi.config import get_cfg
from csi.engine import default_argument_parser, default_setup
from csi.data import CSITrainDataset, LoadVal, LoadTSATestMeas, shift_back_batch, generate_mask_3d, generate_mask_3d_shift, gen_meas_torch
from csi.architectures import DERNN_LNLT
from csi.utils.schedulers import get_cosine_schedule_with_warmup
from csi.losses import CharbonnierLoss, TVLoss
from csi.metrics import torch_psnr, torch_ssim, sam
from csi.utils.utils import checkpoint

args = default_argument_parser().parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask_test = generate_mask_3d_shift(mask_path=cfg.DATASETS.TEST.MASK_PATH).to(device)

test_meas = LoadTSATestMeas(cfg.DATASETS.TEST.PATH).to(device)

model = eval(cfg.MODEL.DENOISER.TYPE)(cfg).to(device)

ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

if cfg.PRETRAINED_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.PRETRAINED_CKPT_PATH}")
    save_state = torch.load(cfg.PRETRAINED_CKPT_PATH, map_location=device)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])

def test(test_meas, name="test_a"):
    model.eval()
    model_out = []
    data = {}
    data['Y'] = test_meas / test_meas.max() * 0.8

    B, _, _ = test_meas.shape
    data['mask'] = mask_test.unsqueeze(0).tile((B, 1, 1, 1))
    data['H'] = shift_back_batch(test_meas, step=cfg.DATASETS.STEP, nC=cfg.DATASETS.WAVE_LENS)
        
    with torch.no_grad():
        with ema.average_parameters():
            model_out = model(data)

    
    for i in range(B):
        out_plot = F.interpolate(model_out[i:i+1, :, :, :], size=(128, 128))
        if name == "TSA": out_plot = torch.flip(out_plot, dims=(2, 3))
       
        
    model_out = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()

    return model_out


def main():
    test_out = test(test_meas, "TSA")
    sio.savemat("./results/dernn_lnlt_5stg_real.mat", {"pred": test_out})
    

if __name__ == "__main__":
    main()