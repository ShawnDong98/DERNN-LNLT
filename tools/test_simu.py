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

mask = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL.MASK_PATH).to(device)

val_datas = LoadVal(cfg.DATASETS.VAL.PATH)

model = eval(cfg.MODEL.DENOISER.TYPE)(cfg).to(device)

ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

if cfg.PRETRAINED_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.PRETRAINED_CKPT_PATH}")
    save_state = torch.load(cfg.PRETRAINED_CKPT_PATH, map_location=device)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])



def eval():
    psnr_list, ssim_list, sam_list = [], [], []
    val_H = []
    val_Y = []
    val_gt = []
    for val_label in val_datas['hsi']:
        val_label = torch.from_numpy(val_label).permute(2, 0, 1).to(device).float()
        YH = gen_meas_torch(val_label, mask, step=cfg.DATASETS.STEP, wave_len=cfg.DATASETS.WAVE_LENS, mask_type=cfg.DATASETS.MASK_TYPE)
        val_H.append(YH['H'].to(device))
        val_Y.append(YH['Y'].to(device))
        val_gt.append(val_label)
    val_gt = torch.stack(val_gt)
    val_H = torch.stack(val_H)
    val_Y = torch.stack(val_Y)
    data = {}
    data['hsi'] = val_gt
    data['H'] = val_H
    B, _, _, _ = val_H.shape
    data['mask'] = mask.unsqueeze(0).tile((B, 1, 1, 1))
    data['Y'] = val_Y

    model.eval()
    begin = time.time()
    with torch.no_grad():
        with ema.average_parameters():
            out = model(data)
            model_out = out

    for i in range(len(model_out)):
        psnr_val = torch_psnr(model_out[i, :, :, :], val_gt[i, :, :, :])
        ssim_val = torch_ssim(model_out[i, :, :, :], val_gt[i, :, :, :])
        sam_val = sam(model_out[i, :, :, :].permute(1, 2, 0).cpu().numpy(), val_gt[i, :, :, :].permute(1, 2, 0).cpu().numpy())
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        sam_list.append(sam_val)

    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(val_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    sam_mean = np.mean(np.asarray(sam_list))

    end = time.time()

    print('===> testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, time: {:.2f}'
                .format(psnr_mean, ssim_mean, sam_mean, (end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, sam_list, psnr_mean, ssim_mean, sam_mean



def main():
    (pred, truth, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = eval()
    sio.savemat("./results/dernn_lnlt_9stg_star_simu.mat", {"pred": pred, "truth" : truth})



if __name__ == "__main__":
    main()