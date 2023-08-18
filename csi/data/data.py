import os
import random
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
from scipy import io as sio

from glob import glob

from box import Box


def generate_mask_3d_shift(mask_path):
    mask = sio.loadmat(mask_path)
    mask3d_shift = mask['mask_3d_shift']
    mask3d_shift = np.transpose(mask3d_shift, [2, 0, 1])
    mask3d_shift = torch.from_numpy(mask3d_shift).to(torch.float32)

    return mask3d_shift

def generate_mask_3d(mask_path, wave_len):
    mask = sio.loadmat(mask_path)
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, wave_len))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d).to(torch.float32)

    return mask3d

def LoadTraining(paths, debug=False):
    imgs = []
    scene_list = []
    for path in paths:
        scene_list.extend(glob(os.path.join(path, "*")))
    scene_list.sort()
    print('training sences:', len(scene_list))
    for scene_path in scene_list if not debug else scene_list[:20]:
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand'] / 65536.
        elif "img" in img_dict:
            img = img_dict['img'] / 65536.
        elif "hsi" in img_dict:
            img = img_dict['hsi']
        elif "HSI" in img_dict:
            img = img_dict['HSI']
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded.'.format(scene_path.split('/')[-1]))
    return imgs


def LoadVal(path_val):
    images = []
    data = {}
    scene_list = os.listdir(path_val)
    scene_list.sort()
    for i in range(len(scene_list)):
        scene_path = os.path.join(path_val, scene_list[i])
        img = sio.loadmat(scene_path)
        if "img_expand" in img:
            img = img['img_expand']
        elif "img" in img:
            img = img['img']
        elif "hsi" in img:
            img = img['hsi']
        images.append(img)
        
    data["hsi"] = images
    return data

def LoadTSATestMeas(path_test):
    measurements = []
    meas_paths = sorted(glob(os.path.join(path_test, '*.mat')))
    for meas_path in meas_paths:
        meas = sio.loadmat(meas_path)['meas_real']
        meas[meas < 0] = 0.0
        meas[meas > 1] = 1.0
        measurements.append(meas)
    
    measurements =  torch.from_numpy(np.stack(measurements, axis=0)).to(torch.float32)

    return measurements


class CSITrainDataset(Dataset):
    def __init__(
        self, 
        cfg,
        crop_size=(256, 256)
    ):
        super().__init__()
        self.cfg = cfg
        self.iteration = cfg.DATASETS.TRAIN.ITERATION
        self.crop_size = crop_size
        self.augment = cfg.DATASETS.TRAIN.AUGMENT
        self.imgs = LoadTraining(cfg.DATASETS.TRAIN.PATHS, cfg.DEBUG)
        if cfg.DATASETS.MASK_TYPE == "mask_3d": 
            self.mask = generate_mask_3d(cfg.DATASETS.TRAIN.MASK_PATH, cfg.DATASETS.WAVE_LENS)
        if cfg.DATASETS.MASK_TYPE == "mask_3d_shift":
            self.mask = generate_mask_3d_shift(cfg.DATASETS.TRAIN.MASK_PATH)
        _, self.mask_h, self.mask_w = self.mask.shape

        self.len_images = len(self.imgs)

    def __getitem__(self, idx):
        data = {}
        if self.augment:
            flag = random.randint(0, 1)
            if flag:
                index = np.random.randint(0, self.len_images-1)
                img = self.imgs[index]
                processed_image = np.zeros((self.crop_size[0], self.crop_size[1],  self.cfg.DATASETS.WAVE_LENS), dtype=np.float32)
        
                h, w, _ = img.shape
                if h > w:
                    img = np.transpose(img, (1, 0, 2))
                    h, w = w, h

                x_index = np.random.randint(0, h - self.crop_size[0])
                y_index = np.random.randint(0, w - self.crop_size[1])
                processed_image = img[x_index:x_index + self.crop_size[0], y_index:y_index + self.crop_size[1], :]

                processed_image = torch.from_numpy(np.transpose(processed_image, (2, 0, 1)))

                
                processed_image = augment_1(processed_image)
            else:
                processed_image = np.zeros((4, self.crop_size[0]//2, self.crop_size[1]//2, self.cfg.DATASETS.WAVE_LENS), dtype=np.float32)
                sample_list = np.random.randint(0, self.len_images, 4)
                for j in range(4):
                    h, w, _ = self.imgs[sample_list[j]].shape
                    x_index = np.random.randint(0, h-self.crop_size[0]//2)
                    y_index = np.random.randint(0, w-self.crop_size[1]//2)
                    processed_image[j] = self.imgs[sample_list[j]][x_index:x_index+self.crop_size[0]//2,y_index:y_index+self.crop_size[1]//2,:]

                processed_image = torch.from_numpy(np.transpose(processed_image, (0, 3, 1, 2)))  # [4,28,128,128]
                processed_image = augment_2(processed_image, self.crop_size)

        if self.cfg.DATASETS.TRAIN.RANDOM_MASK:
            mask_x_index = np.random.randint(0, self.mask_h - self.crop_size[0])
            if self.cfg.DATASETS.MASK_TYPE == "mask_3d":
                mask_y_index = np.random.randint(0, self.mask_w - self.crop_size[1])
                mask = self.mask[:, mask_x_index:mask_x_index + self.crop_size[0], mask_y_index:mask_y_index + self.crop_size[1]]
            else:
                mask_y_index = np.random.randint(0, self.mask_w - (self.crop_size[1] + (self.cfg.DATASETS.WAVE_LENS - 1) * self.cfg.DATASETS.STEP))
                mask = self.mask[:, mask_x_index:mask_x_index + self.crop_size[0], mask_y_index:mask_y_index + (self.crop_size[1] + (self.cfg.DATASETS.WAVE_LENS - 1) * self.cfg.DATASETS.STEP)]
        else:
            mask = self.mask
          
            

        data['hsi'] = processed_image
        data['mask'] = mask

        return data
    
    def __len__(self):
        return self.iteration



def shuffle_crop(train_data, batch_size, crop_size=256, augment=True):
    if augment:
        flag = random.randint(0, 1)
        if flag:
            index = np.random.choice(range(len(train_data)), batch_size)
            processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
            for i in range(batch_size):
                h, w, _ = train_data[index[i]].shape
                x_index = np.random.randint(0, h - crop_size)
                y_index = np.random.randint(0, w - crop_size)
                processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
            gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
            for i in range(gt_batch.shape[0]):
                gt_batch[i] = augment_1(gt_batch[i])
        else:
            gt_batch = []
            processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
            for i in range(batch_size):
                sample_list = np.random.randint(0, len(train_data), 4)
                for j in range(4):
                    h, w, _ = train_data[sample_list[j]].shape
                    x_index = np.random.randint(0, h-crop_size//2)
                    y_index = np.random.randint(0, w-crop_size//2)
                    processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
                generated_sample = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))  # [4,28,128,128]
                gt_batch.append(augment_2(generated_sample, crop_size=(crop_size, crop_size)))
            gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))

    return gt_batch



def augment_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x


def augment_2(generate_gt, crop_size):
    c, h, w = generate_gt.shape[1], crop_size[0], crop_size[1]
    divid_point_h = crop_size[0] // 2
    divid_point_w = crop_size[1] // 2
    output_img = torch.zeros(c,h,w)
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


def shift(inputs, step, nC):
    [nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[i,:,:] = torch.roll(inputs[i,:,:], shifts=step*i, dims=1)
    return inputs

def shift_back(inputs, step, nC):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [row, col] = inputs.shape
    output = torch.zeros(nC, row, col - (nC - 1) * step).float()
    for i in range(nC):
        output[i, :, :] = inputs[:, step * i:step * i + col - (nC - 1) * step]
    return output

def gen_meas_torch(inputs, Phi, step, wave_len, mask_type="mask_3d_shift"):
    data = {}
    [nC, H, W] = inputs.shape
    gt = torch.zeros(nC, H, W+step*(nC-1)).to(inputs.device)
    if mask_type == "mask_3d":
        gt[:,:,0:W] = Phi * inputs
        gt_shift = shift(gt, step=step, nC=wave_len)
    if mask_type == "mask_3d_shift":
        gt[:,:,0:W] = inputs
        gt_shift = shift(gt, step=step, nC=wave_len)
        gt_shift = Phi * gt_shift
    y = torch.sum(gt_shift, 0)
    meas = y / nC * 2
    H = shift_back(meas, step=step, nC=wave_len)
       
    data['Y'] = y
    data['H'] = H

    return data

def shift_batch(inputs, nC = 28, step=2):
    [B, nC, row, col] = inputs.shape
    outputs = torch.zeros((B, nC, row, col + (nC - 1) * step)).float().to(inputs.device)
    for i in range(nC):
        outputs[:, i, :,  step * i:step * i + col] = inputs[:, i, :, :]
    return outputs

def shift_back_batch(inputs, nC=28, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [B, row, col] = inputs.shape
    output = torch.zeros(B, nC, row, col - (nC - 1) * step).float().to(inputs.device)
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output


def gen_meas_torch_batch(inputs, Phi, step, wave_len, mask_type="mask_3d_shift", with_noise=False):
    data = {}
    [B, nC, H, W] = inputs.shape
    if mask_type == "mask_3d":
        modulated_hsi = Phi * inputs
        modulated_hsi_shift = shift_batch(modulated_hsi, step=step, nC=wave_len)
    if mask_type == "mask_3d_shift":
        hsi_shift = shift_batch(inputs, step=step, nC=wave_len)
        modulated_hsi_shift = Phi * hsi_shift

    y = torch.sum(modulated_hsi_shift, 1)

    if with_noise:
        input = y / nC * 2 * 1.2
        # input = y / (y.max() + 1e-7) * 0.9
        QE, bit = 0.4, 2048
        input_noise = torch.tensor(np.random.binomial((input.cpu().numpy() * bit / QE).astype(int), QE)).float()
        input = input_noise / bit
        input = input.to(inputs.device)
        H = shift_back_batch(input, step=step)
        data['Y'] = input
        data['H'] = H
    else:
        meas = y / nC * 2
        H = shift_back_batch(meas, step=step)
        data['Y'] = y
        data['H'] = H


    
       
    return data


if __name__ == '__main__':
    cfg = Box(
        {
            "DEBUG": True,
            "DATASETS": 
            {
                "STEP": 2,
                "WAVE_LENS": 28,
                "MASK_TYPE": "mask_3d_shift",
                "WITH_PAN": False,
                "TRAIN": 
                {
                    "PATHS" : ["../../../datasets/CSI/cave_1024_28"],
                    "ITERATION": 1000,
                    "MASK_PATH": "../../../datasets/CSI/TSA_simu_data/mask_3d_shift.mat",
                    "RANDOM_MASK": False,
                    "AUGMENT": True,
                },
                "VAL": 
                {
                    "PATH": "../../../datasets/CSI/TSA_simu_data/Truth",
                    "MASK_PATH": "../../../datasets/CSI/TSA_simu_data/mask_3d_shift.mat"
                }
            },
            "DATALOADER":
            {
                "BATCH_SIZE": 1
            }
        }
    )

    train_mask_path = "/Users/shawn/Documents/Code/datasets/CSI/TSA_simu_data/mask_3d_shift.mat"
    train_mask = generate_mask_3d_shift(train_mask_path)
    print("train_mask: ", train_mask.shape)

    dataset = CSITrainDataset(cfg)
    
    data = dataset[0]
    print("data['hsi'].shape: ", data['hsi'].shape)
    print("data['mask'].shape: ", data['mask'].shape)
    cv2.imshow("data['hsi'][0]", data['hsi'][0].numpy())
    cv2.imshow("data['mask'][0]", data['mask'][0].numpy())
    

        
    val_mask = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL.MASK_PATH)
    val_data = LoadVal(cfg.DATASETS.VAL.PATH)

    val_img = torch.from_numpy(val_data['hsi'][0]).permute(2, 0, 1).float()
    data = gen_meas_torch(val_img, val_mask, step=cfg.DATASETS.STEP, wave_len=cfg.DATASETS.WAVE_LENS, mask_type=cfg.DATASETS.MASK_TYPE)
    
    print("torch.mean(data['H']): ", torch.mean(data['H']))
    print("data['H'].shape: ", data['H'].shape)
    cv2.imshow("data['H'][0]", data['H'][0].numpy())
    cv2.imshow("data['H'][-1]", data['H'][-1].numpy())
    cv2.imshow("val img", val_img[0].numpy())



    # path_test = "/Users/shawn/Documents/Code/datasets/CSI/508_real_indoor/Measurements/"
    # pan_test = "/Users/shawn/Documents/Code/datasets/CSI/508_real_indoor/Panchromatic/"
    # measurements, pans = LoadTestMeas(path_test, pan_test)
    # print("test measurement shape: ", measurements.shape)
    # print("test pan shape: ", pans.shape)
    # meas = measurements[0]
    # cv2.imshow("test meas", meas.numpy())
    # shifted_meas = shift_back(meas, cfg.DATASETS.STEP)
    # print("shifted_meas.shape: ", shifted_meas.shape)
    # cv2.imshow("test real meas[0]", shifted_meas[0].numpy())
    # cv2.imshow("test real meas[-1]", shifted_meas[-1].numpy())
    # cv2.imshow("test real pan", pans[0].numpy())


    cv2.waitKey(0)