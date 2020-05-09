#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
from models import *
from pathlib import Path
import torch
import torch.optim
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from utils.denoising_utils import *
from PIL import Image
from tensorboardX import SummaryWriter
import random
np.random.seed(30)
import pickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    dtype = torch.cuda.FloatTensor
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:,::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(), np.rot90(flipped, 3, (1, 2)).copy()]
    aug_torch = [np_to_torch(np_image.copy()).type(dtype), np_to_torch(np.rot90(np_image, 1, (1, 2)).copy()).type(dtype),
                 np_to_torch(np.rot90(np_image, 2, (1, 2)).copy()).type(dtype), np_to_torch(np.rot90(np_image, 3, (1, 2)).copy()).type(dtype)]
    aug_torch += [np_to_torch(flipped.copy()).type(dtype), np_to_torch(np.rot90(flipped, 1, (1, 2)).copy()).type(dtype),
                  np_to_torch(np.rot90(flipped, 2, (1, 2)).copy()).type(dtype), np_to_torch(np.rot90(flipped, 3, (1, 2)).copy()).type(dtype)]

    return aug, aug_torch


def MSE(x, y):
    return np.square(x - y).mean()


def save_image(name, image_np, output_path="./results/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


# test
sigma = 25/255. # AWGN noise level
img_name = '09.png'
# change the image name or test sigma level as './results/model/set12_g_2/image name/sigma level/epoch_1000.npz'
model_root = os.path.join('./results/set12_g_2/model', img_name[:-4],str(sigma*255))
model_dir = os.path.join(model_root,'epoch_1000.npz')
test_img_root = './data/denoising/Set12/'
output_dir = './test_output/'
os.makedirs(output_dir, exist_ok=True)

test_img_pil = Image.open(os.path.join(test_img_root, img_name)).convert('L')
test_img_np = pil_to_np(test_img_pil)
noisy_np_test = np.load(os.path.join(model_root,'noise_matrix.npy'))


net = ResNet(test_img_np.shape[0], test_img_np.shape[0], 10, 64, 1).type(dtype)

s = sum([np.prod(list(p.size())) for p in net.parameters()]);
print('Number of params: %d' % s)

# load model
checkpoint = torch.load(model_dir)
net.load_state_dict(checkpoint['model_state_dict'])

net.eval()
with torch.no_grad():
    test_img_aug, _ = create_augmentations(test_img_np)
    test_out = []
    for idx, test_img_aug_ in enumerate(test_img_aug):
        test_noisy_img_torch = np_to_torch(test_img_aug_ + noisy_np_test).type(dtype)
        out_effect_np_ = torch_to_np(net(test_noisy_img_torch))
        test_out.append(out_effect_np_)

    test_out[0] = test_out[0].transpose(1, 2, 0)
    for aug in range(1, 8):
        if aug < 4:
            test_out[aug] = np.rot90(test_out[aug].transpose(1, 2, 0), 4 - aug)
        else:
            test_out[aug] = np.flipud(np.rot90(test_out[aug].transpose(1, 2, 0), 8 - aug))
    final_reuslt = np.mean(test_out, 0)

    psnr_2 = compare_psnr(test_img_np.transpose(1, 2, 0), np.clip(final_reuslt, 0, 1))
    final_ssim = compare_ssim(test_img_np.transpose(1, 2, 0), np.clip(final_reuslt, 0, 1), data_range=1,
                              multichannel=True)
    tmp_name_p = f'{img_name[:-4]}_{sigma*255}_{psnr_2:.2f}_final_{final_ssim:.4f}'
    save_image(tmp_name_p, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)
    print('%s , sigma = %f, psnr: %.2f, ssim: %.4f'%(img_name, sigma*255, psnr_2, final_ssim))


