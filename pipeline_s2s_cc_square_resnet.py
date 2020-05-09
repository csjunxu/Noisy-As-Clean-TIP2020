#!/usr/bin/env python
# coding: utf-8

# Code for **"Blind restoration of a JPEG-compressed image"** and **"Blind image denoising"** figures. Select `fname` below to switch between the two.

from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
from models import *

import torch
import torch.optim
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from skimage import color
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(A, B)
from utils.denoising_utils import *
from PIL import Image
from tensorboardX import SummaryWriter
import scipy.io
from SomeISP_operator_python.ISP_implement_fuction import ISP


np.random.seed(30)
import pickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def generate_reallike_noise_from_rgb(img_np_rgb, sigma_s='RAN', sigma_c='RAN'):
    # follow the pipline
    isp = ISP()
    gt, noise, sigma_s_r, sigma_c_r = isp.cbdnet_noise_generate_srgb(img_np_rgb.transpose(1,2,0), sigma_s, sigma_c)

    return gt.transpose(2, 0, 1), noise.transpose(2, 0, 1), sigma_s_r, sigma_c_r


def add_PG_noise(img, sigma_s='RAN', sigma_c='RAN'):
    min_log = np.log([0.0001])
    if sigma_s == 'RAN':
        sigma_s = min_log + np.random.rand(1) * (np.log([0.16]) - min_log)
        sigma_s = np.exp(sigma_s)

    if sigma_c == 'RAN':
        sigma_c = min_log + np.random.rand(1) * (np.log([0.06]) - min_log)
        sigma_c = np.exp(sigma_c)

    # add noise
    print('Adding Noise: sigma_s='+str(sigma_s*255)+' sigma_c='+str(sigma_c*255))
    noisy_img = img + \
                np.random.normal(0.0, 1.0, img.shape) * (sigma_s * img) + \
                np.random.normal(0.0, 1.0, img.shape) * sigma_c
    # noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img


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


def save_image(name, image_np, output_path="E:/JunXu/NAC_TPAMI/results/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def closure():

    global i, net_input, psnr_1_max, psnr_noisy_max, files_name, psnr_2_max, gt_np, output_dir1
    global final_ssim, final_ssim_max, files_name, ssim_1_max
    global psnr_2, psnr_1, ssim_1, gt_aug_np, gt_aug_torch
    out_effect_np = []
    for aug in range(len(img_noisy_torch)):
        out = net(net_input[aug])
        total_loss = mse(out, img_noisy_torch[aug])
        # total_loss = mse(out, img_noisy_torch[aug])+ gd_loss(out, img_noisy_torch[aug])
        total_loss.backward()
        # psrn_noisy = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        psrn_noisy = compare_psnr(img_noisy_np[aug], out.detach().cpu().numpy()[0])

        if psnr_noisy_max == 0:
            psnr_noisy_max = psrn_noisy
        elif psnr_noisy_max < psrn_noisy:
            psnr_noisy_max = psrn_noisy

        if SAVE_DURING_TRAINING and i % save_every == 0:
            # output_dir
            # out_test_np = torch_to_np(out)  # I +N1
            # out_test_name = f'{i}_test'
            # save_image(out_test_name, np.clip(out_test_np, 0, 1), output_path=output_dir)

            net.eval()
            with torch.no_grad():
                out_effect_np_ = np.clip(torch_to_np(net(img_noisy_torch[aug])), 0, 1)
                out_effect_np.append(out_effect_np_)
                psnr_1 = compare_psnr(gt_aug_np[aug], out_effect_np_)
                ssim_1 = compare_ssim(gt_aug_np[aug].transpose(1, 2, 0), out_effect_np_.transpose(1, 2, 0), data_range=1, multichannel=True)

                # out_effect_name = f'{i}_effect_{psnr_1:.2f}'
                # save_image(out_effect_name, np.clip(out_effect_np_, 0,1), output_path=output_dir)
                writer.add_scalar('scalar_noisy_psnr', psrn_noisy, i)
                writer.add_scalar('scalar_test_psnr', psnr_1, i)
                if psnr_1_max == 0:
                    psnr_1_max = psnr_1
                elif psnr_1_max < psnr_1:
                    psnr_1_max = psnr_1
                    tmp_name_p = f'{files_name[:-4]}_{psnr_1:.2f}_test_{ssim_1:.4f}'
                    save_image(tmp_name_p,out_effect_np_, output_path=output_dir1)

                if ssim_1_max == 0:
                    ssim_1_max = ssim_1
                elif ssim_1_max<ssim_1:
                    ssim_1_max = ssim_1
                    tmp_name_p = f'{files_name[:-4]}_{ssim_1:.4f}_test_{psnr_1:.2f}'
                    save_image(tmp_name_p, out_effect_np_, output_path=output_dir1)

                print('%s Iteration %05d lr: %f, Loss %f , PSNR_noisy: %f, PSNR_noisy_max: %f, '
                      'test psnr: %f , test max psnr: %f ' %
                      (files_name, i, LR, total_loss.item(), psrn_noisy, psnr_noisy_max,
                       psnr_1, psnr_1_max, ))


    if i % 10 == 0:
        out_effect_np[0] = out_effect_np[0].transpose(1, 2, 0)
        out_effect_np[8] = out_effect_np[8].transpose(1, 2, 0)
        for aug in range(1, 8):
            if aug < 4:
               out_effect_np[aug] = np.rot90(out_effect_np[aug].transpose(1, 2, 0), 4-aug)
            else:
                out_effect_np[aug] = np.flipud(np.rot90(out_effect_np[aug].transpose(1, 2, 0), 8-aug))
        # final_reuslt = np.median(out_effect_np, 0)
        for aug in range(9, 16):
            if aug-8 < 4:
               out_effect_np[aug] = np.rot90(out_effect_np[aug].transpose(1, 2, 0), 4-(aug-8))
            else:
                out_effect_np[aug] = np.flipud(np.rot90(out_effect_np[aug].transpose(1, 2, 0), 8-(aug-8)))

        final_reuslt = np.mean(out_effect_np, 0)

        psnr_2 = compare_psnr(gt_np.transpose(1, 2, 0), final_reuslt)
        final_ssim = compare_ssim(gt_np.transpose(1, 2, 0), final_reuslt, data_range=1, multichannel=True)

        # out_test_name = f'{i}_test'
        # save_image(out_test_name, np.clip(final_reuslt, 0, 1), output_path=output_dir)
        if psnr_2_max==0:
            psnr_2_max = psnr_2
        elif psnr_2_max < psnr_2:
            psnr_2_max = psnr_2
            tmp_name_p = f'{files_name[:-4]}_{psnr_2:.2f}_final_{final_ssim:.4f}'
            save_image(tmp_name_p, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)
        if final_ssim_max==0:
            final_ssim_max = final_ssim
        elif final_ssim_max<final_ssim:
            final_ssim_max = final_ssim
            tmp_name = f'{files_name[:-4]}_{final_ssim:.4f}_final_{psnr_2:.2f}'
            save_image(tmp_name, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)

        print('psnr 2: %f, psnr 2 max: %f, final ssim : %f, final ssim max: %f'
              %(psnr_2, psnr_2_max, final_ssim, final_ssim_max))
        writer.add_scalar('final_test_psnr', psnr_2, i)
        writer.add_scalar('final_max_test_psnr', psnr_2_max, i)

    i += 1

    return total_loss


imsize = -1
SAVE_DURING_TRAINING = True
save_every = 1

## denoising
# img_root = 'data/denoising/BM3D_images'
img_root = 'data/denoising/cc_real/'
gt_root = 'data/denoising/cc_mean/'
# cc dataset
# mat_dir = '/raid/huangyuan/real_image_noise_dataset/Nikon_D600/ISO_3200/'
# mat = scipy.io.loadmat(os.path.join(mat_dir, 'C_1.mat'))

files = os.listdir(img_root)
psnr_record_img_name = []
psnr_record_max_test_psnr = []
ssim_record_max_test_psnr = []
psnr_2_max_record = []
final_ssim_max_record = []
current_sigma_s_record = []
current_sigma_c_record = []
output_dir = 'E:/JunXu/NAC_TPAMI/results/cc15_dncnn_s2s/'
output_dir1 = 'E:/JunXu/NAC_TPAMI/results/cc15_dncnn_s2s_1/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)
DATA_AUG = True


for index in range(0, len(files)):
    files_name = str(files[index])
    if files_name.endswith('.png'):
        fname = os.path.join(img_root, files_name)
        gt_dir = os.path.join(gt_root, files_name[:-8]+'mean.png')
        writer = SummaryWriter()
        img_pil = Image.open(fname)
        gt_pil = Image.open(gt_dir)
        img_np = pil_to_np(img_pil)
        gt_np = pil_to_np(gt_pil)
        # noisy_np_norm = np.random.normal(scale=1 / 255., size=img_np.shape)
        input_depth = img_np.shape[0]
        pad = 'replication'  # ['zero', 'replication', 'none']
        OPT_OVER = 'net'  # 'net,input'
        LR = 0.001

        OPTIMIZER = 'myadam'  # 'LBFGS'
        show_every = 1
        exp_weight = 0.99
        num_iter_plan = 1001
        figsize = 4

        num_iter = num_iter_plan
        psnr_2_max = 0
        psnr_noisy_max = 0
        ssim_1_max = 0
        psnr_1_max = 0
        final_ssim_max = 0
        if DATA_AUG:
            gt_aug_np, gt_aug_torch = create_augmentations(gt_np)
            gt_aug_np.extend(gt_aug_np)
            gt_aug_torch.extend(gt_aug_torch)
            img_aug_np, img_aug_torch = create_augmentations(img_np)
            img_aug_np.extend(img_aug_np)
            img_aug_torch.extend(img_aug_torch)
            img_noisy_np = img_aug_np
            img_noisy_torch = img_aug_torch
            img_noisy_noisy_np = img_aug_np
            img_noisy_noisy_torch = img_aug_torch


        net = ResNet(input_depth, img_np.shape[0], 10, 64, 1).type(dtype)

        s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
        print('Number of params: %d' % s)
    # Loss
        mse = torch.nn.MSELoss().type(dtype)

     # Optimize
        net_input = img_noisy_noisy_torch
        i = 0
        p = get_params(OPT_OVER, net, net_input)
        optimize(OPTIMIZER, p, closure, LR, num_iter)
        psnr_record_img_name.append(files_name)
        psnr_record_max_test_psnr.append(psnr_1_max)
        ssim_record_max_test_psnr.append(ssim_1_max)
        psnr_2_max_record.append(psnr_2_max)
        final_ssim_max_record.append(final_ssim_max)


with open('cc_s2s_resnet_realnoise_l2.pickle','wb') as f:
    resnet_aug = {'psnr_record_img_name': psnr_record_img_name,
                  'psnr_record_max_test_psnr': psnr_record_max_test_psnr,
                  'ssim_record_max_test_psnr': ssim_record_max_test_psnr,
                  'psnr_2_max_record': psnr_2_max_record,
                  'final_ssim_max_record':final_ssim_max_record}
    pickle.dump(resnet_aug, f)

print('I am finish training now!')
        # psnr_record_max_test_psnr.append(psnr_max)
        # out_np = torch_to_np(net(net_input))
        # q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)

