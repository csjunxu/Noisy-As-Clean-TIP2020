#!/usr/bin/env python
# coding: utf-8

# Code for **"Blind restoration of a JPEG-compressed image"** and **"Blind image denoising"** figures. Select `fname` below to switch between the two.

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
from skimage import color
# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(A, B)
from utils.denoising_utils import *
from PIL import Image
from tensorboardX import SummaryWriter
import random
random.seed(30)
np.random.seed(30)
import pickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


class GradientDifferenceLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientDifferenceLoss, self).__init__()

    # def forward(self, a):
    #     gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
    #     gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
    #     return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)
    #
    # return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)
    def forward(self, a, b, alpha=2.0):
        gradient_a_x = a[:, :, :, 1:] - a[:, :, :, :-1]
        gradient_a_y = a[:, :, 1:, :] - a[:, :, :-1, :]

        gradient_b_x = b[:, :, :, 1:] - b[:, :, :, :-1]
        gradient_b_y = b[:, :, 1:, :] - b[:, :, :-1, :]

        grad_x_alpha = torch.abs(gradient_a_x - gradient_b_x)**alpha
        grad_y_alpha = torch.abs(gradient_a_y - gradient_b_y)**alpha

        return torch.mean(grad_x_alpha) + torch.mean(grad_y_alpha)

def create_augmentations_for_not_sqaure(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    dtype = torch.cuda.FloatTensor
    flipped = np.rot90(np_image, 2, (1, 2)).copy()

    aug = [np_image.copy(), flipped.copy(),
           np.fliplr(np_image).copy(), np.fliplr(flipped).copy()]
    aug_torch = [np_to_torch(np_image.copy()).type(dtype), np_to_torch(flipped.copy()).type(dtype),
                 np_to_torch(np.fliplr(np_image).copy()).type(dtype), np_to_torch(np.flipud(np_image).copy()).type(dtype)]

    return aug, aug_torch


def MSE(x, y):
    return np.square(x - y).mean()


def save_image(name, image_np, output_path="E:/JunXu/NAC_TPAMI/results/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def closure():

    global i, net_input, psnr_max, psnr_noisy_max, files_name, noisy_np, TEST_PLAN, test_torch_list, test_np_list
    global TRAIN_PLAN, noisy_np_norm, sigma_now, final_ssim, files_name, test_np_list, test_torch_list
    global psnr_2_5_max, final_ssim_max_5, psnr_2_10_max, final_ssim_max_10, psnr_2_15_max, final_ssim_max_15
    global psnr_2_20_max, final_ssim_max_20, psnr_2_25_max, final_ssim_max_25, img_aug_n, SAVE_MODEL_FLAG, multilevel_noise_train
    out_effect_np = []
    img_noisy_pil=[]
    img_noisy_np =[]
    img_noisy_torch=[]
    img_noisy_noisy_np = []
    img_noisy_noisy_pil =[]
    img_noisy_noisy_torch = []
    if multilevel_noise_train:
        min_log = np.log([0.0001])
        sigma_s = min_log + np.random.rand(1) * (np.log([sigma_now]) - min_log)
        sigma_s = np.exp(sigma_s)
        noisy_np = noisy_np_norm * sigma_s
        noisy_torch = np_to_torch(noisy_np)
    else:
        noisy_np = noisy_np_norm * sigma_now
        noisy_torch = np_to_torch(noisy_np)

    for idx in range(len(net_input)):
        img_noisy_pil_, img_noisy_np_, img_noisy_noisy_pil_, img_noisy_noisy_np_ = \
            get_noisy_noisy_image_with_noise(noisy_np, img_aug_np[idx])
        img_noisy_torch_ = np_to_torch(img_noisy_np_).type(dtype)
        img_noisy_noisy_torch_ = np_to_torch(img_noisy_noisy_np_).type(dtype)

        img_noisy_pil.append(img_noisy_pil_)
        img_noisy_np.append(img_noisy_np_)
        img_noisy_torch.append(img_noisy_torch_)
        img_noisy_noisy_np.append(img_noisy_noisy_np_)
        img_noisy_noisy_pil.append(img_noisy_noisy_pil_)
        img_noisy_noisy_torch.append(img_noisy_noisy_torch_)
    for aug in range(len(img_noisy_torch)):
        noisy_torch = np_to_torch(img_noisy_noisy_np[aug] - img_noisy_np[aug])
        out = net(net_input[aug])
        total_loss = mse(out, noisy_torch.type(dtype))

        total_loss.backward()
        psrn_noisy = compare_psnr(np.clip(img_noisy_np[aug], 0, 1), (torch_to_np(net_input[aug]) - out.detach().cpu().numpy()[0]))
        do_i_learned_noise = img_noisy_noisy_np[aug] - out.detach().cpu().numpy()[0]
        mse_what_tf = MSE(noisy_np, do_i_learned_noise)

        if psnr_noisy_max == 0:
            psnr_noisy_max = psrn_noisy
        elif psnr_noisy_max < psrn_noisy:
            psnr_noisy_max = psrn_noisy

        if SAVE_DURING_TRAINING and i % save_every == 0:
            # output_dir
            out_test_np = torch_to_np(out)  # I +N1
            # out_test_name = f'{i}_test'
            # save_image(out_test_name, np.clip(out_test_np, 0, 1), output_path=output_dir)

        net.eval()
        with torch.no_grad():
            out_effect_np_ = torch_to_np(img_noisy_torch[aug] - net(img_noisy_torch[aug])) # torch_to_np(net(img_noisy_torch[aug]))
            out_effect_np.append(out_effect_np_)
            psnr_1 = compare_psnr(img_aug_np[aug], np.clip(out_effect_np_, 0, 1))
            test_do_i_learned_noise = img_noisy_np[aug] - out_effect_np_
            test_what_tf = MSE(noisy_np, test_do_i_learned_noise)
            writer.add_scalar('scalar_noisy_psnr', psrn_noisy, i)
            writer.add_scalar('scalar_test_psnr', psnr_1, i)
            if psnr_max == 0:
                psnr_max = psnr_1
            elif psnr_max < psnr_1:
                psnr_max = psnr_1


    if i % 10 == 0:
        SAVE_MODEL_FLAG = False
        net.eval()
        with torch.no_grad():
            real_test_effect_np = []
            for idxt in range(len(test_torch_list)):
                test_out_effect_np = torch_to_np(test_torch_list[idxt].type(dtype) - net(test_torch_list[idxt].type(dtype))) # torch_to_np(net(test_torch_list[idxt].type(dtype)))
                real_test_effect_np.append(test_out_effect_np)

        print('%s Iteration %05d lr: %f, Loss %f , PSNR_noisy: %f, PSNR_noisy_max: %f, noise mse: %f,'
              'test psnr: %f , test max psnr: %f, test noise mse: %f , current sigma: %f ' %
              (files_name, i, LR, total_loss.item(), psrn_noisy, psnr_noisy_max, mse_what_tf, psnr_1, psnr_max,
               test_what_tf, sigma_s * 255))


        # sigma = 5
        test_out_effect_np_5 = real_test_effect_np[0:4]
        test_out_effect_np_5[0] = test_out_effect_np_5[0].transpose(1, 2, 0)
        test_out_effect_np_5[1] = (np.rot90(test_out_effect_np_5[1], 2, (1, 2))).transpose(1, 2, 0)
        test_out_effect_np_5[2] = (np.fliplr(test_out_effect_np_5[2])).transpose(1, 2, 0)
        test_out_effect_np_5[3] = (np.fliplr(np.rot90(test_out_effect_np_5[3], 2, (1, 2)))).transpose(1, 2, 0)
        # final_reuslt = np.median(out_effect_np, 0)
        final_reuslt_5 = np.mean(test_out_effect_np_5, 0)

        psnr_2_5 = compare_psnr(torch_to_np(net_input[0]).transpose(1, 2, 0), np.clip(final_reuslt_5, 0, 1))
        final_ssim_5 = compare_ssim(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_5, 0, 1), data_range=1, multichannel=True)

        if psnr_2_5_max==0:
            psnr_2_5_max = psnr_2_5
            final_ssim_max_5 = final_ssim_5
            SAVE_MODEL_FLAG = True
        elif psnr_2_5_max< psnr_2_5:
            psnr_2_5_max = psnr_2_5
            final_ssim_max_5 = final_ssim_5
            SAVE_MODEL_FLAG = True
            tmp_name_p_5 = f'{files_name[:-4]}_{TEST_PLAN[0]}_{psnr_2_5:.2f}_final_{final_ssim_5:.4f}'
            save_image(tmp_name_p_5, np.clip(final_reuslt_5.transpose(2, 0, 1), 0, 1), output_path=output_dir)

        print('psnr 2_5: %f, psnr 2_5 max: %f, final ssim_5 : %f, final ssim max_5: %f'
              %(psnr_2_5, psnr_2_5_max, final_ssim_5, final_ssim_max_5))

        # sigma = 10
        test_out_effect_np_10 = real_test_effect_np[4:8]
        test_out_effect_np_10[0] = test_out_effect_np_10[0].transpose(1, 2, 0)
        test_out_effect_np_10[1] = (np.rot90(test_out_effect_np_10[1], 2, (1, 2))).transpose(1, 2, 0)
        test_out_effect_np_10[2] = (np.fliplr(test_out_effect_np_10[2])).transpose(1, 2, 0)
        test_out_effect_np_10[3] = (np.fliplr(np.rot90(test_out_effect_np_10[3], 2, (1, 2)))).transpose(1, 2, 0)
        final_reuslt_10 = np.mean(test_out_effect_np_10, 0)

        psnr_2_10 = compare_psnr(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_10, 0, 1))
        final_ssim_10 = compare_ssim(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_10, 0, 1), data_range=1,
                                     multichannel=True)
        if psnr_2_10_max == 0:
            psnr_2_10_max = psnr_2_10
            final_ssim_max_10 = final_ssim_10
            SAVE_MODEL_FLAG = True
        elif psnr_2_10_max < psnr_2_10:
            psnr_2_10_max = psnr_2_10
            final_ssim_max_10 = final_ssim_10
            SAVE_MODEL_FLAG = True
            tmp_name_p_10 = f'{files_name[:-4]}_{TEST_PLAN[1]}_{psnr_2_10:.2f}_final_{final_ssim_10:.4f}'
            save_image(tmp_name_p_10, np.clip(final_reuslt_10.transpose(2, 0, 1), 0, 1), output_path=output_dir)

        print('psnr 2_10: %f, psnr 2_10 max: %f, final ssim_10 : %f, final ssim max_10: %f'
              % (psnr_2_10, psnr_2_10_max, final_ssim_10, final_ssim_max_10))

        # sigma = 15
        test_out_effect_np_15 = real_test_effect_np[8:12]
        test_out_effect_np_15[0] = test_out_effect_np_15[0].transpose(1, 2, 0)
        test_out_effect_np_15[1] = (np.rot90(test_out_effect_np_15[1], 2, (1, 2))).transpose(1, 2, 0)
        test_out_effect_np_15[2] = (np.fliplr(test_out_effect_np_15[2])).transpose(1, 2, 0)
        test_out_effect_np_15[3] = (np.fliplr(np.rot90(test_out_effect_np_15[3], 2, (1, 2)))).transpose(1, 2, 0)
        # final_reuslt = np.median(out_effect_np, 0)
        final_reuslt_15 = np.mean(test_out_effect_np_15, 0)

        psnr_2_15 = compare_psnr(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_15, 0, 1))
        final_ssim_15 = compare_ssim(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_15, 0, 1), data_range=1,
                                     multichannel=True)
        if psnr_2_15_max == 0:
            psnr_2_15_max = psnr_2_15
            final_ssim_max_15 = final_ssim_15
            SAVE_MODEL_FLAG = True
        elif psnr_2_15_max < psnr_2_15:
            psnr_2_15_max = psnr_2_15
            final_ssim_max_15 = final_ssim_15
            SAVE_MODEL_FLAG = True
            tmp_name_p_15 = f'{files_name[:-4]}_{TEST_PLAN[2]}_{psnr_2_15:.2f}_final_{final_ssim_15:.4f}'
            save_image(tmp_name_p_15, np.clip(final_reuslt_15.transpose(2, 0, 1), 0, 1), output_path=output_dir)

        print('psnr 2_15: %f, psnr 2_15 max: %f, final ssim_15 : %f, final ssim max_15: %f'
              % (psnr_2_15, psnr_2_15_max, final_ssim_15, final_ssim_max_15))

        # sigma = 20
        test_out_effect_np_20 = real_test_effect_np[12:16]
        test_out_effect_np_20[0] = test_out_effect_np_20[0].transpose(1, 2, 0)
        test_out_effect_np_20[1] = (np.rot90(test_out_effect_np_20[1], 2, (1, 2))).transpose(1, 2, 0)
        test_out_effect_np_20[2] = (np.fliplr(test_out_effect_np_20[2])).transpose(1, 2, 0)
        test_out_effect_np_20[3] = (np.fliplr(np.rot90(test_out_effect_np_20[3], 2, (1, 2)))).transpose(1, 2, 0)
        final_reuslt_20 = np.mean(test_out_effect_np_20, 0)

        psnr_2_20 = compare_psnr(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_20, 0, 1))
        final_ssim_20 = compare_ssim(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_20, 0, 1), data_range=1,
                                     multichannel=True)
        if psnr_2_20_max == 0:
            psnr_2_20_max = psnr_2_20
            final_ssim_max_20 = final_ssim_20
            SAVE_MODEL_FLAG = True
        elif psnr_2_20_max < psnr_2_20:
            psnr_2_20_max = psnr_2_20
            final_ssim_max_20 = final_ssim_20
            SAVE_MODEL_FLAG = True
            tmp_name_p_20 = f'{files_name[:-4]}_{TEST_PLAN[3]}_{psnr_2_20:.2f}_final_{final_ssim_20:.4f}'
            save_image(tmp_name_p_20, np.clip(final_reuslt_20.transpose(2, 0, 1), 0, 1), output_path=output_dir)

        print('psnr 2_20: %f, psnr 2_20 max: %f, final ssim_20 : %f, final ssim max_20: %f'
              % (psnr_2_20, psnr_2_20_max, final_ssim_20, final_ssim_max_20))

        # sigma = 25
        test_out_effect_np_25 = real_test_effect_np[16:20]
        test_out_effect_np_25[0] = test_out_effect_np_25[0].transpose(1, 2, 0)
        test_out_effect_np_25[1] = (np.rot90(test_out_effect_np_25[1], 2, (1, 2))).transpose(1, 2, 0)
        test_out_effect_np_25[2] = (np.fliplr(test_out_effect_np_25[2])).transpose(1, 2, 0)
        test_out_effect_np_25[3] = (np.fliplr(np.rot90(test_out_effect_np_25[3], 2, (1, 2)))).transpose(1, 2, 0)
        final_reuslt_25 = np.mean(test_out_effect_np_25, 0)

        psnr_2_25 = compare_psnr(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_25, 0, 1))
        final_ssim_25 = compare_ssim(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt_25, 0, 1), data_range=1,
                                     multichannel=True)
        if psnr_2_25_max == 0:
            psnr_2_25_max = psnr_2_25
            final_ssim_max_25 = final_ssim_25
            SAVE_MODEL_FLAG = True
        elif psnr_2_25_max < psnr_2_25:
            psnr_2_25_max = psnr_2_25
            final_ssim_max_25 = final_ssim_25
            SAVE_MODEL_FLAG = True
            tmp_name_p_25 = f'{files_name[:-4]}_{TEST_PLAN[4]}_{psnr_2_25:.2f}_final_{final_ssim_25:.4f}'
            save_image(tmp_name_p_25, np.clip(final_reuslt_25.transpose(2, 0, 1), 0, 1), output_path=output_dir)

        print('psnr 2_25: %f, psnr 2_25 max: %f, final ssim_25 : %f, final ssim max_25: %f'
              % (psnr_2_25, psnr_2_25_max, final_ssim_25, final_ssim_max_25))

    i += 1

    return total_loss



imsize = -1
SAVE_DURING_TRAINING = True
save_every = 1
TRAIN_PLAN = [55/255.]
TEST_PLAN = [5/255., 10/255., 15/255., 20/255., 25/255.]
SAVE_MODEL_FLAG = False # inital as False and turn True during training
# img_root = 'data/denoising/BM3D_images'
# img_root = 'data/denoising/Set12_o'
img_root = 'data/denoising/BSD68/'


files = os.listdir(img_root)
psnr_record_img_name = []
psnr_record_max_test_psnr = []

psnr_2_5_max_record = []
final_ssim_max_record_5 = []

psnr_2_10_max_record = []
final_ssim_max_record_10 = []

psnr_2_15_max_record = []
final_ssim_max_record_15 = []

psnr_2_20_max_record = []
final_ssim_max_record_20 = []

psnr_2_25_max_record = []
final_ssim_max_record_25 = []
output_dir = 'E:/JunXu/NAC_TPAMI/results/bsd68_g_DnCNN_blind55_Aug_LR=0.001_Epoch=100/'
full_output_dir = Path(output_dir).resolve()
if not os.path.isdir(full_output_dir):
    os.mkdir(full_output_dir)
DATA_AUG = True
multilevel_noise_train = True

for index in range(len(files)):
    files_name = str(files[index])
    test_np_list = []
    test_torch_list = []
    if files_name.endswith('.png'):
        fname = os.path.join(img_root, files_name)
        writer = SummaryWriter()
        img_pil = Image.open(fname)
        img_np = pil_to_np(img_pil)
        noisy_np_norm = np.random.normal(0.0, 1.0, size=img_np.shape)
        input_depth = img_np.shape[0]
        pad = 'replication'  # ['zero', 'replication', 'none']
        OPT_OVER = 'net'  # 'net,input'
        LR = 0.001

        OPTIMIZER = 'myadam'  # 'LBFGS'
        show_every = 1
        exp_weight = 0.99
        num_iter_plan = [101] # previously 2001
        figsize = 4

        num_iter = num_iter_plan[0]
        sigma_now = TRAIN_PLAN[0]

        psnr_noisy_max = 0
        psnr_max = 0
        psnr_2_5_max = 0
        final_ssim_max_5 = 0

        psnr_2_10_max = 0
        final_ssim_max_10 = 0

        psnr_2_15_max = 0
        final_ssim_max_15 = 0

        psnr_2_20_max = 0
        final_ssim_max_20 = 0

        psnr_2_25_max = 0
        final_ssim_max_25 = 0
        if DATA_AUG:
            img_aug_np, img_aug_torch = create_augmentations_for_not_sqaure(img_np)
            # noisy_aug_np, noisy_aug_torch = create_augmentations(noisy_np)
            img_noisy_pil = []
            img_noisy_np = []
            img_noisy_torch = []

            img_noisy_noisy_pil = []
            img_noisy_noisy_np = []
            img_noisy_noisy_torch = []
            # noisy_np = np.random.normal(scale=sigma_, size=img_np.shape)
            # noisy_np, noisy_torch = create_augmentations(noisy_np_)

            for idx_t in range(len(TEST_PLAN)):
                noisy_t_np = TEST_PLAN[idx_t]* noisy_np_norm
                for idx_tt in range(len(img_aug_np)):
                    img_noisy_pil_t, img_noisy_np_t, img_noisy_noisy_pil_t, img_noisy_noisy_np_t = \
                        get_noisy_noisy_image_with_noise(noisy_t_np, img_aug_np[idx_tt])
                    test_np_list.append(img_noisy_np_t)
                    test_torch_list.append(np_to_torch(img_noisy_np_t))

            # del net
            net = DnCNN(input_depth).type(dtype)


            # Compute number of parameters
            s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
            print('Number of params: %d' % s)
            # Loss
            mse = torch.nn.MSELoss().type(dtype)
             # Optimize
            # net_input = img_noisy_noisy_torch
            net_input = img_aug_torch

            i = 0
            p = get_params(OPT_OVER, net, net_input)
            optimize_blind(net, OPTIMIZER, p, closure, LR, num_iter, SAVE_MODEL_FLAG, files_name, sigma_now*255, output_dir)
            # optimize(net, OPTIMIZER, p, closure, LR, num_iter, SAVE_MODEL_FLAG, output_dir)
            psnr_record_img_name.append(files_name)
            psnr_record_max_test_psnr.append(psnr_max)
            psnr_2_5_max_record.append(psnr_2_5_max)
            final_ssim_max_record_5.append(final_ssim_max_5)

            psnr_2_10_max_record.append(psnr_2_10_max)
            final_ssim_max_record_10.append(final_ssim_max_10)

            psnr_2_15_max_record.append(psnr_2_15_max)
            final_ssim_max_record_15.append(final_ssim_max_15)

            psnr_2_20_max_record.append(psnr_2_20_max)
            final_ssim_max_record_20.append(final_ssim_max_20)

            psnr_2_25_max_record.append(psnr_2_25_max)
            final_ssim_max_record_25.append(final_ssim_max_25)

with open('resnet_aug_l2_bsd68_g_blind_55.pickle','wb') as f:
    resnet_aug = {'psnr_record_img_name': psnr_record_img_name,
                  'psnr_record_max_test_psnr': psnr_record_max_test_psnr,
                  'psnr_2_5_max_record': psnr_2_5_max_record,
                  'final_ssim_max_record_5':final_ssim_max_record_5,
                  'psnr_2_10_max_record': psnr_2_10_max_record,
                  'final_ssim_max_record_10': final_ssim_max_record_10,
                  'psnr_2_15_max_record': psnr_2_15_max_record,
                  'final_ssim_max_record_15': final_ssim_max_record_15,
                  'psnr_2_20_max_record': psnr_2_20_max_record,
                  'final_ssim_max_record_20': final_ssim_max_record_20,
                  'psnr_2_25_max_record': psnr_2_25_max_record,
                  'final_ssim_max_record_25': final_ssim_max_record_25
                  }
    pickle.dump(resnet_aug, f)

print('Training finished now!')

