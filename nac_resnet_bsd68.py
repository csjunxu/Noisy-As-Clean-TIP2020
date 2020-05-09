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

# import cv2


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
    p.save(output_path + "{}.jpg".format(name))


def closure():

    global i, net_input, psnr_max, psnr_noisy_max, files_name, psnr_2_max, noisy_np
    global TRAIN_PLAN, noisy_np_norm, sigma_now, final_ssim, final_ssim_max, files_name
    global psnr_curve_max_record, ssim_curve_max_record, training_loss_record

    out_effect_np = []
    if DATA_AUG:
        for aug in range(len(img_noisy_torch)):
            out = net(net_input[aug])


            if i % 200 == 0:
                # save x+n, x+n+n, and estimated x+n
                save_image("004_x+n1", np.clip(img_noisy_np[0], 0, 1))
                save_image("004_x+n1+n2", np.clip(img_noisy_noisy_np[0], 0, 1))

                if aug == 0:
                    save_image("004_x+n1_pred", out.detach().cpu().numpy()[0])
                    # cv2.imwrite("004_x+n1.jpg", cv2.merge(out[0] * 255))

            total_loss = mse(out, img_noisy_torch[aug])

            total_loss.backward()
            psrn_noisy = compare_psnr(np.clip(img_noisy_np[aug], 0, 1), out.detach().cpu().numpy()[0])
            do_i_learned_noise = torch_to_np(net_input[aug])-out.detach().cpu().numpy()[0]

            mse_what_tf = MSE(noisy_np, do_i_learned_noise)

            if psnr_noisy_max == 0:
                psnr_noisy_max = psrn_noisy
            elif psnr_noisy_max < psrn_noisy:
                psnr_noisy_max = psrn_noisy

            if SAVE_DURING_TRAINING and i % save_every == 0:
                # output_dir
                out_test_np = torch_to_np(out)  # I +N1
                #out_test_name = f'{i}_test'
                #save_image(out_test_name, np.clip(out_test_np, 0, 1), output_path=output_dir)

                net.eval()
                loss_add = 0
                with torch.no_grad():
                    out_effect_np_ = torch_to_np(net(img_noisy_torch[aug]))
                    out_effect_np.append(out_effect_np_)
                    psnr_1 = compare_psnr(img_aug_np[aug], np.clip(out_effect_np_, 0, 1))
                    test_do_i_learned_noise = torch_to_np(img_noisy_torch[aug]) - out_effect_np_

                    if psnr_max == 0:
                        psnr_max = psnr_1
                    elif psnr_max < psnr_1:
                        psnr_max = psnr_1

                    loss_add = loss_add + total_loss.item()

        training_loss_record.append(loss_add/len(img_noisy_torch))

        if i % 10 == 0:
            out_effect_np[0] = out_effect_np[0].transpose(1, 2, 0)
            out_effect_np[1] = (np.rot90(out_effect_np[1], 2, (1, 2))).transpose(1, 2, 0)
            out_effect_np[2] = (np.fliplr(out_effect_np[2])).transpose(1, 2, 0)
            out_effect_np[3] = (np.fliplr(np.rot90(out_effect_np[3], 2, (1, 2)))).transpose(1, 2, 0)
            final_result = np.mean(out_effect_np, 0)

            if i % 200 == 0:
                # save estimated x = x_pred
                final_result_reshape = np.reshape(final_result, (final_result.shape[2], final_result.shape[0], final_result.shape[1]))
                save_image("004_x_pred", final_result_reshape)


            psnr_2 = compare_psnr(img_aug_np[0].transpose(1, 2, 0), np.clip(final_result, 0, 1))
            final_ssim = compare_ssim(img_aug_np[0].transpose(1, 2, 0), np.clip(final_result, 0, 1), data_range=1, multichannel=True)

            if psnr_2_max == 0:
                psnr_2_max = psnr_2
                tmp_name_p = f'{files_name[:-4]}_{sigma_now * 255:.2f}_{psnr_2:.2f}_final_{final_ssim:.4f}'
                save_image(tmp_name_p, np.clip(final_result.transpose(2, 0, 1), 0, 1), output_path=output_dir)
            elif psnr_2_max < psnr_2:
                psnr_2_max = psnr_2
                tmp_name_p = f'{files_name[:-4]}_{sigma_now * 255:.2f}_{psnr_2:.2f}_final_{final_ssim:.4f}'
                save_image(tmp_name_p, np.clip(final_result.transpose(2, 0, 1), 0, 1), output_path=output_dir)
            if final_ssim_max == 0:
                final_ssim_max = final_ssim
            elif final_ssim_max < final_ssim:
                final_ssim_max = final_ssim
                tmp_name = f'{files_name[:-4]}_{sigma_now * 255:.2f}_{final_ssim:.4f}_final_{psnr_2:.2f}'
                save_image(tmp_name, np.clip(final_result.transpose(2, 0, 1), 0, 1), output_path=output_dir)


            print('%s Iteration %05d ,psnr 2: %f, psnr 2 max: %f, final ssim : %f, final ssim max: %f'
                  % (files_name, i, psnr_2, psnr_2_max, final_ssim, final_ssim_max))
            writer.add_scalar('final_test_psnr', psnr_2, i)
            writer.add_scalar('final_max_test_psnr', psnr_2_max, i)
            psnr_curve_max_record.append(psnr_2_max)
            ssim_curve_max_record.append(final_ssim_max)


    else:
        noisy_torch = np_to_torch(img_noisy_noisy_np - img_noisy_np)
        out = net(net_input)
        total_loss = mse(out, noisy_torch.type(dtype))

        total_loss.backward()
        psrn_noisy = compare_psnr(np.clip(img_noisy_np, 0, 1), (torch_to_np(net_input) - out.detach().cpu().numpy()[0]))
        do_i_learned_noise = torch_to_np(net_input) - out.detach().cpu().numpy()[0]
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
        loss_add = 0
        with torch.no_grad():
            out_effect_np = torch_to_np(img_noisy_torch - net(img_noisy_torch))
            psnr_1 = compare_psnr(img_np, np.clip(out_effect_np, 0, 1))
            test_do_i_learned_noise = torch_to_np(net(img_noisy_torch))

            if psnr_max == 0:
                psnr_max = psnr_1
            elif psnr_max < psnr_1:
                psnr_max = psnr_1

            loss_add = loss_add + total_loss.item()

        training_loss_record.append(loss_add / len(img_noisy_torch))
        if i % 10 == 0:
            final_result = out_effect_np.transpose(1, 2, 0)
            psnr_2 = compare_psnr(img_np.transpose(1, 2, 0), np.clip(final_result, 0, 1))
            final_ssim = compare_ssim(img_np.transpose(1, 2, 0), np.clip(final_result, 0, 1), data_range=1, multichannel=True)
            if psnr_2_max==0:
                psnr_2_max = psnr_2
                tmp_name_p = f'{files_name[:-4]}_{sigma_now*255:.2f}_{psnr_2:.2f}_final_{final_ssim:.4f}'
                save_image(tmp_name_p, np.clip(final_result.transpose(2, 0, 1), 0, 1), output_path=output_dir)
            elif psnr_2_max< psnr_2:
                psnr_2_max = psnr_2
                tmp_name_p = f'{files_name[:-4]}_{sigma_now*255:.2f}_{psnr_2:.2f}_final_{final_ssim:.4f}'
                save_image(tmp_name_p, np.clip(final_result.transpose(2, 0, 1), 0, 1), output_path=output_dir)
            if final_ssim_max==0:
                final_ssim_max = final_ssim
            elif final_ssim_max<final_ssim:
                final_ssim_max = final_ssim
                tmp_name = f'{files_name[:-4]}_{sigma_now*255:.2f}_{final_ssim:.4f}_final_{psnr_2:.2f}'
                save_image(tmp_name, np.clip(final_result.transpose(2, 0, 1), 0, 1), output_path=output_dir)

            print('%s Iteration %05d ,psnr 2: %f, psnr 2 max: %f, final ssim : %f, final ssim max: %f'
                  %(files_name, i,psnr_2, psnr_2_max, final_ssim, final_ssim_max))
            writer.add_scalar('final_test_psnr', psnr_2, i)
            writer.add_scalar('final_max_test_psnr', psnr_2_max, i)
            psnr_curve_max_record.append(psnr_2_max)
            ssim_curve_max_record.append(final_ssim_max)

    i += 1

    return total_loss

imsize = -1
SAVE_DURING_TRAINING = True
save_every = 1
TRAIN_PLAN = [5/255., 10/255., 15/255., 20/255., 25/255.]
## denoising
img_root = 'data/denoising/BSD68'

best_epoch_record = []
files = os.listdir(img_root)
psnr_record_img_name = []
psnr_record_max_test_psnr = []
psnr_2_max_record = []
final_ssim_max_record = []
output_dir = 'E:/JunXu/NAC_TPAMI/results/BSD68_g_resnet_Aug_Epoch=1000/'
os.makedirs(output_dir, exist_ok=True)
output_model = os.path.join(output_dir,'model')
os.makedirs(output_model, exist_ok=True)

DATA_AUG = True

psnr_curve_max_record = []
ssim_curve_max_record = []
training_loss_record = []

for index in range(0, len(files)):
    files_name = str(files[index])
    if files_name.endswith('004.png'):
        fname = os.path.join(img_root, files_name)
        writer = SummaryWriter()
        img_pil = Image.open(fname)
        img_np = pil_to_np(img_pil)
        # noisy_np_norm = np.random.poisson(1.0, size=img_np.shape)
        noisy_np_norm = np.random.normal(0.0, 1.0, size=img_np.shape)
        input_depth = img_np.shape[0]
        pad = 'replication'  # ['zero', 'replication', 'none']
        OPT_OVER = 'net'
        LR = 0.001

        OPTIMIZER = 'myadam'
        show_every = 1
        exp_weight = 0.99
        num_iter_plan = [1001, 1001, 1001, 1001, 1001]
        for current_sigma in range(len(TRAIN_PLAN)):
            num_iter = num_iter_plan[current_sigma]
            sigma_now = TRAIN_PLAN[current_sigma]
            noisy_np = noisy_np_norm * (sigma_now)

            psnr_2_max = 0
            psnr_noisy_max = 0
            psnr_max = 0
            final_ssim_max = 0
            if DATA_AUG:
                img_aug_np, img_aug_torch = create_augmentations_for_not_sqaure(img_np)
                img_noisy_pil = []
                img_noisy_np = []
                img_noisy_torch = []

                img_noisy_noisy_pil = []
                img_noisy_noisy_np = []
                img_noisy_noisy_torch = []

                for idx in range(len(img_aug_np)):
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
            else:
                img_noisy_pil, img_noisy_np, img_noisy_noisy_pil, img_noisy_noisy_np = \
                    get_noisy_noisy_image_with_noise(noisy_np, img_np)
                img_noisy_torch_ = np_to_torch(img_noisy_np).type(dtype)
                img_noisy_noisy_torch_ = np_to_torch(img_noisy_noisy_np).type(dtype)

            net = ResNet(input_depth, img_np.shape[0], 10, 64, 1).type(dtype)
            s = sum([np.prod(list(p.size())) for p in net.parameters()])
            print('Number of params: %d' % s)
            # Loss
            mse = torch.nn.MSELoss().type(dtype)
            net_input = img_noisy_noisy_torch
            i = 0
            p = get_params(OPT_OVER, net, net_input)
            current_model_dir = os.path.join(output_model,files_name[:-4], str(sigma_now*255))
            os.makedirs(current_model_dir, exist_ok=True)
            np.save(os.path.join(current_model_dir,'noise_matrix.npy'), noisy_np)
            optimize(net, OPTIMIZER, p, closure, LR, num_iter, output_dir=current_model_dir, interval=10)
            psnr_record_img_name.append(files_name)
            psnr_record_max_test_psnr.append(psnr_max)
            psnr_2_max_record.append(psnr_2_max)
            final_ssim_max_record.append(final_ssim_max)
            # best_epoch_record.append(best_epoch)

with open('nac_set12_g.pickle','wb') as f:
    resnet_aug = {'psnr_record_img_name': psnr_record_img_name,
                  'psnr_record_max_test_psnr': psnr_record_max_test_psnr,
                  'psnr_2_max_record': psnr_2_max_record,
                  'final_ssim_max_record':final_ssim_max_record,
                  'best_epoch_record': best_epoch_record}
    pickle.dump(resnet_aug, f)


