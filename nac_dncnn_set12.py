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
from models.dncnn import weights_init_kaiming
import random
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

    global i, net_input, psnr_max, psnr_noisy_max, files_name, psnr_2_max, noisy_np
    global TRAIN_PLAN, noisy_np_norm, sigma_now, final_ssim, final_ssim_max, files_name
    global psnr_curve_max_record, ssim_curve_max_record, training_loss_record

    out_effect_np = []
    if DATA_AUG:
        for aug in range(len(img_noisy_torch)):
            noisy_torch = np_to_torch(img_noisy_noisy_np[aug]-img_noisy_np[aug])
            out = net(net_input[aug])
            total_loss = mse(out, noisy_torch.type(dtype))

            total_loss.backward()
            psrn_noisy = compare_psnr(np.clip(img_noisy_np[aug], 0, 1), (torch_to_np(net_input[aug])-out.detach().cpu().numpy()[0]))
            # do_i_learned_noise = out.detach().cpu().numpy()[0]
            # mse_what_tf = MSE(noisy_np, do_i_learned_noise)

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
                    out_effect_np_ = torch_to_np(img_noisy_torch[aug]-net(img_noisy_torch[aug]))
                    out_effect_np.append(out_effect_np_)
                    psnr_1 = compare_psnr(img_aug_np[aug], np.clip(out_effect_np_, 0, 1))
                    test_do_i_learned_noise = torch_to_np(net(img_noisy_torch[aug]))

                    if psnr_max == 0:
                        psnr_max = psnr_1
                    elif psnr_max < psnr_1:
                        psnr_max = psnr_1

                    loss_add = loss_add + total_loss.item()

        training_loss_record.append(loss_add/len(img_noisy_torch))
        if i % 10 == 0:
            out_effect_np[0] = out_effect_np[0].transpose(1, 2, 0)
            for aug in range(1, 8):
                if aug < 4:
                   out_effect_np[aug] = np.rot90(out_effect_np[aug].transpose(1, 2, 0), 4-aug)
                else:
                    out_effect_np[aug] = np.flipud(np.rot90(out_effect_np[aug].transpose(1, 2, 0), 8-aug))
            final_reuslt = np.mean(out_effect_np, 0)

            psnr_2 = compare_psnr(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt, 0, 1))
            final_ssim = compare_ssim(img_aug_np[0].transpose(1, 2, 0), np.clip(final_reuslt, 0, 1), data_range=1, multichannel=True)

            if psnr_2_max == 0:
                psnr_2_max = psnr_2
                tmp_name_p = f'{files_name[:-4]}_{sigma_now * 255:.2f}_{psnr_2:.2f}_final_{final_ssim:.4f}'
                save_image(tmp_name_p, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)
            elif psnr_2_max < psnr_2:
                psnr_2_max = psnr_2
                tmp_name_p = f'{files_name[:-4]}_{sigma_now * 255:.2f}_{psnr_2:.2f}_final_{final_ssim:.4f}'
                save_image(tmp_name_p, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)
            if final_ssim_max == 0:
                final_ssim_max = final_ssim
            elif final_ssim_max < final_ssim:
                final_ssim_max = final_ssim
                tmp_name = f'{files_name[:-4]}_{sigma_now * 255:.2f}_{final_ssim:.4f}_final_{psnr_2:.2f}'
                save_image(tmp_name, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)

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
        do_i_learned_noise = out.detach().cpu().numpy()[0]
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
            final_reuslt = out_effect_np.transpose(1, 2, 0)
            psnr_2 = compare_psnr(img_np.transpose(1, 2, 0), np.clip(final_reuslt, 0, 1))
            final_ssim = compare_ssim(img_np.transpose(1, 2, 0), np.clip(final_reuslt, 0, 1), data_range=1, multichannel=True)
            if psnr_2_max==0:
                psnr_2_max = psnr_2
                tmp_name_p = f'{files_name[:-4]}_{sigma_now*255:.2f}_{psnr_2:.2f}_final_{final_ssim:.4f}'
                save_image(tmp_name_p, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)
            elif psnr_2_max< psnr_2:
                psnr_2_max = psnr_2
                tmp_name_p = f'{files_name[:-4]}_{sigma_now*255:.2f}_{psnr_2:.2f}_final_{final_ssim:.4f}'
                save_image(tmp_name_p, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)
            if final_ssim_max==0:
                final_ssim_max = final_ssim
            elif final_ssim_max<final_ssim:
                final_ssim_max = final_ssim
                tmp_name = f'{files_name[:-4]}_{sigma_now*255:.2f}_{final_ssim:.4f}_final_{psnr_2:.2f}'
                save_image(tmp_name, np.clip(final_reuslt.transpose(2, 0, 1), 0, 1), output_path=output_dir)

            print('%s, sigma %f, Epoch %05d, psnr 2: %f, psnr 2 max: %f, final ssim : %f, final ssim max: %f'
                  %(files_name, sigma_now*255, i, psnr_2, psnr_2_max, final_ssim, final_ssim_max))
            writer.add_scalar('final_test_psnr', psnr_2, i)
            writer.add_scalar('final_max_test_psnr', psnr_2_max, i)
            psnr_curve_max_record.append(psnr_2_max)
            ssim_curve_max_record.append(final_ssim_max)

    i += 1

    return total_loss


imsize = -1
SAVE_DURING_TRAINING = True
save_every = 1

TRAIN_PLAN = [40/255., 50/255.] #[5/255., 10/255., 15/255., 20/255., 25/255.]

## denoising
img_root = 'data/Set12'

files = os.listdir(img_root)
psnr_record_img_name = []
psnr_record_max_test_psnr = []
psnr_2_max_record = []
final_ssim_max_record = []
output_dir = 'D:/JunXu/NAC_TPAMI/results/set12_g_DnCNN_Aug_LR=0.001_Epoch=180/'
full_output_dir = Path(output_dir).resolve()
if not os.path.isdir(full_output_dir):
    os.mkdir(full_output_dir)
DATA_AUG = True
psnr_curve_max_record = []
ssim_curve_max_record = []
training_loss_record = []

for index in range(6, len(files)):
    files_name = str(files[index])
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
        # num_iter_plan = [101, 101, 101, 101, 101]
        num_iters = 101

        figsize = 4
        for current_sigma in range(len(TRAIN_PLAN)):
            sigma_now = TRAIN_PLAN[current_sigma]
            noisy_np = noisy_np_norm * (sigma_now)
            psnr_2_max = 0
            psnr_noisy_max = 0
            psnr_max = 0
            final_ssim_max = 0
            if DATA_AUG:
                img_aug_np, img_aug_torch = create_augmentations(img_np)
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
                save_image('noisy_img',np.clip(img_noisy_np[0], 0, 1 ), output_dir)
            else:
                img_noisy_pil, img_noisy_np, img_noisy_noisy_pil, img_noisy_noisy_np = \
                    get_noisy_noisy_image_with_noise(noisy_np, img_np)
                img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
                img_noisy_noisy_torch = np_to_torch(img_noisy_noisy_np).type(dtype)
                save_image('noisy_img',np.clip(img_noisy_np, 0, 1 ), output_dir)

            net = DnCNN(input_depth).type(dtype)
            # Compute number of parameters
            s = sum([np.prod(list(p.size())) for p in net.parameters()])
            print('Number of params: %d' % s)
            # Loss
            mse = torch.nn.MSELoss().type(dtype)
             # Optimize
            net_input = img_noisy_noisy_torch
            i = 0
            p = get_params(OPT_OVER, net, net_input)
            optimize(net, OPTIMIZER, p, closure, LR, num_iters)
            psnr_record_img_name.append(files_name)
            psnr_record_max_test_psnr.append(psnr_max)
            psnr_2_max_record.append(psnr_2_max)
            final_ssim_max_record.append(final_ssim_max)

with open('l2_set12_g_dncnn.pickle','wb') as f:
    resnet_aug = {'psnr_record_img_name': psnr_record_img_name,
                  'psnr_record_max_test_psnr': psnr_record_max_test_psnr,
                  'psnr_2_max_record': psnr_2_max_record,
                  'final_ssim_max_record':final_ssim_max_record,
                  'psnr_curve_max_record':psnr_curve_max_record,
                  'ssim_curve_max_record':ssim_curve_max_record,
                  'training_loss_record':training_loss_record}
    pickle.dump(resnet_aug, f)

print('I am finish training now!')
        # psnr_record_max_test_psnr.append(psnr_max)
        # out_np = torch_to_np(net(net_input))
        # q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)

