import os
from .common_utils import *
# import random

def get_p_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.

    Args:
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + img_np*np.random.poisson(sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np


def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np

def get_noisy_noisy_image(img_np, sigma, clip_or= False):
    """Add Gaussian noise and double Gaussian noise to an image.

    :param img_np: image np.array with values from 0 to 1
    :param sigma: std of the noise
    :return: img_noisy_np, img_noisy_noisy_np, img_noisy_pil, img_noisy_noisy_pil
    """
    noisy_np = np.random.normal(scale=sigma, size=img_np.shape)
    if clip_or:
       img_noisy_np = np.clip(img_np + noisy_np, 0, 1).astype(np.float32)
       img_noisy_pil = np_to_pil(img_noisy_np)
       # img_noisy_noisy_np = np.clip(img_np + noisy_np + noisy_np, 0, 1).astype(np.float32)
       img_noisy_noisy_np = np.clip(img_noisy_np + noisy_np, 0, 1).astype(np.float32)   ##
       img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)
    else:
        img_noisy_np = (img_np + noisy_np).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = (img_np + noisy_np + noisy_np).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)

    return noisy_np, img_noisy_pil, img_noisy_np, img_noisy_noisy_pil, img_noisy_noisy_np

def get_noisy1_noisy2_image(img_np, sigma1, sigma2, no_clip= True):
    """Add Gaussian noise and double Gaussian noise to an image.

    :param img_np: image np.array with values from 0 to 1
    :param sigma: std of the noise
    :return: img_noisy_np, img_noisy_noisy_np, img_noisy_pil, img_noisy_noisy_pil
    """

    sigma_norm = 1/255.
    noisy_norm_np = np.random.normal(scale=sigma_norm, size=img_np.shape)

    if not no_clip:
        img_noisy_np = np.clip(img_np + noisy_norm_np*(sigma1/sigma_norm), 0, 1).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = np.clip(img_np + noisy_norm_np*((sigma1+sigma2)/sigma_norm), 0, 1).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)
    elif no_clip:
        img_noisy_np = (img_np + noisy_norm_np*(sigma1/sigma_norm)).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = (img_np + noisy_norm_np*((sigma1+sigma2)/sigma_norm)).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)

    return img_noisy_pil, img_noisy_np, img_noisy_noisy_pil, img_noisy_noisy_np

def get_noisy_noisy_image_with_noise(noise_np, img_np, no_clip= True):
    """Add Gaussian noise and double Gaussian noise to an image.

    :param img_np: image np.array with values from 0 to 1
    :param sigma: std of the noise
    :return: img_noisy_np, img_noisy_noisy_np, img_noisy_pil, img_noisy_noisy_pil
    """

    if not no_clip:
        img_noisy_np = np.clip(img_np + noise_np, 0, 1).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = np.clip(img_np + noise_np*2, 0, 1).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)
    elif no_clip:
        img_noisy_np = (img_np + noise_np).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = (img_noisy_np + noise_np).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)

    return img_noisy_pil, img_noisy_np, img_noisy_noisy_pil, img_noisy_noisy_np

def get_p_noisy_noisy_image_with_noise(noise_np, img_np, no_clip= True):
    """Add Gaussian noise and double Gaussian noise to an image.

    :param img_np: image np.array with values from 0 to 1
    :param sigma: std of the noise
    :return: img_noisy_np, img_noisy_noisy_np, img_noisy_pil, img_noisy_noisy_pil
    """

    if not no_clip:
        img_noisy_np = np.clip(img_np + noise_np*img_np, 0, 1).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = np.clip(img_np + noise_np*img_np*2, 0, 1).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)
    elif no_clip:
        img_noisy_np = (img_np + noise_np*img_np).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = (img_noisy_np + noise_np*img_noisy_np).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)

    return img_noisy_pil, img_noisy_np, img_noisy_noisy_pil, img_noisy_noisy_np

def get_p2_noisy_noisy_image_with_noise(noise_np, img_np, no_clip= True):
    """Add Gaussian noise and double Gaussian noise to an image.

    :param img_np: image np.array with values from 0 to 1
    :param sigma: std of the noise
    :return: img_noisy_np, img_noisy_noisy_np, img_noisy_pil, img_noisy_noisy_pil
    """

    if not no_clip:
        img_noisy_np = np.clip(img_np + noise_np*img_np, 0, 1).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = np.clip(img_np + noise_np*img_np*2, 0, 1).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)
    elif no_clip:
        img_noisy_np = (img_np + noise_np*img_np).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = (img_noisy_np + noise_np*img_noisy_np).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)

    return img_noisy_pil, img_noisy_np, img_noisy_noisy_pil, img_noisy_noisy_np

def get_pg_noisy_noisy_image_with_noise(noise_p_np, noise_g_np, img_np, no_clip= True):
    """Add Gaussian noise and double Gaussian noise to an image.

    :param img_np: image np.array with values from 0 to 1
    :param sigma: std of the noise
    :return: img_noisy_np, img_noisy_noisy_np, img_noisy_pil, img_noisy_noisy_pil
    """

    if not no_clip:
       print('you write nothing!')
    elif no_clip:
        img_noisy_np = (img_np + noise_g_np + noise_p_np*img_np).astype(np.float32)
        img_noisy_pil = np_to_pil(img_noisy_np)
        img_noisy_noisy_np = (img_noisy_np + noise_g_np + noise_p_np*img_np).astype(np.float32)
        img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)

    return img_noisy_pil, img_noisy_np, img_noisy_noisy_pil, img_noisy_noisy_np

def get_pg_noisy_image(img_np, sigma_p, sigma_g):
    """Adds Gaussian noise to an image.

    Args:
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma_g, size=img_np.shape) +
                           np.random.poisson(sigma_p, size=img_np.shape)*img_np, 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np