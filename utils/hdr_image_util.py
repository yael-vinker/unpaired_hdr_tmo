import torchvision.transforms as transforms
import params
import torchvision.utils as vutils
import torch
import pathlib
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import exposure
import math
import cv2
from PIL import Image
import imageio
import torch
import tranforms as transforms_
# import hdr_image_utils
# import Writer

def to_gray(im):
    return np.dot(im[...,:3], [0.299, 0.587, 0.114]).astype('float32')

def read_hdr_image(path):
    path_lib_path = pathlib.Path(path)
    file_extension = os.path.splitext(path)[1]
    if file_extension == ".hdr":
        im = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
    elif file_extension == ".dng":
        im = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
    else:
        raise Exception('invalid hdr file format: {}'.format(file_extension))
    return im


def read_ldr_image(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    im = im_origin / 255
    return im

def hdr_log_loader_factorize(path, range_factor):
    im_origin = read_hdr_image(path)
    max_origin = np.max(im_origin)
    image_new_range = (im_origin / max_origin) * range_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(range_factor + 1)).astype('float32')
    return im

def uint_normalization(im):
    # norm_im = ((im / np.max(im)) * 255).astype("uint8")
    norm_im = (im * 255).astype("uint8")
    # norm_im_clamp = np.clip(norm_im, 0, 255)
    # norm_im_clamp = norm_im
    norm_im = np.clip(norm_im, 0, 255)
    return norm_im

def to_0_1_range(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def exp_normalization(im):
    return (np.exp(im) - 1) / (np.exp(np.max(im)) - 1)

def log100_normalization(im, isHDR):
    IMAGE_SCALE = 100
    IMAGE_MAX_VALUE = 255
    if isHDR:
        norm_im = (np.exp(im) - 1) / IMAGE_SCALE
        if norm_im.shape[2] == 1:
            gamma_corrected = exposure.adjust_gamma(norm_im, 0.5)
            im1 = (gamma_corrected * IMAGE_MAX_VALUE).astype("uint8")
            # norm_im = im1[:, :, 0]
            norm_im = im1
        else:
            tone_map1 = cv2.createTonemapReinhard(1.5, 0, 0, 0)
            im1_dub = tone_map1.process(norm_im.copy()[:, :, ::-1])
            im1 = (im1_dub * IMAGE_MAX_VALUE).astype("uint8")
            norm_im = im1

    else:
        norm_im = (((np.exp(im) - 1) / IMAGE_SCALE) * IMAGE_MAX_VALUE).astype("uint8")
    # norm_im_clamp = np.clip(norm_im, 0, 255)
    return norm_im

def back_to_color(im_hdr, fake):
    im_gray_ = np.sum(im_hdr, axis=2)
    fake = to_0_1_range(fake)
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / im_gray_
    norm_im[:, :, 1] = im_hdr[:, :, 1] / im_gray_
    norm_im[:, :, 2] = im_hdr[:, :, 2] / im_gray_
    output_im = np.power(norm_im, 0.5) * fake
    return output_im

def back_to_color_batch(im_hdr_batch, fake_batch):
    b_size = im_hdr_batch.shape[0]
    output = []
    for i in range(b_size):
        im_hdr = im_hdr_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        fake = fake_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        norm_im = back_to_color(im_hdr, fake)
        output.append(torch.from_numpy(norm_im.transpose((2, 0, 1))).float())
    return torch.stack(output)


def back_to_color_exp_batch(im_hdr_batch, fake_batch):
    b_size = im_hdr_batch.shape[0]
    output = []
    for i in range(b_size):
        im_hdr = im_hdr_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        fake = fake_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        norm_im = back_to_color_exp(im_hdr, fake)
        output.append(torch.from_numpy(norm_im.transpose((2, 0, 1))).float())
    return torch.stack(output)


def back_to_color_exp(im_hdr, fake):
    im_gray_ = np.sum(im_hdr, axis=2)
    fake = to_0_1_range(fake)
    fake = exp_normalization(fake)
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / im_gray_
    norm_im[:, :, 1] = im_hdr[:, :, 1] / im_gray_
    norm_im[:, :, 2] = im_hdr[:, :, 2] / im_gray_
    output_im = np.power(norm_im, 0.5) * fake
    return output_im


# def back_to_color_tensor(fake, im_hdr_display):
#     """
#
#     :param fake: range [-1, -] gray
#     :param im_hdr: range [-1, 1] gray
#     :param im_hdr_display: range [0,1]
#     :return:
#     """
#     gray_im = im_hdr_display.sum(dim=0)
#     rgb_hdr_copy = im_hdr_display.clone()
#     rgb_hdr_copy[0, :, :] = rgb_hdr_copy[0, :, :] / gray_im
#     rgb_hdr_copy[1, :, :] = rgb_hdr_copy[1, :, :] / gray_im
#     rgb_hdr_copy[2, :, :] = rgb_hdr_copy[2, :, :] / gray_im
#     gray_fake_to_0_1 = to_0_1_range_tensor(fake)
#     output_im = torch.pow(rgb_hdr_copy, 0.5) * gray_fake_to_0_1
#     # display_tensor(output_im)
#     return output_im
#
#
# def back_to_color_batch_tensor(fake_batch, hdr_input_display_batch):
#     b_size = fake_batch.shape[0]
#     output = [back_to_color_tensor(fake_batch[i], hdr_input_display_batch[i]) for i in range(b_size)]
#     return torch.stack(output)

def display_tensor(tensor):
    im_display = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    im_display = to_0_1_range(im_display)
    plt.imshow(im_display)
    plt.show()