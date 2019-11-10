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