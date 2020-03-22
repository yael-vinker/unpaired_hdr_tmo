import os
import pathlib

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
from skimage import exposure

import tranforms
from utils import params

# ====== IMAGE PRINTER ======
def print_image_details(im, title):
    print(title)
    print("shape : ", im.shape)
    print("max : ", np.max(im), "  min : ", np.min(im), "mean : ", np.mean(im))
    print("type : ", im.dtype)
    print("unique values : ", np.unique(im).shape[0])
    print()


def print_tensor_details(im, title):
    print(title)
    print("shape : ", im.shape)
    print("max : ", im.max(), "  min : ", im.min(), "mean : ", im.mean())
    print("type : ", im.dtype)
    print()


# ====== IMAGE READER ======
def read_hdr_image(path):
    path_lib_path = pathlib.Path(path)
    file_extension = os.path.splitext(path)[1]
    if file_extension == ".hdr":
        im = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
    elif file_extension == ".dng":
        im = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
    elif file_extension == ".exr":
        im = imageio.imread(path_lib_path, format="EXR-FI").astype('float32')
    else:
        raise Exception('invalid hdr file format: {}'.format(file_extension))
    return im


def read_ldr_image(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    im = im_origin / 255
    return im


def read_ldr_image_original_range(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    return im_origin


# ====== BRIGHTNESS FACTOR ======
def get_bump(im):
    tmp = im
    tmp[(tmp > 255) != 0] = 255
    tmp = tmp.astype('uint8')

    hist, bins = np.histogram(tmp, bins=255)

    a0 = np.mean(hist[0:63])
    a1 = np.mean(hist[64:200])
    return a1 / a0


def get_brightness_factor(im_hdr):
    im_hdr = im_hdr / np.max(im_hdr) * 255
    big = 1.1
    f = 1.0

    for i in range(1000):
        r = get_bump(im_hdr * f)
        if r < big:
            f = f * 1.01
        else:
            if r > 1 / big:
                return f
    return f


# ====== IMAGE MANIPULATE ======
def to_gray(im):
    return np.dot(im[..., :3], [0.299, 0.587, 0.114]).astype('float32')


def to_gray_tensor(rgb_tensor):
    r_image = rgb_tensor[0]
    g_image = rgb_tensor[1]
    b_image = rgb_tensor[2]
    grayscale_image = (0.299*r_image + 0.587*g_image + 0.114*b_image)
    grayscale_image = grayscale_image[None, :, :]
    return grayscale_image


def reshape_im(im, new_y, new_x):
    return skimage.transform.resize(im, (new_y, new_x),
                                             mode='reflect', preserve_range=True).astype("float32")


def to_0_1_range(im):
    if np.max(im) - np.min(im) == 0:
        im = (im - np.min(im)) / (np.max(im) - np.min(im) + params.epsilon)
    else:
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im


def to_minus1_1_range(im):
    return 2 * im - 1


def back_to_color(im_hdr, fake):
    fake = to_0_1_range(fake)
    if np.min(im_hdr) < 0:
        im_hdr = im_hdr + np.abs(np.min(im_hdr))
    im_gray_ = to_gray(im_hdr)
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / (im_gray_ + params.epsilon)
    norm_im[:, :, 1] = im_hdr[:, :, 1] / (im_gray_ + params.epsilon)
    norm_im[:, :, 2] = im_hdr[:, :, 2] / (im_gray_ + params.epsilon)
    norm_im = np.power(norm_im, 0.5)
    norm_im_gray = to_gray(norm_im)
    norm_im_gray = norm_im_gray[:, :, None]
    output_im = (norm_im / (norm_im_gray + params.epsilon)) * fake
    return output_im


def reshape_image(rgb_im):
    h, w = rgb_im.shape[0], rgb_im.shape[1]
    if min(h, w) > 3000:
        rgb_im = skimage.transform.resize(rgb_im, (int(rgb_im.shape[0] / 4),
                                                   int(rgb_im.shape[1] / 4)),
                                          mode='reflect', preserve_range=False).astype("float32")
    elif min(h, w) > 2000:
        rgb_im = skimage.transform.resize(rgb_im, (int(rgb_im.shape[0] / 3),
                                                   int(rgb_im.shape[1] / 3)),
                                          mode='reflect', preserve_range=False).astype("float32")
    return rgb_im


def reshape_image_fixed_size(rgb_im):
    rgb_im = skimage.transform.resize(rgb_im, (1024,
                                               2048),
                                      mode='reflect', preserve_range=False).astype("float32")
    return rgb_im


def back_to_color_batch(im_hdr_batch, fake_batch):
    b_size = im_hdr_batch.shape[0]
    output = []
    for i in range(b_size):
        im_hdr = im_hdr_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        fake = fake_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        norm_im = back_to_color(im_hdr, fake)
        output.append(torch.from_numpy(norm_im.transpose((2, 0, 1))).float())
    return torch.stack(output)

def display_tensor(tensor, cmap):
    im = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    if cmap == "gray":
        im = np.squeeze(im)
    im = to_0_1_range(im)
    plt.imshow(im, cmap=cmap)
    # plt.show()
