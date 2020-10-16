import os
import pathlib

import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch

import tranforms
from utils import params
from math import log, ceil, floor


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
    elif file_extension == ".npy":
        im = np.load(path, allow_pickle=True)[()]
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
    tmp[(tmp > 255)] = 255
    tmp = tmp.astype('uint8')

    hist, bins = np.histogram(tmp, bins=255)

    a0 = np.mean(hist[0:65])
    a1 = np.mean(hist[65:200])
    # a0 = np.sum(hist[0:65])
    # a1 = np.sum(hist[65:200])

    return a1 / a0


def plot_hist(rgb_img, brightness_factor_b, i):
    rgb_img = (rgb_img / np.max(rgb_img)) * brightness_factor_b
    rgb_img[rgb_img > 255] = 255
    rgb_img = rgb_img.astype('uint8')
    counts, bins = np.histogram(rgb_img.reshape(np.prod(rgb_img.shape)), range(256))
    # plot histogram centered on values 0..255
    plt.bar(bins[:-2], counts[:-1], width=1, edgecolor='none')
    plt.xlim([-0.5, 300])
    a0 = np.mean(counts[0:65])
    a1 = np.mean(counts[65:200])
    plt.title(str(a1 / a0) + " i " + str(i), fontSize=12)


def get_brightness_factor(im_hdr, mean_target, factor):
    im_hdr = (im_hdr / np.max(im_hdr)) * 255
    big = 1.1
    f = 1.0

    for i in range(1500):
        r = get_bump(im_hdr * f)

        im_gamma = (((im_hdr / np.max(im_hdr)) ** (1 / (1 + factor * np.log10(f * 255)))) * 255)
        # print("i[%d]  r[%f]  f[%f] mean[%f]" % (i, r, f, np.mean(im_gamma)))
        # if r > 1 and i % 5 == 0:
        #     print("i[%d]  r[%f]  f[%f] mean[%f]" % (i, r, f, np.min(im_gamma)))
        if r > big and np.mean(im_gamma) > mean_target:
            print("i[%d]  r[%f]  f[%f] mean[%f]" % (i, r, f, np.mean(im_gamma)))
            return f
        else:
            f = f * 1.01
    print("i[%d]  r[%f]  f[%f]" % (i, r, f))
    return f


def print_im_data(J):
    print("======================")
    print(np.percentile(J, 50))
    print(np.percentile(J, 1))
    print(np.percentile(J, 99))
    print(np.percentile(J, 100))
    print(J.min(), J.mean(), J.max())
    # plt.hist(np.log(J), rwidth=0.9, color='#607c8e', density=True, bins=5)
    # plt.show()


def get_new_brightness_factor(M):
    # from PIL import Image
    J = np.mean(M, axis=2)
    J = np.reshape(J, (-1,))
    M = M / np.max(J)

    J = J / np.max(J)
    J = J[J>0]

    npix = J.shape[0]
    Cout = 1
    for i in range(100):

        C = np.sqrt(2) ** i

        I = J * C

        I = I[I < 0.99]

        if I.shape[0] / npix < 0.1:
            # Cout = C * np.sqrt(2)

            break

        h = np.histogram(I, bins=5)
        h = h[0]

        if np.mean(h[1]) > 0:
            rat = np.mean(h[0]) / np.mean(h[1])

            # print("%d: %.2f (%.2f)" % (i, rat, I.shape[0] / npix))
            if rat > 0.5:
                Cout = C * np.sqrt(2)

            else:
                print("%d: %.2f (%.2f) [%.4f]" % (i, rat, I.shape[0] / npix, Cout))
    return Cout


def closest_power(x):
    # closest_power_ = max(x + ceil(x * 0.2 / 2.) * 2, 2**(x - 1).bit_length())
    closest_power_ = (2 ** (x - 1).bit_length())
    return 1 if x == 0 else closest_power_ + 12


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
                                    mode='reflect', preserve_range=False,
                                    anti_aliasing=True, order=3).astype("float32")


def to_0_1_range(im):
    if np.max(im) - np.min(im) == 0:
        im = (im - np.min(im)) / (np.max(im) - np.min(im) + params.epsilon)
    else:
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im

def to_0_1_range_outlier(im):
    im_max = np.percentile(im, 99.0)
    im_min = np.percentile(im, 1)
    if np.max(im) - np.min(im) == 0:
        im = (im - im_min) / (im_max - im_min + params.epsilon)
    else:
        im = (im - im_min) / (im_max - im_min)
    return np.clip(im,0,1)


def to_minus1_1_range(im):
    return 2 * im - 1


def back_to_color(im_hdr, fake):
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


def reshape_image(rgb_im, train_reshape):
    h, w = rgb_im.shape[0], rgb_im.shape[1]
    if train_reshape:
        rgb_im = skimage.transform.resize(rgb_im, (params.input_size, params.input_size),
                                          mode='reflect', preserve_range=False,
                                          anti_aliasing=True, order=3).astype("float32")
    else:
        if min(h, w) > 3000:
            rgb_im = skimage.transform.resize(rgb_im, (int(rgb_im.shape[0] / 4),
                                                       int(rgb_im.shape[1] / 4)),
                                              mode='reflect', preserve_range=False,
                                              anti_aliasing=True, order=3).astype("float32")
        elif min(h, w) > 2000:
            rgb_im = skimage.transform.resize(rgb_im, (int(rgb_im.shape[0] / 3),
                                                       int(rgb_im.shape[1] / 3)),
                                              mode='reflect', preserve_range=False,
                                              anti_aliasing=True, order=3).astype("float32")
    return rgb_im


def reshape_image_fixed_size(rgb_im):
    rgb_im = skimage.transform.resize(rgb_im, (1024,
                                               2048),
                                      mode='reflect', preserve_range=False,
                                      order=3).astype("float32")
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
    # im = to_0_1_range(im)

    plt.imshow(im, cmap=cmap, vmin=im.min(), vmax=im.max())
    # plt.show()


# ====== SAVE IMAGES ======
def save_color_tensor_batch_as_numpy(batch, output_path, batch_num):
    b_size = batch.shape[0]
    for i in range(b_size):
        im_hdr = batch[i]
        tensor_0_1 = im_hdr.squeeze()
        im = tensor_0_1.clamp(0,1).permute(1, 2, 0).detach().cpu().numpy()
        im = (im * 255).astype('uint8')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        cur_output_path = os.path.join(output_path, str(batch_num) + "_" + str(i) + ".jpg")
        imageio.imwrite(cur_output_path, im)


def save_gray_tensor_as_numpy(tensor, output_path, im_name):
    tensor = tensor.clamp(0, 1).clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(tensor)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')


def save_gray_tensor_as_numpy_stretch(tensor, output_path, im_name):
    tensor = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(tensor)
    tensor_0_1 = to_0_1_range_outlier(tensor_0_1)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')

def save_gray_tensor_as_numpy_stretch_entire_range(tensor, output_path, im_name):
    tensor = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(tensor)
    tensor_0_1 = to_0_1_range(tensor_0_1)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')


def save_color_tensor_as_numpy(tensor, output_path, im_name):
    tensor = tensor.clamp(0, 1).clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(tensor)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')

