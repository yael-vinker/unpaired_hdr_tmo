import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import os
import pathlib
from math import exp
import time
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import GanTrainer
import tranforms as transforms_
import utils.hdr_image_util as hdr_image_util
# import hdr_image_utils
from models import ssim
from old_files import HdrImageFolder, TMQI
from utils import params
from torch import nn
from data_generator import create_dng_npy_data
import tranforms
import utils.data_loader_util as data_loader_util
import matplotlib.patches as patches



def get_f(im_path, reshape="half"):
    print(im_path)
    import skimage
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    print(rgb_img.shape)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    gray_im = hdr_image_util.to_gray(rgb_img)
    # if reshape == "256":
    #     gray_im_temp = hdr_image_util.reshape_image(gray_im, train_reshape=True)
        # gray_im_temp = skimage.transform.resize(gray_im, (128, 128),
        #                                   mode='reflect', preserve_range=False, anti_aliasing=True).astype("float32")
    # else:
        # gray_im_temp = skimage.transform.resize(gray_im, (params.input_size, params.input_size),
        #                                         mode='reflect', preserve_range=False, anti_aliasing=True).astype(
        #     "float32")
        # gray_im_temp = hdr_image_util.reshape_image(gray_im, train_reshape=True)
    gray_im_temp = hdr_image_util.reshape_image(gray_im, train_reshape=False)
        # gray_im_temp = hdr_image_util.reshape_im(gray_im, gray_im.shape[0] // 2, gray_im.shape[1] // 2)
    # gray_im_temp = gray_im
    f_factor = hdr_image_util.get_brightness_factor(gray_im_temp)
    brightness_factor_a = f_factor * 255
    print("256 shape ", brightness_factor_a)
    brightness_factor_b = f_factor * 255
    print("original shape ", brightness_factor_b)
    # gray_im_a = get_image_with_bf_gamma(gray_im_temp, brightness_factor_a)
    # gray_im_b = get_image_with_bf_gamma(gray_im, brightness_factor_b)
    # rgb_img = get_image_with_bf_gamma(rgb_img, brightness_factor_b)
    # gray_im_temp = skimage.transform.resize(gray_im, (params.input_size, params.input_size),
    #                                         mode='reflect', preserve_range=False, anti_aliasing=True).astype(
    #     "float32")

    gray_im_a = get_image_with_bf(gray_im_temp, brightness_factor_a)
    # hdr_image_util.print_image_details(gray_im_a, "gray_A")

    gray_im_b = get_image_with_bf_gamma(gray_im_temp, brightness_factor_b)
    # gray_im_b = hdr_image_util.reshape_im(gray_im_b, 256, 256)

    rgb_img = get_image_with_bf(rgb_img, brightness_factor_b)

    return gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img


def get_image_with_bf(im, brightness_factor):
    gray_im_a = (im / np.max(im))
    gray_im_a = gray_im_a * brightness_factor
    gray_im_a[gray_im_a > 255] = 255
    gray_im_a = gray_im_a.astype('uint8')
    gray_im_a = gray_im_a / np.max(gray_im_a)
    # hdr_image_util.print_image_details(gray_im_a, "mul")
    return gray_im_a


def get_image_with_bf_gamma(im, brightness_factor):
    log_b = np.log(brightness_factor) / np.log(2)
    # log_b = np.log10(brightness_factor)
    # gray_im = (im / np.max(im)) ** (1 / (1 + log_b))
    gray_im = (im / np.max(im)) ** (1 / (1 + (2/3)*np.log10(brightness_factor)))
    # gray_im = (gray_im - gray_im.min()) / (gray_im.max() - gray_im.min())
    # hdr_image_util.print_image_details(gray_im, "gamma")
    return gray_im


def plot_hist(rgb_img):
    rgb_img = (rgb_img) * 255
    a3 = np.mean(rgb_img)
    # a2 = np.mean(rgb_img)
    rgb_img = rgb_img.astype('uint8')
    hist, bins = np.histogram(rgb_img, bins=255)
    # plot histogram centered on values 0..255
    plt.bar(bins[:-2], hist[:-1], width=1, edgecolor='none')
    plt.xlim([-0.5, 256])
    # a0 = np.mean(counts[0:65])
    # a1 = np.mean(counts[65:200])
    a0 = np.mean(hist[0:60])
    a1 = np.mean(hist[60:200])
    # print("hist", hist)
    a2 = rgb_img[rgb_img != 255].mean()
    title = "mean_gamma[%.4f] mean_hist[%.4f]" % (a3, a2)
    plt.title(title, fontSize=10)
    # print("hist data")
    # print(np.mean(hist[0:200]))


def plot_hist_all():
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/belgium.hdr"
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    gray_im = hdr_image_util.to_gray(rgb_img)
    gray_im_temp = ((gray_im / np.max(gray_im)) * 255)
    rgb_img = ((rgb_img / np.max(rgb_img)) * 255)

    plt.subplot(3, 4, 1)
    plt.imshow(rgb_img.astype('uint8'))
    plt.axis("off")
    plt.subplot(3, 4, 2)
    plot_hist(rgb_img)

    plt.subplot(3, 4, 3)
    plt.imshow(gray_im_temp.astype('uint8'), cmap='gray')
    plt.axis("off")
    plt.subplot(3, 4, 4)
    plot_hist(gray_im_temp)

    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/BigfootPass.exr"
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    gray_im = hdr_image_util.to_gray(rgb_img)
    gray_im_temp = ((gray_im / np.max(gray_im)) * 255)
    rgb_img = ((rgb_img / np.max(rgb_img)) * 255)

    plt.subplot(3, 4, 5)
    plt.imshow(rgb_img.astype('uint8'))
    plt.axis("off")
    plt.subplot(3, 4, 6)
    plot_hist(rgb_img)

    plt.subplot(3, 4, 7)
    plt.imshow(gray_im_temp.astype('uint8'), cmap='gray')
    plt.axis("off")
    plt.subplot(3, 4, 8)
    plot_hist(gray_im_temp)

    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr"
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    gray_im = hdr_image_util.to_gray(rgb_img)
    gray_im_temp = ((gray_im / np.max(gray_im)) * 255)
    rgb_img = ((rgb_img / np.max(rgb_img)) * 255)

    plt.subplot(3, 4, 9)
    plt.imshow(rgb_img.astype('uint8'))
    plt.axis("off")
    plt.subplot(3, 4, 10)
    plot_hist(rgb_img)

    plt.subplot(3, 4, 11)
    plt.imshow(gray_im_temp.astype('uint8'), cmap='gray')
    plt.axis("off")
    plt.subplot(3, 4, 12)
    plot_hist(gray_im_temp)
    plt.show()


def f_test():
    plt.figure()
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/WillyDesk.exr"
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/belgium.hdr"
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_6G7M_20150328_183029_897.dng"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path, reshape="256")

    plt.subplot(3, 4, 1)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log10(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 2)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 3)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log10(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 4)
    plot_hist(gray_im_b)

    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/BigfootPass.exr"
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_0006_20160722_100954_713.dng"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path, reshape="256")

    plt.subplot(3, 4, 5)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log10(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 6)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 7)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log10(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 8)
    plot_hist(gray_im_b)
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr"
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_6G7M_20150328_183029_897.dng"
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_0006_20160726_105942_902.dng"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path, reshape="256")
    plt.subplot(3, 4, 9)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log10(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 10)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 11)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log10(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 12)
    plot_hist(gray_im_b)
    plt.show()


def f_test2():
    plt.figure()

    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/belgium.hdr"
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_6G7M_20150328_183029_897.dng"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path)

    plt.subplot(3, 4, 1)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log2(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 2)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 3)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log2(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 4)
    plot_hist(gray_im_b)

    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/BigfootPass.exr"
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/507.exr"
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_0006_20160722_100954_713.dng"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path)
    plt.subplot(3, 4, 5)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log2(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 6)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 7)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log2(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 8)
    plot_hist(gray_im_b)
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr"
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_0006_20160726_105942_902.dng"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path)
    plt.subplot(3, 4, 9)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log2(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 10)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 11)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log2(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 12)
    plot_hist(gray_im_b)
    plt.show()


def f_test3():
    plt.figure()
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/WillyDesk.exr"
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/belgium.hdr"
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_6G7M_20150328_183029_897.dng"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path)
    gray_im_a = hdr_image_util.reshape_image(gray_im_a, train_reshape=True)
    gray_im_b = hdr_image_util.reshape_image(gray_im_b, train_reshape=True)
    plt.subplot(3, 4, 1)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log2(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 2)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 3)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log2(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 4)
    plot_hist(gray_im_b)

    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/BigfootPass.exr"
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_0006_20160722_100954_713.dng"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path)
    gray_im_a = hdr_image_util.reshape_image(gray_im_a, train_reshape=True)
    gray_im_b = hdr_image_util.reshape_image(gray_im_b, train_reshape=True)
    plt.subplot(3, 4, 5)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log2(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 6)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 7)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log2(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 8)
    plot_hist(gray_im_b)
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr"
    gray_im_a, brightness_factor_a, gray_im_b, brightness_factor_b, rgb_img = get_f(im_path)
    gray_im_a = hdr_image_util.reshape_image(gray_im_a, train_reshape=True)
    gray_im_b = hdr_image_util.reshape_image(gray_im_b, train_reshape=True)
    plt.subplot(3, 4, 9)
    plt.axis("off")
    plt.title("a " + str(brightness_factor_a) + " " + str(np.log2(brightness_factor_a)), fontSize=10)
    plt.imshow(gray_im_a, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 10)
    plot_hist(gray_im_a)

    plt.subplot(3, 4, 11)
    plt.axis("off")
    plt.title("b " + str(brightness_factor_b) + " " + str(np.log2(brightness_factor_b)), fontSize=10)
    plt.imshow(gray_im_b, cmap='gray', vmin=0, vmax=1)

    plt.subplot(3, 4, 12)
    plot_hist(gray_im_b)
    # plt.show()


def transfomr_test():
    rgb_img = hdr_image_util.read_hdr_image(
        "/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_6G7M_20150328_183029_897.dng")
    gray_im = hdr_image_util.to_gray(rgb_img)
    # hdr_image_util.print_image_details(gray_im, "")
    f_factor = 219
    brightness_factor = f_factor * 255
    print(brightness_factor)
    gray_im = (gray_im / np.max(gray_im)) ** (1 / (1 + 1.5 * np.log10(brightness_factor)))
    hdr_image_util.print_image_details(gray_im, "after")
    import torchvision.transforms as torch_transforms
    image_transform_no_norm = torch_transforms.Compose([
        transforms_.ToTensor(),
        transforms_.ScaleTensor(params.input_size),
        transforms_.CenterCropTensor(params.input_size),
    ])
    gray_im_log = image_transform_no_norm(gray_im)
    im = gray_im_log.clone().permute(1, 2, 0).detach().cpu().numpy()
    im = np.squeeze(im)
    # im = to_0_1_range(im)
    imageio.imwrite("/Users/yaelvinker/PycharmProjects/lab/tests/tensor.jpg", im)
    hdr_image_util.print_tensor_details(gray_im_log, "after")
    plt.subplot(2, 1, 1)
    plt.title("tensor")
    hdr_image_util.display_tensor(gray_im_log, 'gray')

    image_transform_no_norm_1 = torch_transforms.Compose([
        transforms_.Scale(params.input_size),
        transforms_.CenterCrop(params.input_size),
        transforms_.ToTensor(),
    ])
    gray_im_log = image_transform_no_norm_1(gray_im)
    im = gray_im_log.clone().permute(1, 2, 0).detach().cpu().numpy()
    im = np.squeeze(im)
    # im = to_0_1_range(im)
    imageio.imwrite("/Users/yaelvinker/PycharmProjects/lab/tests/reg.jpg", im)
    hdr_image_util.print_tensor_details(gray_im_log, "after")
    plt.subplot(2, 1, 2)
    plt.title("reg")
    hdr_image_util.display_tensor(gray_im_log, 'gray')
    plt.show()


def model_save_test():
    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/temp_data/bigFogMap.hdr"
    rgb_img, gray_im_log = create_dng_npy_data.hdr_preprocess(im_path,
                                                              use_factorise_gamma_data=True, factor_coeff=1.0,
                                                              train_reshape=False, gamma_log=10)
    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(gray_im_log[0,0].numpy(), cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 2, 2)
    plot_hist(gray_im_log)

    im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/temp_data/merged_6G7M_20150328_183029_897.dng"
    rgb_img, gray_im_log = create_dng_npy_data.hdr_preprocess(im_path,
                                                              use_factorise_gamma_data=True, factor_coeff=1.0,
                                                              train_reshape=False, gamma_log=10)
    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.imshow(gray_im_log, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 2, 4)
    plot_hist(gray_im_log)

    plt.show()

def raanan_f_test(im_path):
    # im_path = "/Users/yaelvinker/PycharmProjects/lab/utils/folders/data/belgium.hdr"
    train_reshape = True
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    gray_im = hdr_image_util.to_gray(rgb_img)
    rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape)
    gray_im = hdr_image_util.reshape_image(gray_im, train_reshape)
    gray_im = gray_im - gray_im.min()

    plt.subplot(3,1,1)
    plt.imshow(gray_im,cmap='gray')

    f_factor_raanan = hdr_image_util.get_new_brightness_factor(rgb_img) * 256
    print("f_factor_raanan", f_factor_raanan) #2048.000000000003
    b = (gray_im / gray_im.max()) * f_factor_raanan
    # print("before",gray_im.max(), gray_im.min())
    a = np.log10(b + 1)
    print("after",a.max(), a.min(), a.mean())
    a = a / a.max()
    plt.subplot(3,1,2)
    plt.imshow(a, cmap='gray')

    f_factor_old = hdr_image_util.get_brightness_factor(gray_im, mean_target=160, factor=1) * 256
    print("f_factor_old", f_factor_old) #3034315.0695321183
    a = np.log10((gray_im / gray_im.max()) * f_factor_old + 1)
    a = a / a.max()
    plt.subplot(3, 1, 3)
    plt.imshow(a, cmap='gray')
    plt.show()
def c_log_test():
    pass

def regex_test():
    import re
    info = "DATA: 10.61"
    info = "DATA_min_log_1.0new_f_"
    info = "G_unet_ssr_relu_doubleConvT"
    # items = re.findall("DATA_\w*(\d+\.*\d+)", info)
    # items = re.findall("DATA_(\w*)_\d+\.*\d+", info)
    items = re.findall("G_unet_(\w*)_\w*", info)
    print(items)
    print(items[0])  # 10.61

def get_f_from_book(gray_im_input):

    print(gray_im_input.max())
    gray_im = np.reshape(gray_im_input, (-1,))
    # gray_im = gray_im_input[gray_im_input > 0]
    im_max = np.percentile(gray_im, 99)
    im_min = np.percentile(gray_im, 1)
    gray_im_input = (gray_im_input / im_max) * 1000
    im_max = np.percentile(gray_im_input, 99)
    im_min = np.percentile(gray_im_input, 1)
    return np.log2(im_max/im_min +1)
    im_mean = np.exp(np.log(gray_im_input).mean())
    print("log_avg", im_mean, "mean", gray_im_input.mean())
    f = (2 * np.log2(im_mean) - np.log2(im_max) - np.log2(im_max)) / \
        (np.log2(im_max) - np.log2(im_min))
    alpha = 0.18 * 4 ** f
    new_gray_im = (alpha / gray_im.mean()) * gray_im_input
    im_max = np.percentile(new_gray_im, 99)
    im_min = np.percentile(new_gray_im, 1)
    plt.subplot(2, 2, 1)
    plt.imshow(gray_im_input, cmap='gray')
    plt.title(gray_im_input.mean())
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.hist(gray_im, rwidth=0.9, color='#607c8e', density=True, bins=255)
    plt.box(on=None)

    plt.subplot(2, 2, 3)
    plt.imshow(new_gray_im, cmap='gray')
    plt.title(new_gray_im.mean())
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.hist(np.reshape(new_gray_im, (-1,)), rwidth=0.9, color='#607c8e', density=True, bins=255)
    plt.box(on=None)

    plt.show()
    return (im_max/im_min)


def f_factor_test(input_images_path):
    import csv

    # with open('/Users/yaelvinker/Documents/MATLAB/CODE_2016TMM2/results/exr_mean.csv', 'w', newline='') as file:
    #     fieldnames = ['file_name', 'mean']
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)
    #
    #     writer.writeheader()
    for img_name in os.listdir(input_images_path):
        im_path = os.path.join(input_images_path, img_name)
        rgb_img = hdr_image_util.read_hdr_image(im_path)
        if np.min(rgb_img) < 0:
            rgb_img = rgb_img + np.abs(np.min(rgb_img))
        gray_im = hdr_image_util.to_gray(rgb_img)
        rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape=False)
        gray_im = hdr_image_util.reshape_image(gray_im, train_reshape=False)
        f_factor = hdr_image_util.get_new_brightness_factor(rgb_img)
        print("f_factor", f_factor)
        f_factor = f_factor * 255 * 0.1
        new_gray_im = gray_im - gray_im.min()
        new_gray_im = np.log10((new_gray_im / np.max(new_gray_im)) * f_factor + 1)
        new_gray_im = new_gray_im / new_gray_im.max()
        plt.subplot(2,1,1)
        plt.imshow(new_gray_im, cmap='gray')
        title = "max[%.4f], mean[%.4f], min[%.f4]" % (new_gray_im.max(), new_gray_im.mean(), new_gray_im.min())
        plt.title(title)
        if new_gray_im.mean() - 0.5 < 0:
            # gray_im2 = new_gray_im
            # print("prev f", f_factor)
            # n_f_factor = f_factor
            # while gray_im2.mean() < 0.5:
            #     n_f_factor = n_f_factor * np.sqrt(2)
            #     gray_im2 = gray_im - gray_im.min()
            #     gray_im2 = np.log10((gray_im2 / np.max(gray_im2)) * n_f_factor + 1)
            #     gray_im2 = gray_im2 / gray_im2.max()
            # print("new f", n_f_factor, n_f_factor/f_factor)

            # new_gray_im = new_gray_im - new_gray_im.mean()
            # new_gray_im = new_gray_im + 0.5
            # new_gray_im = new_gray_im / new_gray_im.max()
            # # new_gray_im = (new_gray_im - new_gray_im.min()) / (new_gray_im.max() - new_gray_im.min())
            # gray_im2 = new_gray_im

            dist = 0.5 - new_gray_im.mean()
            print("dist", dist)
            gray_im2 = gray_im - gray_im.min()
            gray_im2 = (gray_im2 / gray_im2.max()) * f_factor * (f_factor ** dist)
            print("mean", gray_im2.mean(), "f_factor", f_factor)
            # gray_im2 = (gray_im2 - gray_im2.mean()) + 0.18*f_factor
            print("mean should be [%.4f] and is [%.4f]" % (0.18*f_factor, gray_im2.mean()))
            # print(np.log(0.18*255))
            gray_im2 = np.log10(gray_im2 + 1)
            gray_im2 = gray_im2 / gray_im2.max()
            print("mean should be [%.4f] and is [%.4f]" % (0.5, gray_im2.mean()))

            # gray_im2 += 0.5
            # gray_im2 = gray_im2 / gray_im2.max()
            plt.subplot(2, 1, 2)
            plt.imshow(gray_im2, cmap='gray')
            title = "max[%.4f], mean[%.4f], min[%.4f]" % (gray_im2.max(), gray_im2.mean(), gray_im2.min())
            plt.title(title)
        plt.show()

        # print("\n",img_name)
        # print("our_f", f_factor)
        # # f2 = get_f_from_book(gray_im)
        # f2 = hdr_image_util.get_brightness_factor(gray_im, mean_target=100, factor=1.5)
        # print("f2", f2)
        # print("our/f2", f_factor/f2)

        # writer.writerow({'file_name': os.path.splitext(img_name)[0], 'mean': m})

def save_exr_means(input_images_path):
    import csv

    with open('/Users/yaelvinker/Documents/MATLAB/CODE_2016TMM2/results/exr_mean.csv', 'w', newline='') as file:
        fieldnames = ['file_name', 'mean']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for img_name in os.listdir(input_images_path):
            im_path = os.path.join(input_images_path, img_name)
            rgb_img = hdr_image_util.read_hdr_image(im_path)
            if np.min(rgb_img) < 0:
                rgb_img = rgb_img + np.abs(np.min(rgb_img))
            gray_im = hdr_image_util.to_gray(rgb_img)
            rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape=False)
            gray_im = hdr_image_util.reshape_image(gray_im, train_reshape=False)
            f_factor = hdr_image_util.get_new_brightness_factor(rgb_img) * 255 * 0.1
            new_gray_im = gray_im - gray_im.min()
            new_gray_im = np.log10((new_gray_im / np.max(new_gray_im)) * f_factor + 1)
            new_gray_im = new_gray_im / new_gray_im.max()
            print(os.path.splitext(img_name)[0], new_gray_im.mean())
            writer.writerow({'file_name': os.path.splitext(img_name)[0], 'mean': new_gray_im.mean()})

def f_test_trained_model(model_path, model_name, input_images_path):
    from utils import model_save_util
    start0 = time.time()
    net_path = os.path.join(model_path, model_name, "models", "net_epoch_320.pth")
    model_params = model_save_util.get_model_params(model_name)
    # model_params["factor_coeff"] = 0.5
    print("===============================================")
    print(model_name)
    print(model_params)
    # f_factor_path = os.path.join("/Users/yaelvinker/Documents/university/lab/July/baseline/stretch_1.05data10_d1.0_gamma_ssim2.0_1,2,3_gamma_factor_loss_bilateral1.0_8,4,1_wind5_bmu1.0_sigr0.07_log0.8_eps1e-05_alpha0.5_mu_loss2.0_1,1,1_unet_square_and_square_root_d_model_patchD/test_factors.npy")
    f_factor_path = "none"
    output_images_path = os.path.join(model_path, model_name, "color_STRETCH_2_avg")
    # output_images_path = "/Users/yaelvinker/Downloads/input_images/our_orion"
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model_save_util.run_model_on_path(model_params, device, net_path, input_images_path,
                      output_images_path, "npy", f_factor_path, None, True)
    print(time.time()-start0)

def run_model_on_folder(models_path):
    input_images_path = os.path.join("/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr")
    for model_name in os.listdir(models_path):
        f_test_trained_model(models_path, model_name, input_images_path)




if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Parser for gan network")
    # parser.add_argument("--model_name", type=str)
    # parser.add_argument("--input_path", type=str)
    # args = parser.parse_args()
    # model_name = args.model_name
    models_path = "/Users/yaelvinker/Documents/university/lab/Oct/10_11"
    #    input_path = os.path.join("/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_format_fixed_size")
    # input_path = args.input_path
    input_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
    #    input_path=os.path.join("/cs/labs/raananf/yael_vinker/data/quality_assesment/jass_cab")
    #   input_path=os.path.join("/cs/labs/raananf/yael_vinker/data/quality_assesment/exr_hdr_format_pfstool")
    #    input_path = os.path.join("/cs/labs/raananf/yael_vinker/data/quality_assesment/from_openEXR_data")

    #    run_model_on_folder(models_path)
    model_name = "D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT__pretrain50_lr_g1e-05_d1e-05_decay50_noframe__LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_0.1new_f_"
    f_test_trained_model(models_path, model_name, input_path)
    # regex_test()

    # # import argparse
    # # save_exr_means("/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/exr_format_fixed_size/")
    # # f_factor_test("/Users/yaelvinker/Documents/university/lab/open_exr_fixed_size/exr_format_fixed_size/")
    # # f_factor_test("/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data/")
    # model_path = "/Users/yaelvinker/Documents/university/lab/Sep/09_02_summary/09_03_crop_test/"
    # model_name = "crop_D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT__pretrain50_lr_g1e-05_d1e-05_decay50_noframe_stretch_1.05_LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_0.1new_f_"
    # model_name = "crop_D_multiLayerD_simpleD__num_D3_0.8,0.5,0_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT__pretrain50_lr_g1e-05_d1e-05_decay50_noframe_stretch_1.05_LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_0.1new_f_"

    # model_path = "/Users/yaelvinker/Documents/university/lab/Oct/10_08_summary/single_random_seed/good/rseed_[1,1,1_0.8,0.5,0]"
    # model_name = "D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT__pretrain50_lr_g1e-05_d1e-05_decay50_noframe__LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_10.0contrast_ratio_f_"
    # input_images_path = "/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
    # f_test_trained_model(model_path, model_name, input_images_path)

    # model_path = "/Users/yaelvinker/Documents/university/lab/Oct/10_08_summary/single_random_seed/good/single_rseed"
    # model_name = "1_D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT___manualD_single_[1,1,1_0.5,0.5,0.5]__rseed_Truepretrain50_lr_g1e-05_d1e-05_decay50_noframe__LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_0.1new_f_"
    # input_images_path = "/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
    # f_test_trained_model(model_path, model_name, input_images_path)

    #
    # model_path="/Users/yaelvinker/Documents/university/lab/Oct/10_08_summary/unet_concat/good/[1,1,1_1,1,0.1]/"
    # model_name="D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssrMD_relu_doubleConvT___manualD_double_[1,1,1_1,1,0.1]_pretrain50_lr_g1e-05_d1e-05_decay50_noframe__LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_0.1new_f_"
    # input_images_path = "/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
    # f_test_trained_model(model_path, model_name, input_images_path)




    # run_model_on_folder(models_path)
    # model_name = "D_multiLayerD_simpleD__num_D3ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT__pretrain50_lr_g1e-05_d1e-05_decay50_noframe_stretch_1.05_LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_1.0new_f_"
    # f_test_trained_model(model_name)
    # regex_test()
    # from utils import hdr_image_util
    # im_path = "/Users/yaelvinker/Documents/university/lab/Sep/09_02_summary/09_03_crop_test/crop_D_multiLayerD_simpleD__num_D3_0.8,0.5,0_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT__pretrain50_lr_g1e-05_d1e-05_decay50_noframe_stretch_1.05_LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_0.1new_f_/model_results/factor_coeff_0.3/belgiumgray_stretch.png"
    # # im_path1 = ""
    # # rgb_img = hdr_image_util.read_hdr_image(im_path1)
    # # gray_im = hdr_image_util.to_gray(rgb_img)
    # gray_im = imageio.imread(im_path).astype('float32')
    # # f_factor = hdr_image_util.get_new_brightness_factor(rgb_img)
    # # brightness_factor = f_factor * 255 * 0.1
    # # gray_im = gray_im - gray_im.min()
    # # gray_im = np.log10((gray_im / np.max(gray_im)) * brightness_factor + 1)
    # # gray_im = gray_im / gray_im.max()
    # gray_im1 = hdr_image_util.reshape_im(gray_im, 256, 256)
    # plt.subplot(1,3,1)
    # plt.imshow(gray_im1, cmap='gray')
    # plt.subplot(1,3,2)
    # gray_im1 = hdr_image_util.reshape_im(gray_im, 128, 128)
    # plt.imshow(gray_im1, cmap='gray')
    # plt.subplot(1,3,3)
    # gray_im1 = hdr_image_util.reshape_im(gray_im, 64, 64)
    # plt.imshow(gray_im1, cmap='gray')
    # plt.show()
    # # def reshape_im(im, new_y, new_x):
    #     return skimage.transform.resize(im, (new_y, new_x),
    #                                     mode='reflect', preserve_range=False,
    #                                     anti_aliasing=True, order=3).astype("float32")
