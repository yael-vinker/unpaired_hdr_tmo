from __future__ import print_function

import sys

sys.path.append('/cs/labs/raananf/yael_vinker/01_12/code')
import argparse
import os
import tranforms as transforms_
import numpy as np
import hdr_image_utils
import matplotlib.pyplot as plt
import utils.hdr_image_util as hdr_image_util
# matplotlib.use("Agg")
from os import path
from shutil import copyfile


def display_tensor(tensor_im, isgray):
    np_im = np.array(tensor_im.permute(1, 2, 0))
    im = (np_im - np.min(np_im)) / (np.max(np_im) - np.min(np_im))
    if isgray:
        gray = np.squeeze(im)
        plt.imshow(gray, cmap='gray')
    else:
        plt.imshow(im)
    plt.show()


def print_result(output_dir):
    for img_name in os.listdir(output_dir):
        im_path = os.path.join(output_dir, img_name)
        data = np.load(im_path, allow_pickle=True)
        input_im = data[()]["input_image"]
        color_im = data[()]["display_image"]
        hdr_image_utils.print_tensor_details(input_im, "input_im " + img_name)
        display_tensor(input_im, True)
        hdr_image_utils.print_tensor_details(color_im, "display_image " + img_name)
        display_tensor(color_im, False)


def create_dict_data_log(input_dir, output_dir, isLdr, log_factor_):
    for img_name, i in zip(os.listdir(input_dir), range(2)):
        # for img_name, i in zip(os.listdir(input_dir), range(len(os.listdir(input_dir)))):
        im_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + "_" + str(log_factor_) + '.npy')
        if not path.exists(output_path):
            if isLdr:
                rgb_img = hdr_image_util.read_ldr_image(im_path)
                output_im = hdr_image_util.to_gray(rgb_img)

            else:
                rgb_img = hdr_image_util.read_hdr_image(im_path)
                rgb_img_log = hdr_image_util.hdr_log_loader_factorize(im_path, log_factor_)
                output_im = hdr_image_util.to_gray(rgb_img_log)

            transformed_output_im = transforms_.gray_image_transform(output_im)
            transformed_display_im = transforms_.rgb_display_image_transform(rgb_img)
            data = {'input_image': transformed_output_im, 'display_image': transformed_display_im}
            np.save(output_path, data)
            print(output_path)
        print(i)


def hdr_log_loader_factorize(im_hdr, range_factor, brightness_factor):
    im_hdr = im_hdr / np.max(im_hdr)
    total_factor = range_factor * brightness_factor * 255
    image_new_range = im_hdr * total_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(total_factor + 1)).astype('float32')
    return im


def create_dict_data_log_factorised(input_dir, output_dir, isLdr, log_factor_):
    for img_name, i in zip(os.listdir(input_dir), range(2)):
        im_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + "_" + str(log_factor_) + '.npy')
        if not path.exists(output_path):
            if isLdr:
                rgb_img = hdr_image_util.read_ldr_image(im_path)
                output_im = hdr_image_util.to_gray(rgb_img)

            else:
                rgb_img = hdr_image_util.read_hdr_image(im_path)
                brightness_factor = get_brightness_factor(rgb_img)
                print(brightness_factor)
                rgb_img_log = hdr_log_loader_factorize(rgb_img, log_factor_, brightness_factor)
                output_im = hdr_image_util.to_gray(rgb_img_log)

            transformed_output_im = transforms_.gray_image_transform(output_im)
            transformed_display_im = transforms_.rgb_display_image_transform(rgb_img)
            data = {'input_image': transformed_output_im, 'display_image': transformed_display_im}
            np.save(output_path, data)
            print(output_path)
        print(i)

def create_dict_data_hard_log_factorised(input_dir, output_dir, isLdr, log_factor_):
    for img_name, i in zip(os.listdir(input_dir), range(len(os.listdir(input_dir)))):
        im_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + "_" + str(log_factor_) + '.npy')
        if not path.exists(output_path):
            if isLdr:
                rgb_img = hdr_image_util.read_ldr_image(im_path)
                output_im = hdr_image_util.to_gray(rgb_img)

            else:
                rgb_img = hdr_image_util.read_hdr_image(im_path)
                small_rgb_im = hdr_image_util.reshape_image(rgb_img)
                brightness_factor = get_brightness_factor(small_rgb_im)
                print(brightness_factor)
                rgb_img_log = hdr_log_loader_factorize(rgb_img, log_factor_, brightness_factor)
                output_im = hdr_image_util.to_gray(rgb_img_log)

            transformed_output_im = transforms_.gray_image_transform(output_im)
            transformed_display_im = transforms_.rgb_display_image_transform(rgb_img)
            data = {'input_image': transformed_output_im, 'display_image': transformed_display_im}
            np.save(output_path, data)
            print(output_path)
        print(i)

def split_test_data(output_train_dir, output_test_dir, selected_test_images, log_factor):
    for test_im_name in selected_test_images:
        test_im_name_log = os.path.splitext(test_im_name)[0] + "_" + str(log_factor)
        for img_name in os.listdir(output_train_dir):
            im_path = os.path.join(output_train_dir, img_name)
            im_pref = os.path.splitext(img_name)[0]
            if test_im_name_log == im_pref:
                print(test_im_name_log)
                im_new_path = os.path.join(output_test_dir, img_name)
                os.rename(im_path, im_new_path)
                break


def get_bump(im):
    tmp = im
    tmp[(tmp > 200) != 0] = 255
    tmp = tmp.astype('uint8')

    hist, bins = np.histogram(tmp, bins=255)

    a0 = np.mean(hist[0:64])
    a1 = np.mean(hist[65:200])
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


def display_with_bf(im, bf):
    im1 = hdr_log_loader_factorize(im, 1, 1)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(im1)
    plt.subplot(1, 2, 2)
    plt.axis("off")
    new_im = hdr_log_loader_factorize(im, 1, bf)
    # new_im = (new_im / np.max(new_im) * 255).astype('uint8')
    plt.imshow(new_im)
    plt.show()
    plt.close()


def save_brightness_factor_for_image(input_dir, output_dir, isLdr):
    # brightness_factors = [29.17062352545879, 125.94695101928836]
    data = {}
    output_path = os.path.join(output_dir, 'brightness_factors.npy')
    for img_name, i in zip(os.listdir(input_dir), range(len(os.listdir(input_dir)))):
        print(i)
        im_path = os.path.join(input_dir, img_name)
        if isLdr:
            rgb_img = hdr_image_util.read_ldr_image(im_path)
        else:
            rgb_img = hdr_image_util.read_hdr_image(im_path)
        brightness_factor = get_brightness_factor(rgb_img)
        print(brightness_factor)
        display_with_bf(rgb_img, brightness_factor)
        data[img_name] = brightness_factor
        np.save(output_path, data)
    np.save(output_path, data)


def save_images_from_existing_path(existing_samples_path, input_path, output_path):
    for img_name in os.listdir(existing_samples_path):
        print(img_name[:-9])
        img_name = img_name[:-9]
        name_pref = os.path.splitext(img_name)[0]
        im_path = os.path.join(input_path, name_pref + ".npy")
        if os.path.exists(im_path):
            output_im_path = os.path.join(output_path, img_name + ".npy")
            if not os.path.exists(output_im_path):
                copyfile(im_path, output_im_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--root_hdr", type=str, default=os.path.join("/cs/labs/raananf/yael_vinker/dng_collection"))
    parser.add_argument("--root_ldr", type=str, default=os.path.join("/cs/dataset/flickr30k/images"))
    parser.add_argument("--output_hdr", type=str,
                        default=os.path.join("/cs/labs/raananf/yael_vinker/01_12/code/data/hdr_log_data/hdr_log_data"))
    parser.add_argument("--output_ldr", type=str,
                        default=os.path.join("/cs/labs/raananf/yael_vinker/01_12/code/data/ldr_npy/ldr_npy"))
    parser.add_argument("--hdr_test_dir", type=str, default=os.path.join(
        "/cs/labs/raananf/yael_vinker/data/test/hdrplus_log10/hdrplus_log10"))
    parser.add_argument("--log_factor", type=int, default=1000)

    # parser.add_argument("--root_hdr", type=str, default=os.path.join("data/hdr_data/hdr_data"))
    # parser.add_argument("--root_ldr", type=str, default=os.path.join("/cs/dataset/flickr30k/images"))
    # parser.add_argument("--output_hdr", type=str, default=os.path.join("data/hdr_log_data/hdr_log_data"))
    # parser.add_argument("--output_ldr", type=str, default=os.path.join("/cs/labs/raananf/yael_vinker/data/train/ldr_flicker_dict/ldr_flicker_dict"))
    args = parser.parse_args()
    input_hdr_dir = os.path.join(args.root_hdr)
    input_ldr_dir = os.path.join(args.root_ldr)
    output_hdr_dir = os.path.join(args.output_hdr)
    output_ldr_dir = os.path.join(args.output_ldr)
    hdr_test_dir = os.path.join(args.hdr_test_dir)
    channels = 1
    images_mean = 0
    b_factor_output_dir = os.path.join("/Users/yaelvinker/PycharmProjects/lab/data_generator/brightness_factors.npy")
    data = np.load(b_factor_output_dir, allow_pickle=True)
    print(len(data[()]))
    input_im = data[()]["merged_0127_20161030_122722_988.dng"]
    print(input_im)
    # save_brightness_factor_for_image(input_hdr_dir, b_factor_output_dir, isLdr=False)
    # log_factor = args.log_factor
    # print("log factor = ", log_factor)
    # # create_dict_data_log_factorised(input_hdr_dir, output_hdr_dir, isLdr=False, log_factor_=log_factor)
    # print_result(output_saccthdr_dir)
    # selected_test_images_list = ['merged_JN34_20150324_141124_427', 'merged_9bf4_20150818_173321_675',
    #                              'merged_6G7M_20150328_160426_633', 'merged_0155_20160817_141742_989',
    #                              'merged_0043_20160831_082253_094', 'merged_0030_20151008_090054_301',
    #                              'merged_4742_20150918_185622_484', 'belgium', 'cathedral', 'synagogue',
    #                              'merged_0006_20160724_095820_954', 'merged_0009_20160702_164047_896']
    # split_test_data(output_hdr_dir, hdr_test_dir, selected_test_images_list, log_factor)
    # print_result(output_hdr_dir)
