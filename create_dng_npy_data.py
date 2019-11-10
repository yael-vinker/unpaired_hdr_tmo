from __future__ import print_function
import argparse
import os
import tranforms as transforms_
import numpy as np
import hdr_image_utils
import matplotlib.pyplot as plt
import utils.hdr_image_util as hdr_image_util
from os import path

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
    for img_name, i in zip(os.listdir(input_dir), range(910)):
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--root_hdr", type=str, default=os.path.join("/cs/labs/raananf/yael_vinker/dng_collection"))
    parser.add_argument("--root_ldr", type=str, default=os.path.join("/cs/dataset/flickr30k/images"))
    parser.add_argument("--output_hdr", type=str, default=os.path.join("/cs/labs/raananf/yael_vinker/data/train/hdrplus_log10/hdrplus_log10"))
    parser.add_argument("--output_ldr", type=str, default=os.path.join("/cs/labs/raananf/yael_vinker/data/train/ldr_flicker_dict/ldr_flicker_dict"))
    parser.add_argument("--hdr_test_dir", type=str, default=os.path.join(
        "/cs/labs/raananf/yael_vinker/data/test/hdrplus_log10/hdrplus_log10"))
    parser.add_argument("--log_factor", type=int, default=1)

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
    log_factor = args.log_factor
    print("log factor = ", log_factor)
    # create_dict_data_log(input_hdr_dir, output_hdr_dir, False, log_factor)
    # print_result(output_hdr_dir)
    selected_test_images_list = ['merged_JN34_20150324_141124_427', 'merged_9bf4_20150818_173321_675',
                                 'merged_6G7M_20150328_160426_633', 'merged_0155_20160817_141742_989',
                                 'merged_0043_20160831_082253_094', 'merged_0030_20151008_090054_301',
                                 'merged_4742_20150918_185622_484', 'belgium', 'cathedral', 'synagogue',
                                 'merged_0006_20160724_095820_954', 'merged_0009_20160702_164047_896']
    split_test_data(output_hdr_dir, hdr_test_dir, selected_test_images_list, log_factor)
    # print_result(output_hdr_dir)