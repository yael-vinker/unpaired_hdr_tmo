from __future__ import print_function
import argparse
import os
import tranforms as transforms_
import numpy as np
import matplotlib.pyplot as plt
import utils.hdr_image_util as hdr_image_util
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
        hdr_image_util.print_tensor_details(input_im, "input_im " + img_name)
        display_tensor(input_im, True)
        hdr_image_util.print_tensor_details(color_im, "display_image " + img_name)
        display_tensor(color_im, False)


def hdr_log_loader_factorize(im_hdr, range_factor, brightness_factor):
    im_hdr = im_hdr / np.max(im_hdr)
    total_factor = range_factor * brightness_factor * 255
    image_new_range = im_hdr * total_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(total_factor + 1)).astype('float32')
    return im


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


def display_with_bf(im, bf):
    im1 = hdr_log_loader_factorize(im, 1, 1)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.axis("off")
    plt.imshow(im1)
    plt.subplot(3, 1, 2)
    plt.axis("off")
    new_im = hdr_log_loader_factorize(im, 1, bf)
    # new_im = (new_im / np.max(new_im) * 255).astype('uint8')
    plt.imshow(new_im)
    new_im_01 = hdr_log_loader_factorize(im, 0.1, bf)
    plt.subplot(3, 1, 3)
    plt.axis("off")
    plt.imshow(new_im_01)
    plt.show()
    plt.close()


def save_brightness_factor_for_image(input_dir, output_dir, old_f, isLdr):
    import skimage
    # brightness_factors = [29.17062352545879, 125.94695101928836]
    data = {}
    output_path = os.path.join(output_dir, 'brightness_factors_exr.npy')
    for img_name, i in zip(os.listdir(input_dir), range(len(os.listdir(input_dir)))):
        print(i, img_name)
        im_path = os.path.join(input_dir, img_name)
        if isLdr:
            rgb_img = hdr_image_util.read_ldr_image(im_path)
        else:
            rgb_img = hdr_image_util.read_hdr_image(im_path)
        if np.min(rgb_img) < 0:
            rgb_img = rgb_img - np.min(rgb_img)
        else:
            print("not neg")
        rgb_img = skimage.transform.resize(rgb_img, (int(rgb_img.shape[0] / 2),
                                                           int(rgb_img.shape[1] / 2)),
                                                  mode='reflect', preserve_range=False).astype("float32")
        brightness_factor = hdr_image_util.get_brightness_factor(rgb_img)
        print("new f ",brightness_factor)
        print("old f ", old_f[os.path.splitext(img_name)[0] + ".hdr"])
        # display_with_bf(rgb_img, brightness_factor)
        data[img_name] = brightness_factor
        # np.save(output_path, data)
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


def split_train_test_data(input_path, train_path, test_path, num_train_images=1, num_test_images=1):
    data = os.listdir(input_path)
    # import random
    # random.shuffle(data)
    train_data = data[:896]
    test_data = data[896: 896 + 12]

    for img_name in train_data:
        im_path = os.path.join(input_path, img_name)
        output_im_path = os.path.join(train_path, img_name)
        copyfile(im_path, output_im_path)

    for img_name in test_data:
        im_path = os.path.join(input_path, img_name)
        output_im_path = os.path.join(test_path, img_name)
        copyfile(im_path, output_im_path)


def apply_preprocess_for_ldr(im_path):
    rgb_img = hdr_image_util.read_ldr_image_original_range(im_path)
    gray_im = hdr_image_util.to_gray(rgb_img)
    if args.use_normalization:
        gray_im = hdr_image_util.to_0_1_range(gray_im)
        gray_im = hdr_image_util.to_minus1_1_range(gray_im)
    rgb_img = transforms_.image_transform_no_norm(rgb_img)
    gray_im = transforms_.image_transform_no_norm(gray_im)
    return rgb_img, gray_im


def hdr_preprocess(im_path, args, reshape=False):
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if reshape:
        rgb_img = hdr_image_util.reshape_image(rgb_img)
    gray_im = hdr_image_util.to_gray(rgb_img)
    if args.use_factorise_data:
        gray_im_temp = hdr_image_util.reshape_im(gray_im, 128, 128)
        brightness_factor = hdr_image_util.get_brightness_factor(gray_im_temp) * 255 * args.factor_coeff
        print(brightness_factor)
    else:
        # factor is log_factor 1000
        brightness_factor = 1000
    gray_im = (gray_im / np.max(gray_im)) * brightness_factor
    gray_im_log = np.log(gray_im + 1)
    if args.use_normalization:
        gray_im_log = hdr_image_util.to_0_1_range(gray_im_log)
        gray_im_log = hdr_image_util.to_minus1_1_range(gray_im_log)
    return rgb_img, gray_im_log


def apply_preprocess_for_hdr(im_path, args):
    rgb_img, gray_im_log = hdr_preprocess(im_path, args, reshape=False)
    rgb_img = transforms_.image_transform_no_norm(rgb_img)
    gray_im_log = transforms_.image_transform_no_norm(gray_im_log)
    return rgb_img, gray_im_log


def create_data(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    for img_name, i in zip(os.listdir(input_dir), range(args.number_of_images)):
        im_path = os.path.join(input_dir, img_name)
        if args.isLdr:
            rgb_img, gray_im = apply_preprocess_for_ldr(im_path)
        else:
            rgb_img, gray_im = apply_preprocess_for_hdr(im_path, args)
        data = {'input_image': gray_im, 'display_image': rgb_img}
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        np.save(output_path, data)
        print(output_path)
        print(i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--input_dir", type=str, default="/Users/yaelvinker/PycharmProjects/lab/data/ldr_data/ldr_data")
    parser.add_argument("--output_dir_pref", type=str, default="/Users/yaelvinker/PycharmProjects/lab/data/factorised_data_original_range")
    parser.add_argument("--isLdr", type=int, default=1)
    parser.add_argument("--number_of_images", type=int, default=3)
    parser.add_argument("--use_factorise_data", type=int, default=1)  # bool
    parser.add_argument("--factor_coeff", type=float, default=0.1)
    parser.add_argument("--use_normalization", help='if to change range to [-1, 1]', type=int, default=0)

    args = parser.parse_args()
    if args.isLdr:
        pref = "flicker"
    else:
        pref = "hdrplus"
    output_dir_name = pref + "_use_factorise_data_" + str(args.use_factorise_data) + \
                      "_factor_coeff_" + str(args.factor_coeff) + "_use_normalization_" + str(args.use_normalization)
    args.output_dir = os.path.join(args.output_dir_pref, output_dir_name)
    os.mkdir(args.output_dir)
    create_data(args)
    print_result(args.output_dir)

    # split_train_test_data("/cs/snapless/raananf/yael_vinker/data/new_data/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1",
    #                       "/cs/snapless/raananf/yael_vinker/data/new_data/train/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1",
    #                       "/cs/snapless/raananf/yael_vinker/data/new_data/test/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1")
    #
    # split_train_test_data(
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/flicker_use_factorise_data_1_factor_coeff_0.1_use_normalization_0",
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/train/flicker_use_factorise_data_1_factor_coeff_0.1_use_normalization_0",
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/test/flicker_use_factorise_data_1_factor_coeff_0.1_use_normalization_0")
    #
    # split_train_test_data(
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/hdrplus_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1",
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/train/hdrplus_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1",
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/test/hdrplus_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1")
    #
    # split_train_test_data(
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/hdrplus_use_factorise_data_1_factor_coeff_0.1_use_normalization_0",
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/train/hdrplus_use_factorise_data_1_factor_coeff_0.1_use_normalization_0",
    #     "/cs/snapless/raananf/yael_vinker/data/new_data/test/hdrplus_use_factorise_data_1_factor_coeff_0.1_use_normalization_0")

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
