from __future__ import print_function
import sys
import inspect
import os
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import argparse
import os
import tranforms as transforms_
import numpy as np
import matplotlib.pyplot as plt
import utils.hdr_image_util as hdr_image_util
from shutil import copyfile


def display_tensor(tensor_im, isgray, im_name=""):
    # save_gray_tensor_as_numpy(tensor_im, "/Users/yaelvinker/PycharmProjects/lab/utils/hdrplus_gamma_use_factorise_data_1_factor_coeff_1.0/fix2/",
    #                           os.path.splitext(im_name)[0] + "_input.png")
    np_im = np.array(tensor_im.permute(1, 2, 0))
    # hdr_image_util.print_image_details(np_im, os.path.splitext(im_name)[0])
    # im = (np_im - np.min(np_im)) / (np.max(np_im) - np.min(np_im))
    im = np_im
    if isgray:
        gray = np.squeeze(im)
        plt.imshow(gray, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(im, vmin=0, vmax=1)
    plt.show()


def save_gray_tensor_as_numpy(tensor, output_path, im_name):
    import imageio
    tensor = tensor.clamp(0, 1)
    tensor = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    # tensor_0_1 = np.squeeze(hdr_image_util.to_0_1_range(tensor))
    tensor_0_1 = np.squeeze(tensor)
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name), im, format='PNG-FI')


def print_result(output_dir):
    total_mean = 0
    for img_name in os.listdir(output_dir):
        im_path = os.path.join(output_dir, img_name)
        data = np.load(im_path, allow_pickle=True)
        input_im = data[()]["input_image"]
        color_im = data[()]["display_image"]
        total_mean += (input_im.mean())
        hdr_image_util.print_tensor_details(input_im, "input_im " + img_name)
        display_tensor(input_im, True, img_name)
        hdr_image_util.print_tensor_details(color_im / color_im.max(), "display_image " + img_name)
        hdr_image_util.print_tensor_details(color_im, "display_image " + img_name)
        display_tensor(color_im / color_im.max(), False)
    print((total_mean * 255) / len(os.listdir(output_dir)))



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
            rgb_img = rgb_img + np.abs(np.min(rgb_img))
        else:
            print("not neg")
        rgb_img = skimage.transform.resize(rgb_img, (int(rgb_img.shape[0] / 2),
                                                           int(rgb_img.shape[1] / 2)),
                                                  mode='reflect', preserve_range=False, order=3).astype("float32")
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
    rgb_img = transforms_.image_transform_no_norm(rgb_img)
    gray_im = transforms_.image_transform_no_norm(gray_im)
    return rgb_img, gray_im


def hdr_sigma_preprocess(im_path, args, reshape=False):
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
    return rgb_img, gray_im_log


def get_mean_and_factor(gamma_log, use_new_f):
    if use_new_f:
        mean_target = 1
        factor = 1
    elif gamma_log == 2:
        mean_target = 80
        factor = (2/3)
    elif gamma_log == 10:
        mean_target = 160
        factor = 1.5
    elif gamma_log == 1:
        mean_target = 110
        factor = 1
    else:
        assert 0, "Unsupported gamma log"
    return mean_target, factor


def get_f(use_new_f, rgb_img, gray_im, mean_target, factor):
    if use_new_f:
        f_factor = hdr_image_util.get_new_brightness_factor(rgb_img)
    else:
        f_factor = hdr_image_util.get_brightness_factor(gray_im, mean_target, factor)
    return f_factor


def hdr_preprocess(im_path, factor_coeff, train_reshape, gamma_log, f_factor_path, use_new_f):
    mean_target, factor = get_mean_and_factor(gamma_log, use_new_f)
    print("mean_target, factor")
    print(mean_target, factor)
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    gray_im = hdr_image_util.to_gray(rgb_img)
    rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape)
    gray_im = hdr_image_util.reshape_image(gray_im, train_reshape)
    if f_factor_path != "none":
        data = np.load(f_factor_path, allow_pickle=True)
        if os.path.basename(im_path) in data[()]:
            f_factor = data[()][os.path.basename(im_path)]
        else:
            f_factor = get_f(use_new_f, rgb_img, gray_im, mean_target, factor)
    else:
        f_factor = get_f(use_new_f, rgb_img, gray_im, mean_target, factor)
    brightness_factor = f_factor * 255 * factor_coeff
    print("brightness_factor", brightness_factor)
    gamma = (1 / (1 + factor * np.log10(brightness_factor)))
    gray_im = (gray_im / np.max(gray_im)) ** gamma
    return rgb_img, gray_im, gamma


def hdr_preprocess_change_f(im_path, args, f_new, reshape=True):
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if reshape:
        rgb_img = hdr_image_util.reshape_image(rgb_img)
    gray_im = hdr_image_util.to_gray(rgb_img)
    if args.use_factorise_data:
        gray_im_temp = hdr_image_util.reshape_im(gray_im, 128, 128)
        brightness_factor = hdr_image_util.get_brightness_factor(gray_im_temp) * 255 * f_new
        print(brightness_factor)
    else:
        # factor is log_factor 1000
        brightness_factor = 1000
    gray_im = (gray_im / np.max(gray_im)) * brightness_factor
    gray_im_log = np.log(gray_im + 1)
    return rgb_img, gray_im_log


def apply_preprocess_for_hdr(im_path, args):
    rgb_img, gray_im_log, gamma_factor = hdr_preprocess(im_path,
                                          factor_coeff=args.factor_coeff,
                                          train_reshape=True, gamma_log=args.gamma_log,
                                                        f_factor_path=args.f_factor_path,
                                                        use_new_f=args.use_new_f)
    rgb_img = transforms_.image_transform_no_norm(rgb_img)
    gray_im_log = transforms_.image_transform_no_norm(gray_im_log)
    return rgb_img, gray_im_log, gamma_factor


def apply_window_tone_map_for_hdr(im_path, args=None):
    import imageio
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    import old_files.transfer_to_ldr.transfer_to_ldr as run_window_tone_map
    rgb_im_tone_map = run_window_tone_map.run_single_window_tone_map(im_path, reshape=False)
    gray_im = hdr_image_util.to_gray(rgb_im_tone_map)

    output_path_gray = os.path.join(args.output_dir, os.path.splitext(args.img_name)[0] + '.png')
    output_path_color = os.path.join(args.output_dir, os.path.splitext(args.img_name)[0] + '_color.png')
    imageio.imwrite(output_path_gray, gray_im)
    print(output_path_gray)
    imageio.imwrite(output_path_color, rgb_im_tone_map)
    return rgb_img, gray_im


def create_data(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    gamma_factor = 0
    for img_name, i in zip(os.listdir(input_dir), range(args.number_of_images)):
        args.img_name = img_name
        im_path = os.path.join(input_dir, img_name)
        if args.isLdr:
            rgb_img, gray_im = apply_preprocess_for_ldr(im_path)
        else:
            rgb_img, gray_im, gamma_factor = apply_preprocess_for_hdr(im_path, args)
        data = {'input_image': gray_im, 'display_image': rgb_img, 'gamma_factor': gamma_factor}
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        np.save(output_path, data)
        print(output_path)
        print(i)


def save_f_factor(input_dir, output_dir, train_reshape, dict_name):
    output_path = os.path.join(output_dir, dict_name)
    print(output_path)
    f_factor_dict = {}
    counter = 0
    for img_name in os.listdir(input_dir):
        im_path = os.path.join(input_dir, img_name)
        rgb_img = hdr_image_util.read_hdr_image(im_path)
        if np.min(rgb_img) < 0:
            rgb_img = rgb_img + np.abs(np.min(rgb_img))
        rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape)
        f_factor = hdr_image_util.get_new_brightness_factor(rgb_img)
        f_factor_dict[img_name] = f_factor
        np.save(output_path, f_factor_dict)
        print("[%d] [%s] [%.2f]" % (counter, img_name, f_factor))
        counter += 1



def add_f_factor_to_data(input_dir, f_factor_path, output_dir, number_of_images):
    gamma_factors = np.load(f_factor_path, allow_pickle=True)
    print(output_dir)
    for img_name, i in zip(os.listdir(input_dir), range(number_of_images)):
        im_path = os.path.join(input_dir, img_name)
        data = np.load(im_path, allow_pickle=True)
        data[()]["gamma_factor"] = gamma_factors[os.path.splitext(img_name)[0]]
        output_path = os.path.join(output_dir, img_name)
        np.save(output_path, data)
        test_a = np.load(output_path, allow_pickle=True)[()]
        print(test_a)


def save_exr_f_factors(input_images_path, output_path, mean_target, factor):
    f_factors = {}
    dirs = os.listdir(input_images_path)
    for i in range(len(dirs)):
        img_name = dirs[i]
        im_path = os.path.join(input_images_path, img_name)
        rgb_img = hdr_image_util.read_hdr_image(im_path)
        if np.min(rgb_img) < 0:
            rgb_img = rgb_img + np.abs(np.min(rgb_img))
        gray_im = hdr_image_util.to_gray(rgb_img)
        gray_im = hdr_image_util.reshape_image(gray_im, train_reshape=False)
        f_factor = hdr_image_util.get_brightness_factor(gray_im, mean_target, factor)
        f_factors[img_name] = f_factor
        print("[%d] [%f] %s" % (i, f_factor, img_name))
        np.save(output_path, f_factors)


if __name__ == '__main__':
    # input_dir = "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_format_fixed_size"
    # output_dir = "/cs/snapless/raananf/yael_vinker/data/open_exr_source"
    # train_reshape = False
    # dict_name = "exr_newf.npy"
    # save_f_factor(input_dir, output_dir, train_reshape, dict_name)
    # save_exr_f_factors("/cs/snapless/raananf/yael_vinker/data/dng_data_fid",
    #                    "/cs/snapless/raananf/yael_vinker/data/dng_data_fid.npy",
    #                    0, 1.5)
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--input_dir", type=str, default="/Users/yaelvinker/PycharmProjects/lab/utils/folders/hdr_data")
    parser.add_argument("--output_dir_pref", type=str, default="/Users/yaelvinker/PycharmProjects/lab/data_generator/res_test")
    parser.add_argument("--isLdr", type=int, default=0)
    parser.add_argument("--number_of_images", type=int, default=7)
    parser.add_argument("--use_factorise_data", type=int, default=1)  # bool
    parser.add_argument("--factor_coeff", type=float, default=1.0)
    parser.add_argument("--gamma_log", type=int, default=2)
    parser.add_argument("--f_factor_path", type=str, default="none")
    parser.add_argument("--use_new_f", type=int, default=0)

    args = parser.parse_args()
    if args.isLdr:
       pref = "flicker"
    else:
       pref = "hdrplus"
    output_dir_name = pref #+ "_new_f_" + str(args.use_new_f)
    args.output_dir = os.path.join(args.output_dir_pref, output_dir_name)
    if not os.path.exists(args.output_dir):
       os.mkdir(args.output_dir)
    create_data(args)
    # print_result(args.output_dir)
    # save_f_factor(args)
    # split_train_test_data("/cs/snapless/raananf/yael_vinker/data/new_data/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1",
    #                       "/cs/snapless/raananf/yael_vinker/data/new_data/train/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1",
    #                       "/cs/snapless/raananf/yael_vinker/data/new_data/test/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1")