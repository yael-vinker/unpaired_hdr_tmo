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
    plt.figure()
    if isgray:
        gray = np.squeeze(im)
        plt.imshow(gray, cmap='gray', vmin=0, vmax=1)
        plt.title(im_name)
    else:
        plt.imshow(im, vmin=0, vmax=1)
        plt.title(im_name)
    # plt.show()


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
        gamma_factor = data[()]["gamma_factor"]
        brightness_factor = (10 ** (1 / gamma_factor - 1)) * 1
        total_mean += (input_im.mean())
        print("brightness_factor", brightness_factor / 256)
        hdr_image_util.print_tensor_details(color_im, "input_im " + img_name)

        display_tensor(input_im, True, img_name)
        # hdr_image_util.print_tensor_details(color_im / color_im.max(), "display_image " + img_name)
        # hdr_image_util.print_tensor_details(color_im, "display_image " + img_name)
        display_tensor(color_im / color_im.max(), False)
    print((total_mean * 255) / len(os.listdir(output_dir)))

def print_result_log(output_dir):
    total_mean = 0
    for img_name in os.listdir(output_dir):
        im_path = os.path.join(output_dir, img_name)
        data = np.load(im_path, allow_pickle=True)
        input_im = data[()]["input_image"]
        total_mean += (input_im.mean())
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


def split_train_test_data(input_path, train_path, test_path, test_fid_test):
    data = os.listdir(input_path)
    train_data = data[:896]
    test_data = data[896: 896 + 12]
    test_fid_data = data[896 + 12:]

    for img_name in train_data:
        im_path = os.path.join(input_path, img_name)
        output_im_path = os.path.join(train_path, img_name)
        copyfile(im_path, output_im_path)

    for img_name in test_data:
        im_path = os.path.join(input_path, img_name)
        output_im_path = os.path.join(test_path, img_name)
        copyfile(im_path, output_im_path)

    for img_name in test_fid_data:
        im_path = os.path.join(input_path, img_name)
        output_im_path = os.path.join(test_fid_test, img_name)
        copyfile(im_path, output_im_path)


def copy_train_data_from_old_set(input_path_names, input_images_path, output_path):
    names = os.listdir(input_path_names)
    counter = 446
    while counter < 896:
        for img_name in names:
            print("[%d] %s"%(counter,img_name))
            source_path = os.path.join(input_images_path, img_name)
            if os.path.exists(source_path):
                output_im_path = os.path.join(output_path, img_name)
                if not os.path.exists(output_im_path):
                    copyfile(source_path, output_im_path)
                    counter += 1




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


def get_f(use_new_f, rgb_img, gray_im, mean_target, factor, use_contrast_ratio_f, factor_coeff, f_factor_path, im_name):
    if f_factor_path != "none":
        data = np.load(f_factor_path, allow_pickle=True)
        if im_name in data[()]:
            f_factor = data[()][im_name]
            print("[%s] found in dict [%.4f]" % (im_name, f_factor))
            return f_factor * 255 * factor_coeff
        else:
            f_factor = hdr_image_util.get_new_brightness_factor(rgb_img) * 255 * factor_coeff
    elif use_contrast_ratio_f:
        print("===============")
        print(np.percentile(gray_im, 99.0), np.percentile(gray_im, 99.9), gray_im.max())
        print(np.percentile(rgb_img, 99.0), np.percentile(rgb_img, 99.9), rgb_img.max())
        print(np.percentile(gray_im, 1.0), np.percentile(gray_im, 0.1), gray_im.min())
        im_max = np.percentile(gray_im, 100)
        im_min = np.percentile(gray_im, 1.0)
        if im_min == 0:
            im_min += 0.0001
        f_factor = im_max / im_min * factor_coeff
        return f_factor
    elif use_new_f:
        f_factor = hdr_image_util.get_new_brightness_factor(rgb_img) * 255 * factor_coeff
        return f_factor
    else:
        f_factor = hdr_image_util.get_brightness_factor(gray_im, mean_target, factor) * 255 * factor_coeff
    return f_factor


def fix_im_avg(gray_im, f_factor, brightness_factor):
    new_gray_im = np.log10((gray_im / np.max(gray_im)) * brightness_factor + 1)
    new_gray_im = new_gray_im / new_gray_im.max()
    if new_gray_im.mean() < 0.5:
        dist = 0.6 - new_gray_im.mean()
        print("dist", dist)
        gray_im2 = gray_im
        gray_im2 = (gray_im2 / gray_im2.max()) * brightness_factor * (brightness_factor ** dist)
        gray_im2 = np.log10(gray_im2 + 1)
        gray_im2 = gray_im2 / gray_im2.max()
        return gray_im2
    return new_gray_im


def hdr_preprocess(im_path, factor_coeff, train_reshape, gamma_log, f_factor_path, use_new_f, data_trc,
                   test_mode=False, use_contrast_ratio_f=False):
    mean_target, factor = get_mean_and_factor(gamma_log, use_new_f)
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    # print(rgb_img.max(), rgb_img.min(), rgb_img.mean())
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    gray_im = hdr_image_util.to_gray(rgb_img)
    rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape)
    gray_im = hdr_image_util.reshape_image(gray_im, train_reshape)
    im_name = os.path.splitext(os.path.basename(im_path))[0]
    f_factor = get_f(use_new_f, rgb_img, gray_im, mean_target, factor, use_contrast_ratio_f,
                     factor_coeff, f_factor_path, im_name)
    if f_factor < 1:
        print("==== %s ===== buggy im" % im_path)
    # print("f_factor", f_factor)
    brightness_factor = f_factor
    print("brightness_factor", brightness_factor)
    if "min" in data_trc:
        gray_im = gray_im - gray_im.min()
    gamma = (1 / (1 + factor * np.log10(brightness_factor)))
    if "log" in data_trc:
        if test_mode:
            gray_im = fix_im_avg(gray_im, f_factor, brightness_factor)
        else:
            gray_im = np.log10((gray_im / np.max(gray_im)) * brightness_factor + 1)
            gray_im = gray_im / gray_im.max()
    elif "gamma" in data_trc:
        gray_im = (gray_im / np.max(gray_im)) ** gamma
    return rgb_img, gray_im, gamma


def hdr_preprocess_crop_data(im_path, factor_coeff, train_reshape, gamma_log, f_factor_path, use_new_f, data_trc):
    mean_target, factor = get_mean_and_factor(gamma_log, use_new_f)
    use_contrast_ratio_f = False
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.abs(np.min(rgb_img))
    gray_im = hdr_image_util.to_gray(rgb_img)

    im_name = os.path.splitext(os.path.basename(im_path))[0]
    f_factor = get_f(use_new_f, rgb_img, gray_im, mean_target, factor, use_contrast_ratio_f,
                     factor_coeff, f_factor_path, im_name)
    if f_factor < 1:
        print("==== %s ===== buggy im" % im_path)
    h, w = gray_im.shape[0], gray_im.shape[1]
    crop_h, crop_w = max(h // 2, 256), max(w // 2, 256)
    if h > w:
        gray_im = gray_im[: crop_h, :]
        rgb_img = rgb_img[: crop_h, :, :]
    else:
        gray_im = gray_im[:, :crop_w]
        rgb_img = rgb_img[:, :crop_w, :]
    rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape)

    gray_im = hdr_image_util.reshape_image(gray_im, train_reshape)
    brightness_factor = f_factor * 255 * factor_coeff
    print("brightness_factor", brightness_factor)
    if "min" in data_trc:
        gray_im = gray_im - gray_im.min()
    gamma = (1 / (1 + factor * np.log10(brightness_factor)))
    if "log" in data_trc:
        gray_im = np.log10((gray_im / np.max(gray_im)) * brightness_factor + 1)
        gray_im = gray_im / gray_im.max()
    elif "gamma" in data_trc:
        gray_im = (gray_im / np.max(gray_im)) ** gamma
    plt.imshow(gray_im)
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
    if args.crop_data:
        rgb_img, gray_im_log, gamma_factor = hdr_preprocess_crop_data(im_path,
                                                            factor_coeff=args.factor_coeff,
                                                            train_reshape=True, gamma_log=args.gamma_log,
                                                            f_factor_path=args.f_factor_path,
                                                            use_new_f=args.use_new_f,
                                                            data_trc=args.data_trc)
    else:
        rgb_img, gray_im_log, gamma_factor = hdr_preprocess(im_path,
                                          factor_coeff=args.factor_coeff,
                                          train_reshape=True, gamma_log=args.gamma_log,
                                                        f_factor_path=args.f_factor_path,
                                                        use_new_f=args.use_new_f,
                                                        data_trc=args.data_trc)
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
    names_path = args.input_dir_names
    if os.path.isdir(names_path):
        input_names = os.listdir(names_path)
    else:
        input_names = np.load(names_path, allow_pickle=True)[()]
    input_names = input_names[:args.number_of_images]
    for img_name, i in zip(input_names, range(args.number_of_images)):
        args.img_name = img_name
        im_path = os.path.join(input_dir, img_name)
        if os.path.exists(im_path):
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


def dng_to_npy_for_fid():
    import skimage
    npy_names_dir = "/cs/snapless/raananf/yael_vinker/data/new_data_crop_fix/test_fid"
    output_dir = "/cs/snapless/raananf/yael_vinker/data/oct_fid_npy/"
    input_dng_files = "/cs/labs/raananf/yael_vinker/dng_collection/"
    for im_name in os.listdir(npy_names_dir):
        im_path = os.path.join(input_dng_files, os.path.splitext(im_name)[0] + ".dng")
        im = hdr_image_util.read_hdr_image(im_path)
        im = skimage.transform.resize(im, (int(im.shape[0] / 2),
                                                   int(im.shape[1] / 2)),
                                          mode='reflect', preserve_range=False,
                                          anti_aliasing=True, order=3).astype("float32")
        print(im_name, im.shape)
        np.save(os.path.join(output_dir, im_name), im)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--input_dir", type=str, default="/Users/yaelvinker/PycharmProjects/lab/utils/folders/data")
    parser.add_argument("--output_dir_pref", type=str, default="/Users/yaelvinker/PycharmProjects/lab/data_generator/res_test")
    parser.add_argument("--isLdr", type=int, default=0)
    parser.add_argument("--number_of_images", type=int, default=7)
    parser.add_argument("--use_factorise_data", type=int, default=1)  # bool
    parser.add_argument("--factor_coeff", type=float, default=1)
    parser.add_argument("--gamma_log", type=int, default=10)
    parser.add_argument("--f_factor_path", type=str, default="none")
    parser.add_argument("--use_new_f", type=int, default=1)
    parser.add_argument("--data_trc", type=str, default="log_min")
    parser.add_argument("--crop_data", type=int, default=1)
    parser.add_argument("--input_dir_names", type=str)

    args = parser.parse_args()
    if args.isLdr:
       output_dir_name = "flicker"
    else:
       output_dir_name = "hdrplus"
    if args.use_new_f:
        output_dir_name += "_new_f_"
    output_dir_name += args.data_trc + "_factor" + str(args.factor_coeff)
    if args.crop_data:
        output_dir_name += "_crop"
    args.output_dir = os.path.join(args.output_dir_pref, output_dir_name)
    if not os.path.exists(args.output_dir):
       os.mkdir(args.output_dir)
    # create_data(args)
    # print_result("/Users/yaelvinker/Documents/university/lab/Oct/10_15/data_test/")
    # plt.show()
    copy_train_data_from_old_set("/cs/snapless/raananf/yael_vinker/data/new_data/train/hdrplus_new_f_min_log_factor1.0_crop/hdrplus_new_f_min_log_factor1.0_crop",
                                 "/cs/snapless/raananf/yael_vinker/data/new_data_crop_fix/train/train/",
                                 "/cs/snapless/raananf/yael_vinker/data/new_data_crop_fix/train_prev_subset/train")
    # save_f_factor(args)
    # copy_train_data_from_old_set(input_path_names, input_images_path, output_path)
    # split_train_test_data("/cs/snapless/raananf/yael_vinker/data/new_data/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1",
    #                       "/cs/snapless/raananf/yael_vinker/data/new_data/train/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1",
    #                       "/cs/snapless/raananf/yael_vinker/data/new_data/test/flicker_use_factorise_data_0_factor_coeff_1000.0_use_normalization_1")