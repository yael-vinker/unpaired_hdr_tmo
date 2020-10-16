import sys
import inspect
import os
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import argparse
import torch
import time
import csv
import cv2
import imageio
import numpy as np
import skimage
import shutil
import utils.hdr_image_util as hdr_image_util
from utils import model_save_util
from PIL import Image
import matplotlib.pyplot as plt


def create_hdr_dataset_from_dng(dng_path, output_hdr_path):
    if not os.path.exists(output_hdr_path):
        os.mkdir(output_hdr_path)
    for img_name in os.listdir(dng_path):
        print(img_name)
        im_path = os.path.join(dng_path, img_name)
        original_im = hdr_image_util.read_hdr_image(im_path)
        original_im = hdr_image_util.reshape_image(original_im)
        im_bgr = cv2.cvtColor(original_im, cv2.COLOR_RGB2BGR)
        hdr_name = os.path.splitext(img_name)[0] + ".hdr"
        cv2.imwrite(os.path.join(output_hdr_path, hdr_name), im_bgr)


def create_exr_reshaped_dataset_from_exr(exr_path, output_exr_path):
    if not os.path.exists(output_exr_path):
        os.mkdir(output_exr_path)
    for img_name in os.listdir(exr_path):
        file_extension = os.path.splitext(img_name)[1]
        if file_extension == ".png":
            print(img_name)
            im_path = os.path.join(exr_path, img_name)
            # original_im = hdr_image_util.read_hdr_image(im_path)
            # print(original_im.shape)
            # original_im = hdr_image_util.reshape_image_fixed_size(original_im)
            original_im = imageio.imread(im_path)
            original_im = skimage.transform.resize(original_im, (int(original_im.shape[0] / 2),
                                                       int(original_im.shape[1] / 2)),
                                              mode='reflect', preserve_range=False).astype('float32')
            original_im = hdr_image_util.to_0_1_range(original_im)
            original_im = (original_im * 255).astype('uint8')
            hdr_image_util.print_image_details(original_im, img_name)
            # im_bgr = cv2.cvtColor(original_im, cv2.COLOR_RGB2BGR)
            # print(im_bgr.shape)
            # cv2.imwrite(os.path.join(output_exr_path, img_name), im_bgr)
            # im = imageio.imread(os.path.join(output_exr_path, img_name))
            # print(im.shape)
            imageio.imwrite(os.path.join(output_exr_path, os.path.splitext(img_name)[0] + "our_gray" + ".png"), original_im)


def save_clipped_open_exr(input_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for img_name in os.listdir(input_path):
        file_extension = os.path.splitext(img_name)[1]
        if file_extension == ".jpg":
            im_path = os.path.join(input_path, img_name)
            print(im_path)
            original_im = imageio.imread(im_path)
            original_im = (original_im - original_im.min()) / (original_im.max() - original_im.min())
            # original_im = original_im / np.max(original_im)
            original_im = (original_im * 1.1 - 0.05) * 255
            new_im = np.clip(original_im, 0, 255).astype('uint8')
            cur_output_path = os.path.join(output_path, img_name)
            imageio.imwrite(cur_output_path, new_im)


def rename_files(input_path):
    for img_name in os.listdir(input_path):

        im_name = os.path.splitext(img_name)[0]
        file_extension = os.path.splitext(img_name)[1]
        if file_extension == '.jpg':
            new_im_name = im_name[:-10] + '.jpg'
            print(new_im_name)
            os.rename(os.path.join(input_path, img_name), os.path.join(input_path, new_im_name))


def rename_our_files(input_path_, file_name):
    import re
    input_path = input_path_ + "/" + file_name#+ "_color_stretch"
    for img_name in os.listdir(input_path):

        im_name = os.path.splitext(img_name)[0]
        file_extension = os.path.splitext(img_name)[1]
        if file_extension == '.png':
            items = re.findall("manualD.*\[.*\]", img_name)[0]
            rseed = ""
            if "rseed" in img_name:
                rseed = "rseed"
            im_number = im_name[-2:]
            new_im_name = "%s_%s_%s_%s_%s.png" % (file_name, im_name[:3], items, rseed, im_number)
            # print(im_name)
            # new_im_name = file_name + "_" + im_name[28:38] + im_name[88:129] + im_name[-3:] + '.png'
            # # new_im_name =
            print(new_im_name)
            os.rename(os.path.join(input_path, img_name), os.path.join(input_path, new_im_name))


def sort_files():
    import csv
    im_to_save = "_stretch"
    images = ["OtterPoint", "synagogue", "belgium"]
    epoch = "360"
    arch_dir = "/cs/labs/raananf/yael_vinker/05_05/results/"
    new_path = "/cs/labs/raananf/yael_vinker/05_05/summary/sort_" + epoch + im_to_save
    for im_name in images:
        cur_output_path = os.path.join(new_path, im_name)
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    with open('/cs/labs/raananf/yael_vinker/05_05/summary/05_05_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            cur_dir_name = row[0]
            val = row[1]
            cur_dir_path = os.path.join(os.path.abspath(arch_dir), cur_dir_name, "hdr_format_factorised_1", epoch)
            for im_name in images:
                old_im_path = os.path.join(cur_dir_path, im_name + im_to_save + ".jpg")
                new_name = ("%.5s" % val) + "_" + row[0] + ".jpg"
                new_im_path = os.path.join(new_path, im_name, new_name)
                shutil.copy(old_im_path, new_im_path)


def gather_all_architectures_exr(arch_dir, output_path, epoch, date, im_number):
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "exr_320")
        old_name = im_number + "_stretch.png"
        cur_output_path = os.path.join(output_path, epoch, im_number)
        output_name = date + "_" + arch_name + ".png"
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        if os.path.exists(os.path.join(im_path, old_name)):
            shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
        # os.rename(os.path.join(im_path, old_name), os.path.join(output_path, output_name))


def gather_im_by_epoch(arch_dir, output_path, im_number):
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "model_results")
        cur_output_path = os.path.join(output_path, arch_name + "_" + im_number)
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        for ep in os.listdir(im_path):
            cur_im_path = os.path.join(im_path, ep)
            old_name = im_number + "_stretch.png"
            output_name = ep + "_" + im_number + ".png"
            if os.path.exists(os.path.join(cur_im_path, old_name)):
                shutil.copy(os.path.join(cur_im_path, old_name), os.path.join(cur_output_path, output_name))


def save_dng_data_for_fid(input_path, output_path, other_path):
    from shutil import copyfile
    data = os.listdir(other_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for img_name in data:
        img_base_name = os.path.splitext(img_name)[0]
        origin_im_path = os.path.join(input_path, img_base_name + ".dng")
        new_im_path = os.path.join(output_path, img_base_name + ".dng")
        if os.path.exists(origin_im_path):
            copyfile(origin_im_path, new_im_path)


def sort_exr_res_by_btmqi():
    input_dir = "/Users/yaelvinker/Documents/university/lab/Sep/" \
               "09_02_summary/09_03_crop_test/crop_D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT__" \
               "pretrain50_lr_g1e-05_d1e-05_decay50_noframe_stretch_1.05_LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_0.1new_f_/exr"
    new_path = "/Users/yaelvinker/Documents/university/lab/Sep/" \
               "09_02_summary/09_03_crop_test/crop_D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT__" \
               "pretrain50_lr_g1e-05_d1e-05_decay50_noframe_stretch_1.05_LOSS_d5.0_gamma_ssim1.0_2,4,4__DATA_min_log_0.1new_f_/exr_sort_btmqi"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    with open('/Users/yaelvinker/Documents/MATLAB/CODE_2016TMM2/results/our_111_log01.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            cur_im_name = row[0] + ".png"
            val = row[1]
            print(cur_im_name, val)
            old_im_path = os.path.join(os.path.abspath(input_dir), cur_im_name)
            new_name = ("%.5s" % val) + "_" + cur_im_name
            new_im_path = os.path.join(new_path, new_name)
            shutil.copy(old_im_path, new_im_path)


def npy_to_jpg(input_path, output_path):
    import pathlib
    path = pathlib.Path(input_path)
    files_npy = list(path.glob('*.npy'))
    for f in files_npy:
        print(os.path.splitext(os.path.basename(f))[0])
        data = np.load(f, allow_pickle=True)
        color_im = data[()]["display_image"]
        color_im = color_im.permute(1, 2, 0).numpy()
        print(color_im.shape, color_im.max())
        cur_output_path = os.path.join(output_path, os.path.splitext(os.path.basename(f))[0] + ".raw")
        imageio.imsave(cur_output_path, color_im, format="RAW-FI")


def gather_all_architectures_accuracy(arch_dir, output_path, epoch, date):
    for test_name in os.listdir(arch_dir):
        cur_path = os.path.join(os.path.abspath(arch_dir), test_name)
        for arch_name in os.listdir(cur_path):
            im_path = os.path.join(os.path.abspath(arch_dir), test_name, arch_name, "accuracy")
            old_name = "acc" + epoch + ".png"
            cur_output_path = os.path.join(output_path, test_name, "accuracy_" + epoch)
            output_name = date + "_" + arch_name + ".png"
            if not os.path.exists(cur_output_path):
                os.makedirs(cur_output_path)
            if os.path.exists(os.path.join(im_path, old_name)):
                shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
            else:
                print(os.path.join(im_path,old_name))


def gather_all_architectures_loss(arch_dir, output_path, epoch, date):
    for test_name in os.listdir(arch_dir):
        cur_path = os.path.join(os.path.abspath(arch_dir), test_name)
        for arch_name in os.listdir(cur_path):
            im_path = os.path.join(os.path.abspath(arch_dir), test_name, arch_name, "loss_plot")
            old_name = "summary epoch_=_" + epoch + "all.png"
            cur_output_path = os.path.join(output_path, test_name, "loss" + epoch)
            output_name = date + "_" + arch_name + ".png"
            if not os.path.exists(cur_output_path):
                os.makedirs(cur_output_path)
            if os.path.exists(os.path.join(im_path, old_name)):
                print(os.path.join(cur_output_path, output_name))
                shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
            else:
                print(os.path.join(im_path,old_name))


def gather_all_architectures(arch_dir, output_path, epoch, date, im_number):
    for test_name in os.listdir(arch_dir):
        cur_path = os.path.join(os.path.abspath(arch_dir), test_name)
        for arch_name in os.listdir(cur_path):
            im_path = os.path.join(os.path.abspath(arch_dir), test_name, arch_name, "model_results", epoch)
            cur_output_path = os.path.join(output_path, test_name, epoch, im_number)
            if not os.path.exists(cur_output_path):
                os.makedirs(cur_output_path)
            old_name = im_number + ".png"
            if os.path.exists(os.path.join(im_path, old_name)):
                output_name = date + "_" + arch_name + ".png"
                shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
            old_name = im_number + "_1.png"
            if os.path.exists(os.path.join(im_path, old_name)):
                output_name = date + "_" + arch_name + "_1.png"
                shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
            old_name = im_number + "_0.png"
            if os.path.exists(os.path.join(im_path, old_name)):
                output_name = date + "_" + arch_name + "_0.png"
                shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))


def save_training_summary(args):
    results_path = args.results_path
    summary_path = args.summary_path
    epochs = ["320"]
    im_numbers = ["OtterPoint", "synagogue", "belgium", "2"]
    for epoch in epochs:
        for im_number in im_numbers:
            gather_all_architectures(results_path, summary_path, epoch, "", im_number)
    gather_all_architectures_accuracy(results_path, summary_path, "320", "")


def run_trained_model(args):
    model_name = args.model_name
    model_path = args.model_path
    output_name = args.output_name
    input_images_path = args.input_path
    f_factor_path = args.f_factor_path
    input_images_names_path = args.input_images_names_path

    start0 = time.time()
    net_path = os.path.join(model_path, model_name, "models", "net_epoch_320.pth")
    train_settings_path = os.path.join(model_path + "/" + model_name, "run_settings.npy")
    print("train_settings_path",train_settings_path)
    print("model_path", model_path)
    print("net_path", net_path)
    model_params = model_save_util.get_model_params(model_name, train_settings_path)
    model_params["test_mode_f_factor"] = args.test_mode_f_factor
    model_params["test_mode_frame"] = args.test_mode_frame
    # model_params["factor_coeff"] = 0.5
    print("===============================================")
    print(model_params)

    output_images_path = os.path.join(model_path, model_name, output_name)
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model_save_util.run_model_on_path(model_params, device, net_path, input_images_path,
                                      output_images_path, f_factor_path, None, input_images_names_path)
    print(time.time()-start0)


def read_txt_file_with_bad_fid_images():
    input_fid_images_path = "/cs/labs/raananf/yael_vinker/Oct/10_13/results_10_13/hist_fit_test/1_D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT___rseed_Truepretrain50_lr_g1e-05_d1e-05_decay50_noframe__LOSS_d1.0_gamma_ssim1.0_1,1,1__DATA_min_log_0.1hist_fit_/fid_color_stretch_fix/"
    output_path = "/cs/labs/raananf/yael_vinker/Oct/10_13/results_10_13/hist_fit_test/1_D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT___rseed_Truepretrain50_lr_g1e-05_d1e-05_decay50_noframe__LOSS_d1.0_gamma_ssim1.0_1,1,1__DATA_min_log_0.1hist_fit_/bad_fid_images/"
    with open('bad_images_fid.txt') as f:
        lines = [line.rstrip() for line in f]
    for im in lines:
        if im in os.listdir(input_fid_images_path):
            shutil.move(input_fid_images_path + im, output_path + im)
            print(im)
        else:
            print(im, " not in dir")

def im_crop_test(filename):
    im = Image.open(filename)

    plt.figure()
    plt.imshow(im)

    width, height = im.size
    left = 10
    top = 10
    right = width - 10
    bottom = height - 10
    im = im.crop((left, top, right, bottom))
    plt.figure()
    plt.imshow(im)

    im = im.resize((299,299), Image.BICUBIC)
    im = np.asarray(im)[..., :3]
    plt.figure()
    plt.imshow(im)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--func_to_run", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--f_factor_path", type=str)
    parser.add_argument("--test_mode_f_factor", type=int, default=0)
    parser.add_argument("--test_mode_frame", type=int, default=0)
    parser.add_argument("--input_images_names_path", type=str)
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--summary_path", type=str)

    args = parser.parse_args()

    func_to_run = args.func_to_run
    if func_to_run == "run_trained_model":
        run_trained_model(args)
    elif func_to_run == "run_summary":
        save_training_summary(args)