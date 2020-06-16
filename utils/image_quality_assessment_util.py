import operator
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage

import tranforms
import utils.data_loader_util as data_loader_util
import utils.hdr_image_util as hdr_image_util
#from old_files import TMQI


def save_text_to_image(output_path, text, im_name=""):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (256, 256), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((5, 5), text, fill=(0, 0, 0))
    img.save(os.path.join(output_path, im_name + "all.png"))


def save_single_tone_mapped_result(output_path, title, method_name, image):
    plt.figure(figsize=(30, 30))
    plt.axis("off")
    plt.title(title, fontsize=15)
    plt.imshow(image)
    plt.savefig(os.path.join(output_path, method_name))
    plt.close()
    im = (image * 255).astype('uint8')
    imageio.imwrite(os.path.join(output_path, method_name + "imageio.png"), im, format='PNG-FI')


def get_rgb_imageio_im_file_name(self, im_and_q, color):
    return im_and_q["im_name"] + "_imageio_" + color + ".png"


def ours(original_im,
         net_path="/cs/labs/raananf/yael_vinker/11_06/results/ssim2_log_1000_torus_depth_3/models/net.pth"):
    import torch
    import old_files.torus.Unet as TorusUnet
    import torch.nn as nn
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    # G_net = Unet.UNet(1, 1, 0, bilinear=False, depth=1).to(device)
    print(device)
    G_net = TorusUnet.UNet(1, 1, 0, bilinear=False, depth=3).to(device)
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        G_net = nn.DataParallel(G_net, list(range(torch.cuda.device_count())))
    checkpoint = torch.load(net_path)
    state_dict = checkpoint['modelG_state_dict']
    # G_net.load_state_dict(checkpoint['modelG_state_dict'])

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # load params
    new_state_dict = state_dict
    G_net.load_state_dict(new_state_dict)
    G_net.eval()
    preprocessed_im = apply_preproccess_for_hdr_im(original_im).to(device)
    preprocessed_im_batch = preprocessed_im.unsqueeze(0)
    # data = np.load("data/hdr_log_data/hdr_log_data/belgium_10002.npy", allow_pickle=True)
    # L_hdr_log = data[()]["input_image"].to(device)
    # inputs = L_hdr_log.unsqueeze(0)
    # outputs = net(inputs)
    # L_ldr = np.squeeze(L_ldr.clone().permute(1, 2, 0).detach().cpu().numpy())
    # _L_ldr = to_0_1_range(L_ldr)
    with torch.no_grad():
        ours_tone_map_gray = torch.squeeze(G_net(preprocessed_im_batch.detach()), dim=0)
    ours_tone_map_gray_numpy = ours_tone_map_gray.clone().permute(1, 2, 0).detach().cpu().numpy()
    ours_tone_map_rgb = hdr_image_util.back_to_color(original_im,
                                                     ours_tone_map_gray_numpy)
    # ours_tone_map = np.squeeze(ours_tone_map, axis=0)
    return ours_tone_map_rgb


def apply_preproccess_for_hdr_im(hdr_im, addFrame=False):
    im_hdr_log = log_1000(hdr_im)
    im_log_gray = hdr_image_util.to_gray(im_hdr_log)
    im_log_normalize_tensor = tranforms.tmqi_input_transforms(im_log_gray)
    if addFrame:
        im_log_normalize_tensor = data_loader_util.add_frame_to_im(im_log_normalize_tensor)
    return im_log_normalize_tensor


def hdr_log_loader_factorize(im_hdr, range_factor, brightness_factor):
    im_hdr = im_hdr / np.max(im_hdr)
    total_factor = range_factor * brightness_factor * 255
    image_new_range = im_hdr * total_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(total_factor + 1)).astype('float32')
    return im


def apply_preproccess_for_hdr_im_factorised(hdr_im, addFrame):
    temp_hdr_im = skimage.transform.resize(hdr_im, (128, 128),
                                       mode='reflect', preserve_range=True).astype("float32")
    brightness_factor = hdr_image_util.get_brightness_factor(temp_hdr_im)
    if np.min(hdr_im) < 0:
        hdr_im = hdr_im - np.min(hdr_im)
    im_hdr_log = hdr_log_loader_factorize(hdr_im, 1, brightness_factor)
    im_log_gray = hdr_image_util.to_gray(im_hdr_log)
    im_log_normalize_tensor = tranforms.tmqi_input_transforms(im_log_gray)
    if addFrame:
        im_log_normalize_tensor = data_loader_util.add_frame_to_im(im_log_normalize_tensor)
    return im_log_normalize_tensor


def apply_preproccess_for_hdr_im_factorize(hdr_im, log_factor, addFrame=False):
    brightness_factor = hdr_image_util.get_brightness_factor(hdr_im)
    print(brightness_factor)
    im_hdr_log = hdr_log_loader_factorize(hdr_im, log_factor, brightness_factor)
    im_log_gray = hdr_image_util.to_gray(im_hdr_log)
    im_log_normalize_tensor = tranforms.tmqi_input_transforms(im_log_gray)
    if addFrame:
        im_log_normalize_tensor = data_loader_util.add_frame_to_im(im_log_normalize_tensor)
    return im_log_normalize_tensor


def net_G_pipeline(im_log_normalize_tensor, hdr_im):
    transform_exp = tranforms.Exp(1000)
    exp_im = transform_exp(im_log_normalize_tensor)
    fake_im_gray_numpy = exp_im.clone().permute(1, 2, 0).detach().cpu().numpy()
    exp_im_color = hdr_image_util.back_to_color(hdr_im, fake_im_gray_numpy)
    streched_im = hdr_image_util.to_0_1_range(exp_im_color)
    return streched_im


def log_1(im, range_factor=1):
    max_origin = np.max(im)
    image_new_range = (im / max_origin) * range_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(range_factor + 1)).astype('float32')
    return im


def log_100(im, range_factor=100):
    max_origin = np.max(im)
    image_new_range = (im / max_origin) * range_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(range_factor + 1)).astype('float32')
    return im


def log_1000(im, range_factor=1000):
    max_origin = np.max(im)
    image_new_range = (im / max_origin) * range_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(range_factor + 1)).astype('float32')
    return im


def log_1000_exp(im):
    im_log_normalize_tensor = apply_preproccess_for_hdr_im(im)
    log1000_im = net_G_pipeline(im_log_normalize_tensor, im)
    return log1000_im


def log_100_exp(im):
    im = log_100(im)
    return exp_map(im)


def exp_map(im):
    import math
    im_0_1 = (im - np.min(im)) / (np.max(im) - np.min(im))
    im_exp = np.exp(im_0_1) - 1
    im_end = im_exp / (math.exp(1) - 1)
    return im_end


def Dargo_tone_map(im):
    # Tonemap using Drago's method to obtain 24-bit color image
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7, 0.85)
    ldrDrago = tonemapDrago.process(im)
    return ldrDrago


def Durand_tone_map(im):
    # Tonemap using Durand's method obtain 24-bit color image
    tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    ldrDurand = tonemapDurand.process(im)
    # ldrDurand = 3 * ldrDurand
    # hdr_image_utils.print_image_details(ldrDurand,"durand")
    return ldrDurand


def Reinhard_tone_map(im):
    # Tonemap using Reinhard's method to obtain 24-bit color image
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(im)
    return ldrReinhard


def Mantiuk_tone_map(im):
    # Tonemap using Mantiuk's method to obtain 24-bit color image
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(im)
    return ldrMantiuk


def Durand_tone_map(im):
    # Tonemap using Durand's method obtain 24-bit color image
    tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    ldrDurand = tonemapDurand.process(im)
    return ldrDurand


def ours_input(im_name):
    path = "/cs/labs/raananf/yael_vinker/NEW_TMQI/our_res_torus2"
    im_path = os.path.join(path, im_name + "_imageio_rgb.png")
    im_ldr = hdr_image_util.read_ldr_image(im_path)
    print("my shape : ", im_ldr.shape)
    print(im_ldr.dtype, np.max(im_ldr))
    return im_ldr


def calculate_TMQI_results_for_selected_methods(im_hdr_original, img_name, new_output_path="", save_images=False):
    tone_map_methods = {
        # "Our_torus": ours,
        # "ours_input": ours_input,
        "Reinhard": Reinhard_tone_map,
        "Dargo": Dargo_tone_map,
        "Mantiuk": Mantiuk_tone_map,
        "Durand": Durand_tone_map,
        "log100_exp": log_100_exp,
        "log1000_exp": log_1000_exp,
        "log100": log_100,
        "log1000": log_1000,
        "log": log_1}
    methods_and_Q_results = {}
    for method_name in tone_map_methods.keys():
        if method_name == "ours_input":
            tone_mapped_result = tone_map_methods[method_name](os.path.splitext(img_name)[0])
        else:
            tone_mapped_result = tone_map_methods[method_name](im_hdr_original)
        Q, S, N = TMQI.TMQI(im_hdr_original, tone_mapped_result)
        methods_and_Q_results[method_name] = Q
        title = method_name + "\nQ = " + str(Q) + "\nS = " + str(S) + "\n" + "N = " + str(N) + "\n" + "max = " \
                + str(np.max(tone_mapped_result)) + "   min = " + str(np.min(tone_mapped_result))
        if save_images:
            save_single_tone_mapped_result(new_output_path, title, method_name, tone_mapped_result)

    sorted_methods_and_Q_results_by_Q = sorted(methods_and_Q_results.items(), key=operator.itemgetter(1))[::-1]
    text = ""
    for method_and_q_tuple in sorted_methods_and_Q_results_by_Q:
        subtitle = method_and_q_tuple[0] + " : " + str(method_and_q_tuple[1]) + "\n"
        text += subtitle
    return text


def create_tone_mapped_datasets_for_fid(input_path, output_path):
    tone_map_methods = {
        "Reinhard": Reinhard_tone_map,
        "Dargo": Dargo_tone_map,
        "Mantiuk": Mantiuk_tone_map,
        "Durand": Durand_tone_map,
        "log100_exp": log_100_exp,
        "log1000_exp": log_1000_exp,
        "log100": log_100,
        "log1000": log_1000,
        "log": log_1}
    for method_name in tone_map_methods.keys():
        method_output_path = os.path.join(output_path, method_name)
        if not os.path.exists(method_output_path):
            os.mkdir(method_output_path)
        for img_name in os.listdir(input_path):
            im_path = os.path.join(input_path, img_name)
            original_im = hdr_image_util.read_hdr_image(im_path)
            original_im = skimage.transform.resize(original_im, (299, 299),
                                                   mode='reflect', preserve_range=False).astype("float32")
            tone_mapped_result = tone_map_methods[method_name](original_im)
            tone_mapped_result = tone_mapped_result / np.max(tone_mapped_result)
            im = (tone_mapped_result * 255).astype('uint8')
            imageio.imwrite(os.path.join(method_output_path, os.path.splitext(img_name)[0] + ".jpg"), im,
                            format='JPEG-PIL')


def calaulate_and_save_TMQI_from_path(input_path, output_path):
    for img_name in os.listdir(input_path):
        hdr_path = os.path.join(input_path, img_name)
        im_hdr_original = hdr_image_util.read_hdr_image(hdr_path)
        im_hdr_original = skimage.transform.resize(im_hdr_original, (int(im_hdr_original.shape[0] / 4),
                                                                     int(im_hdr_original.shape[1] / 4)), mode='reflect',
                                                   preserve_range=False).astype("float32")
        new_output_path = os.path.join(output_path, os.path.splitext(img_name)[0])
        if not os.path.exists(new_output_path):
            os.makedirs(new_output_path)
        result_text = calculate_TMQI_results_for_selected_methods(im_hdr_original, img_name, new_output_path)
        save_text_to_image(new_output_path, result_text)


def log_to_image(im_origin, log_factor):
    import numpy as np
    max_origin = np.max(im_origin)
    image_new_range = (im_origin / max_origin) * log_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(log_factor + 1)).astype('float32')
    return im


def load_original_test_hdr_images(root, log_factor):
    import skimage
    original_hdr_images = []
    counter = 1
    for img_name in os.listdir(root):
        im_path = os.path.join(root, img_name)
        im_hdr_original = hdr_image_util.read_hdr_image(im_path)
        im_hdr_original = skimage.transform.resize(im_hdr_original, (int(im_hdr_original.shape[0] / 3),
                                                                     int(im_hdr_original.shape[1] / 3)),
                                                   mode='reflect', preserve_range=False).astype("float32")
        im_hdr_log = log_to_image(im_hdr_original, log_factor)
        im_log_gray = hdr_image_util.to_gray(im_hdr_log)
        im_log_normalize_tensor = tranforms.tmqi_input_transforms(im_log_gray)


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


def sort_files():
    import shutil
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
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "loss_plot")
        old_name = "summary epoch_=_" + epoch + "all.png"
#        summarepoch_=_0all
        cur_output_path = os.path.join(output_path, "accuracy_" + epoch)
        output_name = date + "_" + arch_name + ".png"
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        if os.path.exists(os.path.join(im_path, old_name)):
            print(os.path.join(cur_output_path, output_name))
            shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
        else:
            print(os.path.join(im_path,old_name))
        # os.rename(os.path.join(im_path, old_name), os.path.join(output_path, output_name))


def gather_all_architectures(arch_dir, output_path, epoch, date, im_number):
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "model_results", epoch)
        old_name = im_number + "_stretch.jpg"
        cur_output_path = os.path.join(output_path, epoch, im_number + "_color_stretch")
        output_name = date + "_" + arch_name + ".jpg"
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        if os.path.exists(os.path.join(im_path, old_name)):
            shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
        # os.rename(os.path.join(im_path, old_name), os.path.join(output_path, output_name))


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


if __name__ == '__main__':
    rename_files("/Users/yaelvinker/Downloads/image-quality-assessment-master/src/tests/test_images/fattal/")
    # save_dng_data_for_fid("/cs/labs/raananf/yael_vinker/dng_collection",
    #                       "/cs/snapless/raananf/yael_vinker/data/dng_data_fid",
    #                       "/cs/snapless/raananf/yael_vinker/data/04_26_new_data/hdrplus_gamma_log_10_other_images_wrap/hdrplus_gamma_log_10_other_images/")
    # sort_files()
    #    normalization_test()
    # sub_test()
    # struct_loss_res("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr")
    epochs = ["120"]
    im_numbers = ["synagogue", "belgium"]
    # for epoch in epochs:
    #   for im_number in im_numbers:
    #      gather_all_architectures("/cs/labs/raananf/yael_vinker/05_12/results_05_12",
    #             "/cs/labs/raananf/yael_vinker/05_12/summary_05_12", epoch, "", im_number)
    # gather_all_architectures_accuracy("/cs/labs/raananf/yael_vinker/05_12/results_05_12",
    #                                   "/cs/labs/raananf/yael_vinker/05_12/summary_05_12",
    #                                   "120", "")

