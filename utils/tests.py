import os
import pathlib
from math import exp

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


def print_im_details(im, title, disply=False):
    print(title)
    print("max = ", np.max(im), "min = ", np.min(im), "mean = ", np.mean(im), "dtype = ", im.dtype)
    if disply:
        plt.imshow(im)
        plt.show()


def print_batch_details(batch, title, disply=False):
    print(title)
    b_size = batch.shape[0]
    for i in range(b_size):
        im = batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        hdr_image_utils.print_image_details(im, i)
        if disply:
            im_n = gan_trainer_utils.to_0_1_range(im)
            plt.imshow(im_n)
            plt.show()


def compare_tensors(b1, b2):
    b_size = b1.shape[0]
    for i in range(b_size):
        im1 = b1[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        im2 = b2[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        print(np.array_equal(im1, im2))


def test_normalize_transform(batch, device):
    print_batch_details(batch, "before")
    before_batch = batch.clone()
    normalize = transforms_.NormalizeForDisplay((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), device)
    new_batch = normalize(batch)
    print_batch_details(new_batch, "after")
    to_pil = transforms.ToPILImage()
    b_size = batch.shape[0]
    for i in range(b_size):
        new_batch[i] = to_pil(new_batch[i])
    batch_origin = F.normalize(new_batch, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
    # normalize_back = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # batch_origin = normalize_back(new_batch)
    compare_tensors(before_batch, batch_origin)


def verify_G_output_range(fake_results, min, max):
    b_size = fake_results.shape[0]
    for i in range(b_size):
        fake_copy = fake_results[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        if np.min(fake_copy) < min:
            print("Error in G output, min value is smaller than ", min)
        if np.max(fake_copy) > max:
            print("Error in G output, max value is larger than ", max)
        # print(fake_copy)
        # gray_num = (fake_copy == 0.5).sum()
        # print("Percentage of 0.5 pixels = ",gray_num / (fake_copy.shape[0] * fake_copy.shape[1]), "%")


def verify_model_load(model, model_name, optimizer):
    # Print model's state_dict
    print("State_dict for " + model_name)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name)
        # print(var_name, "\t", optimizer.state_dict()[var_name])


def model_parameters_update_test(params_befor_opt_step, model, model_name):
    params_after = list(model.parameters())[0].clone()
    is_equal = torch.equal(params_befor_opt_step.data, params_after.data)
    if is_equal:
        print("Error: parameters of model " + model_name + " remain the same")
    else:
        print(model_name + " parameters were updated successfully")


def plot_data(dataloader, device, title):
    """Plot some training images"""
    real_batch = next(iter(dataloader))
    first_b = real_batch[0].to(device)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(first_b[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def plot_npy_data(dataloader, device, title):
    """Plot some training images"""
    real_batch = next(iter(dataloader))
    first_b = real_batch[params.image_key].to(device)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(first_b[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def get_data_root():
    batch_size, num_epochs, model, G_lr, D_lr, train_data_root_npy, train_data_root_ldr, isCheckpoint_str, \
    test_data_root_npy, test_data_root_ldr, g_opt_for_single_d, \
    result_dir_pref, input_dim, apply_g_ssim = GanTrainer.parse_arguments()
    # Decide which device we want to run on
    return train_data_root_npy


def transforms_test():
    transform_original = transforms.Compose([
        transforms.Resize(params.input_size),
        transforms.CenterCrop(params.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_custom = transforms.Compose([
        transforms_.Scale(params.input_size),
        transforms_.CenterCrop(params.input_size),
        transforms_.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_root = get_data_root()
    dataset_origin = dset.ImageFolder(root=data_root,
                                      transform=transform_original)

    dataset_custom = HdrImageFolder.HdrImageFolder(root=data_root,
                                                   transform=transform_custom)

    original_transform_im = np.asarray(dataset_origin[0][0].permute(1, 2, 0).detach().cpu().numpy())
    custom_transform_im = np.asarray(dataset_custom[0][0].permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(original_transform_im)
    plt.show()
    plt.imshow(custom_transform_im)
    plt.show()
    print(np.array_equal(original_transform_im, custom_transform_im))
    hdr_image_utils.print_image_details(original_transform_im, "original_transform_im")
    hdr_image_utils.print_image_details(custom_transform_im, "custom_transform_im")


def ssim_test():
    transform_custom = transforms.Compose([
        transforms_.Scale(params.input_size),
        transforms_.CenterCrop(params.input_size),
        transforms_.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    data_root = get_data_root()
    print(data_root)
    data_root_2 = "data/hdr_data_test"
    dataset1 = HdrImageFolder.HdrImageFolder(root=data_root,
                                             transform=transform_custom)
    dataset2 = HdrImageFolder.HdrImageFolder(root=data_root_2,
                                             transform=transform_custom)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=4,
                                              shuffle=False, num_workers=params.workers)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=4,
                                              shuffle=False, num_workers=1)
    batch1 = (next(iter(dataloader1)))[0]
    batch2 = (next(iter(dataloader2)))[0]
    ssim_loss = ssim.SSIM(window_size=11)
    print(ssim_loss(batch1, batch2))


def test1(root):
    for img_name in os.listdir(root):
        print("-------------------------------------------")
        im_path = os.path.join(root, img_name)
        path_lib_path = pathlib.Path(im_path)
        file_extension = os.path.splitext(im_path)[1]
        if file_extension == ".hdr":
            im_origin = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
        elif file_extension == ".dng":
            im_origin = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
        elif file_extension == ".bmp":
            im_origin = imageio.imread(path_lib_path).astype('float32')

        hdr_image_utils.print_image_details(im_origin, img_name + " 1")

        # im_origin = im_origin / np.max(im_origin)
        # # im_origin = np.log(im_origin + 1)
        # hdr_image_utils.print_image_details(im_origin, img_name)
        #
        # std_im = (im_origin - 0.5) / 0.5
        # hdr_image_utils.print_image_details(std_im, "std_im")
        #
        # im_n = (std_im - np.min(std_im)) / (np.max(std_im) - np.min(std_im))
        # hdr_image_utils.print_image_details(im_n, "im_n")

        im_log = np.log(im_origin + 1)

        im_log_norm = im_log / np.max(im_log)  # 0-1
        hdr_image_utils.print_image_details(im_log_norm, "im_log_norm")

        std_im = (im_log_norm - 0.5) / 0.5  # -1 1
        hdr_image_utils.print_image_details(std_im, "std_im")

        im_n = (std_im - np.min(std_im)) / (np.max(std_im) - np.min(std_im))  # 0-1 (im_log_norm)
        hdr_image_utils.print_image_details(im_n, "im_n")

        im_org = np.exp(im_n * np.log(255)) - 1
        hdr_image_utils.print_image_details(im_org, "im_org")
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("im_origin")
        plt.imshow(im_origin / np.max(im_origin))
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("im_log_norm")
        plt.imshow(im_log_norm)
        plt.show()


def show_two_images(im1, im2):
    hdr_image_utils.print_tensor_details(im1, "no exp")
    hdr_image_utils.print_tensor_details(im2, "exp")
    im1 = np.squeeze(np.asarray(im1.permute(1, 2, 0).detach().cpu().numpy()))
    im2 = np.squeeze(np.asarray(im2.permute(1, 2, 0).detach().cpu().numpy()))
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 1, 1)
    plt.imshow(im1, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.imshow(im2, cmap='gray')
    plt.show()
    hdr_image_utils.print_image_details(im1, "no exp")
    hdr_image_utils.print_image_details(im2, "exp")


def exp_transform_test():
    im = gan_trainer_utils.read_ldr_image("data/ldr_data/ldr_data/im_96.bmp")
    im = gan_trainer_utils.to_gray(im)
    im = transforms_.gray_image_transform(im)
    exp_transform = transforms_.Exp(100)
    im_after_exp = exp_transform(im)
    show_two_images(gan_trainer_utils.to_0_1_range_tensor(im), im_after_exp)


def tmqi_test(input_path, im_hdr_path):
    im_hdr = imageio.imread(im_hdr_path, format="HDR-FI")
    tmqi_res = []
    names = []
    for img_name in os.listdir(input_path):
        if os.path.splitext(img_name)[1] == ".png":
            im_path = os.path.join(input_path, img_name)
            print(img_name)
            rgb_img = imageio.imread(im_path)
            print(rgb_img.dtype)
            tmqi_cur = TMQI.TMQI(im_hdr, rgb_img)[0]
            tmqi_res.append(tmqi_cur)
            names.append(os.path.splitext(img_name)[0])
    plt.figure()
    plt.plot(np.arange(9), tmqi_res, "-r.")
    plt.xticks(np.arange(9), names, rotation=45)
    # plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    # plt.subplots_adjust(bottom=0.15)
    plt.savefig("/Users/yaelvinker/Documents/university/lab/matlab_input_niqe/res")


def model_test():
    import models.unet_multi_filters.Unet as squre_unet
    import old_files.torus.Unet as TorusUnet
    import old_files.unet.Unet as Unet
    from torchsummary import summary
    import utils.model_save_util as msu

    # new_net = TorusUnet.UNet(input_dim_, input_dim_, input_images_mean_, bilinear=False, depth=unet_depth_).to(
    #     device_)
    unet_bilinear = Unet.UNet(1, 1, 0, bilinear=True, depth=4)
    unet_conv = Unet.UNet(1, 1, 0, bilinear=False, depth=4)
    torus = TorusUnet.UNet(1, 1, 0, bilinear=False, depth=3)

    layer_factor = msu.get_layer_factor(params.original_unet)
    new_unet_conv = squre_unet.UNet(1, 1, 0, depth=4, layer_factor=layer_factor,
                                    con_operator=params.original_unet, filters=32, bilinear=False,
                                    network=params.unet_network, dilation=0)

    new_torus = squre_unet.UNet(1, 1, 0, depth=3, layer_factor=layer_factor,
                                con_operator=params.original_unet, filters=32, bilinear=False,
                                network=params.torus_network, dilation=2)
    # print(unet_conv)
    # summary(unet_conv, (1, 256, 256), device="cpu")

    print(new_torus)
    summary(new_torus, (1, 256, 256), device="cpu")


def to_gray(im):
    return np.dot(im[..., :3], [0.299, 0.587, 0.114]).astype('float32')


def to_0_1_range(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def to_0_1_range_tensor(im):
    return (im - im.min()) / (im.max() - im.min())


def ssim_test():
    im_tone_mapped = imageio.imread(
        "/Users/yaelvinker/PycharmProjects/lab/local_log_1000_unet_original_unet_depth_2/model_results/1/1_epoch_1_rgb.png")
    im_tone_mapped = to_gray(im_tone_mapped)
    im_tone_mapped = to_0_1_range(im_tone_mapped)
    im_tone_mapped_tensor = torch.from_numpy(im_tone_mapped)
    im_tone_mapped_tensor = im_tone_mapped_tensor[None, None, :, :]

    hdr_im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/S0010.hdr", format="HDR-FI")
    hdr_im = to_gray(hdr_im)
    hdr_im = to_0_1_range(hdr_im)
    hdr_im_tensor = torch.from_numpy(hdr_im)
    hdr_im_tensor = hdr_im_tensor[None, None, :, :]
    ssim.ssim(hdr_im_tensor, im_tone_mapped_tensor)
    print("tmqi")
    print(TMQI.TMQI(hdr_im, im_tone_mapped))


def to_numpy_display(im):
    im = im.clone().permute(1, 2, 0).detach().cpu().numpy()
    return np.squeeze(im)


def frame_test():
    SHAPE_ADDITION = 45
    data = np.load("/Users/yaelvinker/PycharmProjects/lab/data/ldr_npy/ldr_npy/im_96_one_dim.npy", allow_pickle=True)
    input_im = data[()]["input_image"]
    hdr_image_utils.print_tensor_details(input_im, "im")
    input_im = to_0_1_range_tensor(input_im)
    input_im = torch.squeeze(input_im)

    first_row = input_im[0].repeat(SHAPE_ADDITION, 1)
    im = torch.cat((first_row, input_im), 0)
    last_row = input_im[-1].repeat(SHAPE_ADDITION, 1)
    im = torch.cat((im, last_row), 0)

    left_col = torch.t(im[:, 0].repeat(SHAPE_ADDITION, 1))

    im = torch.cat((left_col, im), 1)

    right_col = torch.t(im[:, -1].repeat(SHAPE_ADDITION, 1))
    im = torch.cat((im, right_col), 1)
    im = torch.unsqueeze(im, dim=0)
    c, h, w = im.shape
    th, tw = 256, 256
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    i, j, h, w = i, j, th, tw
    im = im[:, i:i + h, j:j + w]
    print(im.shape)
    plt.imshow(to_numpy_display(im), cmap='gray')
    plt.show()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def our_custom_ssim_test():
    data = np.load("/Users/yaelvinker/PycharmProjects/lab/data/ldr_npy/ldr_npy/im_96_one_dim.npy", allow_pickle=True)
    img1 = data[()]["input_image"]
    img2 = img1 + 5
    # im_tone_mapped = imageio.imread(
    #     "/Users/yaelvinker/PycharmProjects/lab/local_log_1000_unet_original_unet_depth_2/model_results/1/1_epoch_1_rgb.png")
    # im_tone_mapped = to_gray(im_tone_mapped)
    # im_tone_mapped = to_0_1_range(im_tone_mapped)
    # im_tone_mapped_tensor = torch.from_numpy(im_tone_mapped)
    im_tone_mapped_tensor_tensor_b = torch.zeros([2, img1.shape[0], img1.shape[1], img1.shape[2]])
    im_tone_mapped_tensor_tensor_b[0] = img1
    im_tone_mapped_tensor_tensor_b[1] = img1

    im_tone_mapped_tensor_tensor_b2 = torch.zeros([2, img1.shape[0], img1.shape[1], img1.shape[2]])
    im_tone_mapped_tensor_tensor_b2[0] = img2
    im_tone_mapped_tensor_tensor_b2[1] = img2
    # im_tone_mapped_tensor = im_tone_mapped_tensor[None, None, :, :]

    hdr_im = imageio.imread(
        "/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/S0010.hdr",
        format="HDR-FI")
    hdr_im = to_gray(hdr_im)
    hdr_im = to_0_1_range(hdr_im)
    hdr_im_tensor = torch.from_numpy(hdr_im)
    hdr_im_tensor_b = torch.zeros([2, 1, hdr_im_tensor.shape[0], hdr_im_tensor.shape[1]])
    hdr_im_tensor_b[0, :] = hdr_im_tensor
    hdr_im_tensor_b[1, :] = hdr_im_tensor
    our_ssim_loss = ssim.OUR_CUSTOM_SSIM(window_size=5)
    print(our_ssim_loss(im_tone_mapped_tensor_tensor_b2, im_tone_mapped_tensor_tensor_b))
    print(our_ssim_loss(im_tone_mapped_tensor_tensor_b, im_tone_mapped_tensor_tensor_b))
    print(ssim.ssim(im_tone_mapped_tensor_tensor_b2, im_tone_mapped_tensor_tensor_b))
    # print(our_ssim_loss(im_tone_mapped_tensor, im_tone_mapped_tensor * 2))


def exr_to_hdr():
    im507 = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/507_bgr.hdr", format="HDR-FI")
    # im507_hdr = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/507_lum.hdr", format="HDR-FI")
    im507_hdr = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/507_convertigo.hdr", format="HDR-FI")
    im507_ = imageio.imread("/Users/yaelvinker/Documents/university/lab/open_exr_images_tests//507.exr",
                            format="EXR-FI")
    # im507_ = cv2.imread("/Users/yaelvinker/PycharmProjects/lab/utils/507.exr", -1)
    # im507_hdr = im507_hdr / np.max(im507_hdr)
    # im507_hdr = np.log((im507_ / np.max(im507_)) * 1000 + 1)
    # im507_hdr = im507_hdr / np.max(im507_hdr)
    # hdr_image_util.print_image_details(im507_hdr, "hdr")
    hdr_image_util.print_image_details(im507_hdr, "hdr online")
    hdr_image_util.print_image_details(im507, "hdr")
    hdr_image_util.print_image_details(im507_, "exr")
    im507 = np.log((im507_ / np.max(im507_)) * 1000 + 1)
    im507 = im507 / np.max(im507)
    print(im507.shape)

    # im507_crop = im507_hdr[1500:2500, 0:1500, :]
    # im507_crop_ = im507_hdr[1500:2500, 0:1500, :]
    im507_crop = im507_ - np.min(im507_)
    im507_crop_ = im507_
    # im507 = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/507_no_bgr.hdr", format="HDR-FI")

    # im507 = np.log((im507 / np.max(im507)) * 1000 + 1)
    # im507 = im507 / np.max(im507)
    plt.subplot(1, 2, 1)
    plt.imshow(im507_hdr)
    plt.subplot(1, 2, 2)
    plt.imshow(im507_crop)
    plt.show()

    # original_im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/507.exr", format="EXR-FI")
    # original_im = cv2.imread("/Users/yaelvinker/PycharmProjects/lab/utils/507.exr", -1)
    # im_bgr = cv2.cvtColor(original_im, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("/Users/yaelvinker/PycharmProjects/lab/utils/507_bgr.hdr", im_bgr)
    # cv2.imwrite("/Users/yaelvinker/PycharmProjects/lab/utils/507_no_bgr.hdr", original_im)

    # original_im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data/ldr_data/ldr_data/im_96.bmp")
    # im_bgr = cv2.cvtColor(original_im, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("/Users/yaelvinker/PycharmProjects/lab/data/ldr_data/ldr_data/im_96.jpg", im_bgr)
    # im2 = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data/ldr_data/ldr_data/im_96.jpg")
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_im)
    # plt.subplot(1,2,2)
    # plt.imshow(im2)
    # plt.show()


def get_bump(im):
    import numpy as np
    tmp = im
    tmp[(tmp > 200) != 0] = 255
    tmp = tmp.astype('uint8')

    hist, bins = np.histogram(tmp, bins=255)

    a0 = np.mean(hist[0:64])
    a1 = np.mean(hist[65:200])
    return a1 / a0


def find_f(im):
    import numpy as np
    im = im / np.max(im) * 255
    big = 1.1
    f = 1.0

    for i in range(1000):
        r = get_bump(im * f)
        if r < big:
            f = f * 1.01
        else:
            if r > 1 / big:
                break
    return f


def f_factor_test():
    import skimage
    exr_path = "/Users/yaelvinker/Documents/university/lab/open_exr_images_tests/exr_format_fixed_size/"
    b_factor_output_dir = os.path.join(
        "/Users/yaelvinker/Documents/university/lab/open_exr_images_tests//brightness_factors.npy")
    data = np.load(b_factor_output_dir, allow_pickle=True)[()]
    f_TunnelView = data["TunnelView(1).hdr"]
    f_507 = data["507.hdr"]
    f_RoundStoneBarn = data["RoundStoneBarn.hdr"]

    im_507 = imageio.imread(os.path.join(exr_path, "507.exr"), format="EXR-FI")

    im_TunnelView = imageio.imread(os.path.join(exr_path, "TunnelView(1).exr"), format="EXR-FI")
    im_RoundStoneBarn = imageio.imread(os.path.join(exr_path, "RoundStoneBarn.exr"), format="EXR-FI")
    im_507 = im_507 - np.min(im_507)
    im_RoundStoneBarn_reshape = skimage.transform.resize(im_RoundStoneBarn, (int(im_RoundStoneBarn.shape[0] / 4),
                                                                             int(im_RoundStoneBarn.shape[1] / 4)),
                                                         mode='reflect', preserve_range=False).astype("float32")
    im_TunnelView_reshape = skimage.transform.resize(im_TunnelView, (int(im_TunnelView.shape[0] / 4),
                                                                     int(im_TunnelView.shape[1] / 4)),
                                                     mode='reflect', preserve_range=False).astype("float32")
    im_507_reshape = skimage.transform.resize(im_507, (int(im_507.shape[0] / 4),
                                                       int(im_507.shape[1] / 4)),
                                              mode='reflect', preserve_range=False).astype("float32")
    print(f_507)
    print(f_RoundStoneBarn)
    print(f_TunnelView)
    print()
    new_f_507 = find_f(im_507)
    new_f_RoundStoneBarn = find_f(im_RoundStoneBarn)
    new_f_TunnelView = find_f(im_TunnelView)
    print(new_f_507)
    print(new_f_RoundStoneBarn)
    print(new_f_TunnelView)
    im_TunnelView = im_TunnelView - np.min(im_TunnelView)
    im_RoundStoneBarn = im_RoundStoneBarn - np.min(im_RoundStoneBarn)
    hdr_image_util.print_image_details(im_507, "im_507")
    hdr_image_util.print_image_details(im_TunnelView, "im_TunnelView")
    hdr_image_util.print_image_details(im_RoundStoneBarn, "im_RoundStoneBarn")
    im_507 = np.log((im_507 / np.max(im_507)) * f_507 * 255 + 1) / np.log(f_507 * 255 + 1)
    im_RoundStoneBarn_new = np.log(
        (im_RoundStoneBarn / np.max(im_RoundStoneBarn)) * new_f_RoundStoneBarn * 255 + 1) / np.log(
        new_f_RoundStoneBarn * 255 + 1)
    im_RoundStoneBarn = np.log(
        (im_RoundStoneBarn / np.max(im_RoundStoneBarn)) * f_RoundStoneBarn * 255 + 1) / np.log(
        f_RoundStoneBarn * 255 + 1)

    im_TunnelView = np.log((im_TunnelView / np.max(im_TunnelView)) * f_TunnelView * 255 + 1) / np.log(
        f_TunnelView * 255 + 1)
    hdr_image_util.print_image_details(im_507, "im_507")
    hdr_image_util.print_image_details(im_TunnelView, "im_TunnelView")
    hdr_image_util.print_image_details(im_RoundStoneBarn, "im_RoundStoneBarn")
    hdr_image_util.print_image_details(im_RoundStoneBarn_new, "im_RoundStoneBarn")
    plt.subplot(3, 1, 1)
    plt.imshow(im_RoundStoneBarn_new)
    plt.subplot(3, 1, 2)
    plt.imshow(im_TunnelView)
    plt.subplot(3, 1, 3)
    plt.imshow(im_RoundStoneBarn)
    plt.show()

def parse_nima_output():
    output = [
  {
    "image_id": "WallDrug",
    "mean_score_prediction": 4.745138734579086
  },
  {
    "image_id": "SunsetPoint(2)",
    "mean_score_prediction": 5.313142746686935
  },
  {
    "image_id": "TaughannockFalls",
    "mean_score_prediction": 4.682181499898434
  },
  {
    "image_id": "KingsCanyon",
    "mean_score_prediction": 5.17176316678524
  },
  {
    "image_id": "WillySentinel",
    "mean_score_prediction": 5.737436603754759
  },
  {
    "image_id": "GeneralSherman",
    "mean_score_prediction": 5.431764550507069
  },
  {
    "image_id": "TupperLake(2)",
    "mean_score_prediction": 5.031382076442242
  },
  {
    "image_id": "OCanadaLights",
    "mean_score_prediction": 5.467333905398846
  },
  {
    "image_id": "LabWindow",
    "mean_score_prediction": 5.855379443615675
  },
  {
    "image_id": "HallofFame",
    "mean_score_prediction": 5.55325984954834
  },
  {
    "image_id": "WillyDesk",
    "mean_score_prediction": 6.004342528060079
  },
  {
    "image_id": "CemeteryTree(1)",
    "mean_score_prediction": 4.604124039411545
  },
  {
    "image_id": "OtterPoint",
    "mean_score_prediction": 5.0836784690618515
  },
  {
    "image_id": "BarHarborPresunrise",
    "mean_score_prediction": 4.8247911632061005
  },
  {
    "image_id": "Route66Museum",
    "mean_score_prediction": 5.977623842656612
  },
  {
    "image_id": "WestBranchAusable(1)",
    "mean_score_prediction": 5.215016715228558
  },
  {
    "image_id": "RoundBarnInside",
    "mean_score_prediction": 5.285686329007149
  },
  {
    "image_id": "URChapel(1)",
    "mean_score_prediction": 5.157505020499229
  },
  {
    "image_id": "DevilsBathtub",
    "mean_score_prediction": 5.314411774277687
  },
  {
    "image_id": "RoundStoneBarn",
    "mean_score_prediction": 4.968158543109894
  },
  {
    "image_id": "HooverGarage",
    "mean_score_prediction": 5.0409155786037445
  },
  {
    "image_id": "BandonSunset(2)",
    "mean_score_prediction": 5.255815587937832
  },
  {
    "image_id": "MiddlePond",
    "mean_score_prediction": 4.962006606161594
  },
  {
    "image_id": "MasonLake(2)",
    "mean_score_prediction": 4.795713901519775
  },
  {
    "image_id": "Flamingo",
    "mean_score_prediction": 5.471691556274891
  },
  {
    "image_id": "GoldenGate(1)",
    "mean_score_prediction": 5.542832113802433
  },
  {
    "image_id": "Exploratorium(2)",
    "mean_score_prediction": 5.880201011896133
  },
  {
    "image_id": "PeckLake",
    "mean_score_prediction": 5.478645049035549
  },
  {
    "image_id": "McKeesPub",
    "mean_score_prediction": 5.906127788126469
  },
  {
    "image_id": "MammothHotSprings",
    "mean_score_prediction": 4.580318421125412
  },
  {
    "image_id": "TunnelView(2)",
    "mean_score_prediction": 5.021063856780529
  },
  {
    "image_id": "LabBooth",
    "mean_score_prediction": 5.6617686823010445
  },
  {
    "image_id": "PaulBunyan",
    "mean_score_prediction": 5.41439026594162
  },
  {
    "image_id": "MtRushmore(2)",
    "mean_score_prediction": 5.128768265247345
  },
  {
    "image_id": "BloomingGorse(1)",
    "mean_score_prediction": 5.430323071777821
  },
  {
    "image_id": "BenJerrys",
    "mean_score_prediction": 5.994783543050289
  },
  {
    "image_id": "ElCapitan",
    "mean_score_prediction": 4.783578298985958
  },
  {
    "image_id": "BigfootPass",
    "mean_score_prediction": 4.733184292912483
  },
  {
    "image_id": "MirrorLake",
    "mean_score_prediction": 4.928409330546856
  },
  {
    "image_id": "AirBellowsGap",
    "mean_score_prediction": 5.310614064335823
  },
  {
    "image_id": "LittleRiver",
    "mean_score_prediction": 4.88445320725441
  },
  {
    "image_id": "HancockSeedField",
    "mean_score_prediction": 4.963666260242462
  },
  {
    "image_id": "TheNarrows(1)",
    "mean_score_prediction": 5.134310662746429
  },
  {
    "image_id": "TheGrotto",
    "mean_score_prediction": 4.803582668304443
  },
  {
    "image_id": "LetchworthTeaTable(1)",
    "mean_score_prediction": 5.41020805388689
  },
  {
    "image_id": "RoadsEndFireDamage",
    "mean_score_prediction": 5.183445505797863
  },
  {
    "image_id": "AmikeusBeaverDamPM2",
    "mean_score_prediction": 4.5778296291828156
  },
  {
    "image_id": "RedwoodSunset",
    "mean_score_prediction": 4.362965613603592
  },
  {
    "image_id": "Frontier",
    "mean_score_prediction": 4.989865064620972
  },
  {
    "image_id": "DelicateFlowers",
    "mean_score_prediction": 4.823823198676109
  },
  {
    "image_id": "LadyBirdRedwoods",
    "mean_score_prediction": 5.157182976603508
  },
  {
    "image_id": "MackinacBridge",
    "mean_score_prediction": 5.761691376566887
  },
  {
    "image_id": "M3MiddlePond",
    "mean_score_prediction": 5.609974253922701
  },
  {
    "image_id": "NorthBubble",
    "mean_score_prediction": 5.1318783685564995
  },
  {
    "image_id": "AmikeusBeaverDamPM1",
    "mean_score_prediction": 4.824190109968185
  },
  {
    "image_id": "MtRushmoreFlags",
    "mean_score_prediction": 4.657231226563454
  },
  {
    "image_id": "OldFaithfulInn",
    "mean_score_prediction": 5.482168275862932
  },
  {
    "image_id": "507_durand",
    "mean_score_prediction": 5.130073726177216
  },
  {
    "image_id": "LabTypewriter",
    "mean_score_prediction": 5.5104410499334335
  },
  {
    "image_id": "HDRMark",
    "mean_score_prediction": 5.810696180909872
  },
  {
    "image_id": "RITTiger",
    "mean_score_prediction": 4.837134562432766
  },
  {
    "image_id": "LetchworthTeaTable(2)",
    "mean_score_prediction": 5.371234133839607
  },
  {
    "image_id": "TheNarrows(3)",
    "mean_score_prediction": 5.012859724462032
  },
  {
    "image_id": "UpheavalDome",
    "mean_score_prediction": 4.9012769386172295
  },
  {
    "image_id": "BloomingGorse(2)",
    "mean_score_prediction": 5.206291787326336
  },
  {
    "image_id": "CadesCove",
    "mean_score_prediction": 5.120595194399357
  },
  {
    "image_id": "Peppermill",
    "mean_score_prediction": 5.109213680028915
  },
  {
    "image_id": "TheNarrows(2)",
    "mean_score_prediction": 4.386589162051678
  },
  {
    "image_id": "CanadianFalls",
    "mean_score_prediction": 5.019321598112583
  },
  {
    "image_id": "MasonLake(1)",
    "mean_score_prediction": 4.881610043346882
  },
  {
    "image_id": "Exploratorium(1)",
    "mean_score_prediction": 5.213731154799461
  },
  {
    "image_id": "GoldenGate(2)",
    "mean_score_prediction": 5.18636754155159
  },
  {
    "image_id": "OCanadaNoLights",
    "mean_score_prediction": 5.286436006426811
  },
  {
    "image_id": "DevilsTower",
    "mean_score_prediction": 4.696122892200947
  },
  {
    "image_id": "HalfDomeSunset",
    "mean_score_prediction": 5.018310450017452
  },
  {
    "image_id": "TunnelView(1)",
    "mean_score_prediction": 4.8241138979792595
  },
  {
    "image_id": "SouthBranchKingsRiver",
    "mean_score_prediction": 5.108864486217499
  },
  {
    "image_id": "MtRushmore(1)",
    "mean_score_prediction": 4.919717520475388
  },
  {
    "image_id": "SequoiaRemains",
    "mean_score_prediction": 4.158863946795464
  },
  {
    "image_id": "DelicateArch",
    "mean_score_prediction": 5.138658564537764
  },
  {
    "image_id": "URChapel(2)",
    "mean_score_prediction": 5.353516578674316
  },
  {
    "image_id": "AhwahneeGreatLounge",
    "mean_score_prediction": 5.671114020049572
  },
  {
    "image_id": "BandonSunset(1)",
    "mean_score_prediction": 5.514545053243637
  },
  {
    "image_id": "HancockKitchenOutside",
    "mean_score_prediction": 5.750345703214407
  },
  {
    "image_id": "HooverDam",
    "mean_score_prediction": 5.100682832300663
  },
  {
    "image_id": "WaffleHouse",
    "mean_score_prediction": 5.0881843864917755
  },
  {
    "image_id": "GeneralGrant",
    "mean_score_prediction": 5.023101158440113
  },
  {
    "image_id": "HancockKitchenInside",
    "mean_score_prediction": 5.714430693536997
  },
  {
    "image_id": "WestBranchAusable(2)",
    "mean_score_prediction": 4.729960784316063
  },
  {
    "image_id": "BalancedRock",
    "mean_score_prediction": 4.863842889666557
  },
  {
    "image_id": "DevilsGolfCourse",
    "mean_score_prediction": 4.83895930275321
  },
  {
    "image_id": "Zentrum",
    "mean_score_prediction": 4.492658250033855
  },
  {
    "image_id": "NiagaraFalls",
    "mean_score_prediction": 5.063793130218983
  },
  {
    "image_id": "ArtistPalette",
    "mean_score_prediction": 4.985267795622349
  },
  {
    "image_id": "TupperLake(1)",
    "mean_score_prediction": 5.318654768168926
  },
  {
    "image_id": "Petroglyphs",
    "mean_score_prediction": 4.871259339153767
  },
  {
    "image_id": "CemeteryTree(2)",
    "mean_score_prediction": 5.22406130284071
  },
  {
    "image_id": "BarHarborSunrise",
    "mean_score_prediction": 5.367681674659252
  },
  {
    "image_id": "LasVegasStore",
    "mean_score_prediction": 5.210922617465258
  },
  {
    "image_id": "FourCornersStorm",
    "mean_score_prediction": 5.452222965657711
  },
  {
    "image_id": "SunsetPoint(1)",
    "mean_score_prediction": 4.48302998393774
  },
  {
    "image_id": "SmokyTunnel",
    "mean_score_prediction": 4.917546816170216
  },
  {
    "image_id": "LuxoDoubleChecker",
    "mean_score_prediction": 4.960071079432964
  },
  {
    "image_id": "YosemiteFalls",
    "mean_score_prediction": 4.688029170036316
  }
]
    output2 = [
  {
    "image_id": "WallDrug",
    "mean_score_prediction": 4.968319430947304
  },
  {
    "image_id": "SunsetPoint(2)",
    "mean_score_prediction": 5.624675627797842
  },
  {
    "image_id": "TaughannockFalls",
    "mean_score_prediction": 4.3708068281412125
  },
  {
    "image_id": "KingsCanyon",
    "mean_score_prediction": 4.996549427509308
  },
  {
    "image_id": "WillySentinel",
    "mean_score_prediction": 5.792388334870338
  },
  {
    "image_id": "GeneralSherman",
    "mean_score_prediction": 5.384132355451584
  },
  {
    "image_id": "TupperLake(2)",
    "mean_score_prediction": 4.861401669681072
  },
  {
    "image_id": "OCanadaLights",
    "mean_score_prediction": 5.615086808800697
  },
  {
    "image_id": "LabWindow",
    "mean_score_prediction": 5.21818009018898
  },
  {
    "image_id": "HallofFame",
    "mean_score_prediction": 5.866032775491476
  },
  {
    "image_id": "WillyDesk",
    "mean_score_prediction": 5.761138113215566
  },
  {
    "image_id": "CemeteryTree(1)",
    "mean_score_prediction": 4.587957717478275
  },
  {
    "image_id": "OtterPoint",
    "mean_score_prediction": 5.344192832708359
  },
  {
    "image_id": "BarHarborPresunrise",
    "mean_score_prediction": 4.790673956274986
  },
  {
    "image_id": "Route66Museum",
    "mean_score_prediction": 5.758112292736769
  },
  {
    "image_id": "WestBranchAusable(1)",
    "mean_score_prediction": 4.707668304443359
  },
  {
    "image_id": "RoundBarnInside",
    "mean_score_prediction": 5.042449787259102
  },
  {
    "image_id": "URChapel(1)",
    "mean_score_prediction": 4.825338780879974
  },
  {
    "image_id": "DevilsBathtub",
    "mean_score_prediction": 5.2285747937858105
  },
  {
    "image_id": "RoundStoneBarn",
    "mean_score_prediction": 5.2954848781228065
  },
  {
    "image_id": "HooverGarage",
    "mean_score_prediction": 4.964006267488003
  },
  {
    "image_id": "BandonSunset(2)",
    "mean_score_prediction": 4.992149330675602
  },
  {
    "image_id": "MiddlePond",
    "mean_score_prediction": 4.839066803455353
  },
  {
    "image_id": "MasonLake(2)",
    "mean_score_prediction": 5.037300296127796
  },
  {
    "image_id": "Flamingo",
    "mean_score_prediction": 5.158975370228291
  },
  {
    "image_id": "GoldenGate(1)",
    "mean_score_prediction": 5.574511922895908
  },
  {
    "image_id": "Exploratorium(2)",
    "mean_score_prediction": 5.6576830223202705
  },
  {
    "image_id": "PeckLake",
    "mean_score_prediction": 5.360576696693897
  },
  {
    "image_id": "McKeesPub",
    "mean_score_prediction": 5.382652413100004
  },
  {
    "image_id": "MammothHotSprings",
    "mean_score_prediction": 4.840903148055077
  },
  {
    "image_id": "TunnelView(2)",
    "mean_score_prediction": 4.900018155574799
  },
  {
    "image_id": "LabBooth",
    "mean_score_prediction": 5.2703783214092255
  },
  {
    "image_id": "PaulBunyan",
    "mean_score_prediction": 5.292959116399288
  },
  {
    "image_id": "MtRushmore(2)",
    "mean_score_prediction": 5.198314130306244
  },
  {
    "image_id": "BloomingGorse(1)",
    "mean_score_prediction": 5.383135177195072
  },
  {
    "image_id": "BenJerrys",
    "mean_score_prediction": 5.739915832877159
  },
  {
    "image_id": "ElCapitan",
    "mean_score_prediction": 4.805858165025711
  },
  {
    "image_id": "BigfootPass",
    "mean_score_prediction": 4.65417967736721
  },
  {
    "image_id": "MirrorLake",
    "mean_score_prediction": 4.604226842522621
  },
  {
    "image_id": "AirBellowsGap",
    "mean_score_prediction": 5.596962846815586
  },
  {
    "image_id": "LittleRiver",
    "mean_score_prediction": 4.991946421563625
  },
  {
    "image_id": "HancockSeedField",
    "mean_score_prediction": 5.057914689183235
  },
  {
    "image_id": "TheNarrows(1)",
    "mean_score_prediction": 5.018363252282143
  },
  {
    "image_id": "TheGrotto",
    "mean_score_prediction": 5.0244865864515305
  },
  {
    "image_id": "LetchworthTeaTable(1)",
    "mean_score_prediction": 5.001456126570702
  },
  {
    "image_id": "RoadsEndFireDamage",
    "mean_score_prediction": 5.096344374120235
  },
  {
    "image_id": "AmikeusBeaverDamPM2",
    "mean_score_prediction": 4.888666160404682
  },
  {
    "image_id": "RedwoodSunset",
    "mean_score_prediction": 4.896033555269241
  },
  {
    "image_id": "Frontier",
    "mean_score_prediction": 5.005969829857349
  },
  {
    "image_id": "DelicateFlowers",
    "mean_score_prediction": 4.890189848840237
  },
  {
    "image_id": "LadyBirdRedwoods",
    "mean_score_prediction": 4.929734744131565
  },
  {
    "image_id": "MackinacBridge",
    "mean_score_prediction": 5.84359186142683
  },
  {
    "image_id": "M3MiddlePond",
    "mean_score_prediction": 5.495171424001455
  },
  {
    "image_id": "NorthBubble",
    "mean_score_prediction": 4.968926049768925
  },
  {
    "image_id": "AmikeusBeaverDamPM1",
    "mean_score_prediction": 4.964521490037441
  },
  {
    "image_id": "MtRushmoreFlags",
    "mean_score_prediction": 4.804915376007557
  },
  {
    "image_id": "OldFaithfulInn",
    "mean_score_prediction": 5.528424113988876
  },
  {
    "image_id": "LabTypewriter",
    "mean_score_prediction": 4.990197770297527
  },
  {
    "image_id": "HDRMark",
    "mean_score_prediction": 5.431831553578377
  },
  {
    "image_id": "RITTiger",
    "mean_score_prediction": 4.869820836931467
  },
  {
    "image_id": "LetchworthTeaTable(2)",
    "mean_score_prediction": 5.051731884479523
  },
  {
    "image_id": "TheNarrows(3)",
    "mean_score_prediction": 4.9205373376607895
  },
  {
    "image_id": "UpheavalDome",
    "mean_score_prediction": 4.688512057065964
  },
  {
    "image_id": "BloomingGorse(2)",
    "mean_score_prediction": 5.488514922559261
  },
  {
    "image_id": "CadesCove",
    "mean_score_prediction": 5.427744477987289
  },
  {
    "image_id": "Peppermill",
    "mean_score_prediction": 4.811564974486828
  },
  {
    "image_id": "TheNarrows(2)",
    "mean_score_prediction": 4.274650387465954
  },
  {
    "image_id": "CanadianFalls",
    "mean_score_prediction": 4.840592086315155
  },
  {
    "image_id": "MasonLake(1)",
    "mean_score_prediction": 5.058977656066418
  },
  {
    "image_id": "Exploratorium(1)",
    "mean_score_prediction": 5.213787786662579
  },
  {
    "image_id": "GoldenGate(2)",
    "mean_score_prediction": 5.649047542363405
  },
  {
    "image_id": "OCanadaNoLights",
    "mean_score_prediction": 5.431007616221905
  },
  {
    "image_id": "DevilsTower",
    "mean_score_prediction": 4.9003114476799965
  },
  {
    "image_id": "HalfDomeSunset",
    "mean_score_prediction": 5.062497168779373
  },
  {
    "image_id": "TunnelView(1)",
    "mean_score_prediction": 4.582784965634346
  },
  {
    "image_id": "SouthBranchKingsRiver",
    "mean_score_prediction": 4.546284548938274
  },
  {
    "image_id": "MtRushmore(1)",
    "mean_score_prediction": 4.6995320320129395
  },
  {
    "image_id": "SequoiaRemains",
    "mean_score_prediction": 5.109896384179592
  },
  {
    "image_id": "DelicateArch",
    "mean_score_prediction": 5.0498221591115
  },
  {
    "image_id": "URChapel(2)",
    "mean_score_prediction": 5.120395481586456
  },
  {
    "image_id": "AhwahneeGreatLounge",
    "mean_score_prediction": 5.4250946044921875
  },
  {
    "image_id": "BandonSunset(1)",
    "mean_score_prediction": 5.183998391032219
  },
  {
    "image_id": "HancockKitchenOutside",
    "mean_score_prediction": 5.552189700305462
  },
  {
    "image_id": "HooverDam",
    "mean_score_prediction": 4.864067509770393
  },
  {
    "image_id": "WaffleHouse",
    "mean_score_prediction": 4.897450812160969
  },
  {
    "image_id": "GeneralGrant",
    "mean_score_prediction": 4.962745942175388
  },
  {
    "image_id": "HancockKitchenInside",
    "mean_score_prediction": 5.560864396393299
  },
  {
    "image_id": "WestBranchAusable(2)",
    "mean_score_prediction": 4.601587802171707
  },
  {
    "image_id": "BalancedRock",
    "mean_score_prediction": 4.7448092848062515
  },
  {
    "image_id": "DevilsGolfCourse",
    "mean_score_prediction": 4.581669073551893
  },
  {
    "image_id": "Zentrum",
    "mean_score_prediction": 4.184578370302916
  },
  {
    "image_id": "NiagaraFalls",
    "mean_score_prediction": 4.797078497707844
  },
  {
    "image_id": "ArtistPalette",
    "mean_score_prediction": 5.258216168731451
  },
  {
    "image_id": "TupperLake(1)",
    "mean_score_prediction": 5.421011060476303
  },
  {
    "image_id": "Petroglyphs",
    "mean_score_prediction": 5.25830290466547
  },
  {
    "image_id": "CemeteryTree(2)",
    "mean_score_prediction": 5.138535745441914
  },
  {
    "image_id": "BarHarborSunrise",
    "mean_score_prediction": 5.095343686640263
  },
  {
    "image_id": "LasVegasStore",
    "mean_score_prediction": 5.037281580269337
  },
  {
    "image_id": "FourCornersStorm",
    "mean_score_prediction": 5.584082044661045
  },
  {
    "image_id": "SunsetPoint(1)",
    "mean_score_prediction": 4.929943315684795
  },
  {
    "image_id": "SmokyTunnel",
    "mean_score_prediction": 4.7213203981518745
  },
  {
    "image_id": "LuxoDoubleChecker",
    "mean_score_prediction": 4.75617778301239
  },
  {
    "image_id": "YosemiteFalls",
    "mean_score_prediction": 4.9610453844070435
  }
]
    output3 = [
  {
    "image_id": "HooverGarage_pre",
    "mean_score_prediction": 4.863120324909687
  },
  {
    "image_id": "WestBranchAusable(1)_pre",
    "mean_score_prediction": 4.7225499749183655
  },
  {
    "image_id": "PaulBunyan_pre",
    "mean_score_prediction": 5.4582453444600105
  },
  {
    "image_id": "BigfootPass_pre",
    "mean_score_prediction": 4.314834225922823
  },
  {
    "image_id": "UpheavalDome_pre",
    "mean_score_prediction": 4.82876106351614
  },
  {
    "image_id": "HallofFame_pre",
    "mean_score_prediction": 5.900733336806297
  },
  {
    "image_id": "LetchworthTeaTable(1)_pre",
    "mean_score_prediction": 5.041259616613388
  },
  {
    "image_id": "McKeesPub_pre",
    "mean_score_prediction": 5.677172772586346
  },
  {
    "image_id": "Petroglyphs_pre",
    "mean_score_prediction": 5.002979584038258
  },
  {
    "image_id": "WillySentinel_pre",
    "mean_score_prediction": 5.512386336922646
  },
  {
    "image_id": "RoundBarnInside_pre",
    "mean_score_prediction": 5.086464278399944
  },
  {
    "image_id": "HancockSeedField_pre",
    "mean_score_prediction": 4.857205048203468
  },
  {
    "image_id": "AmikeusBeaverDamPM2_pre",
    "mean_score_prediction": 4.753152787685394
  },
  {
    "image_id": "BalancedRock_pre",
    "mean_score_prediction": 4.824850849807262
  },
  {
    "image_id": "MtRushmore(2)_pre",
    "mean_score_prediction": 5.1479900404810905
  },
  {
    "image_id": "DevilsTower_pre",
    "mean_score_prediction": 4.690601073205471
  },
  {
    "image_id": "NiagaraFalls_pre",
    "mean_score_prediction": 4.90588815510273
  },
  {
    "image_id": "Zentrum_pre",
    "mean_score_prediction": 4.493002399802208
  },
  {
    "image_id": "MasonLake(1)_pre",
    "mean_score_prediction": 4.981562502682209
  },
  {
    "image_id": "MiddlePond_pre",
    "mean_score_prediction": 4.691436029970646
  },
  {
    "image_id": "CadesCove_pre",
    "mean_score_prediction": 5.28689730912447
  },
  {
    "image_id": "GeneralSherman_pre",
    "mean_score_prediction": 5.348544545471668
  },
  {
    "image_id": "CemeteryTree(1)_pre",
    "mean_score_prediction": 4.64660058170557
  },
  {
    "image_id": "Peppermill_pre",
    "mean_score_prediction": 5.0910501554608345
  },
  {
    "image_id": "TheNarrows(2)_pre",
    "mean_score_prediction": 4.111482813954353
  },
  {
    "image_id": "TheNarrows(3)_pre",
    "mean_score_prediction": 4.939015783369541
  },
  {
    "image_id": "MackinacBridge_pre",
    "mean_score_prediction": 5.532604046165943
  },
  {
    "image_id": "AirBellowsGap_pre",
    "mean_score_prediction": 5.290933422744274
  },
  {
    "image_id": "507_durand_lumincance",
    "mean_score_prediction": 5.161415785551071
  },
  {
    "image_id": "LabWindow_pre",
    "mean_score_prediction": 5.308989606797695
  },
  {
    "image_id": "DevilsBathtub_pre",
    "mean_score_prediction": 5.4615392461419106
  },
  {
    "image_id": "GoldenGate(1)_pre",
    "mean_score_prediction": 5.6396181508898735
  },
  {
    "image_id": "LabBooth_pre",
    "mean_score_prediction": 5.74685237929225
  },
  {
    "image_id": "PeckLake_pre",
    "mean_score_prediction": 5.555454224348068
  },
  {
    "image_id": "SequoiaRemains_pre",
    "mean_score_prediction": 4.722694434225559
  },
  {
    "image_id": "MtRushmoreFlags_pre",
    "mean_score_prediction": 4.991321213543415
  },
  {
    "image_id": "TunnelView(2)_pre",
    "mean_score_prediction": 5.109981916844845
  },
  {
    "image_id": "DelicateArch_pre",
    "mean_score_prediction": 5.019728600978851
  },
  {
    "image_id": "GeneralGrant_pre",
    "mean_score_prediction": 4.9518856555223465
  },
  {
    "image_id": "FourCornersStorm_pre",
    "mean_score_prediction": 5.669017024338245
  },
  {
    "image_id": "KingsCanyon_pre",
    "mean_score_prediction": 4.990831382572651
  },
  {
    "image_id": "Route66Museum_pre",
    "mean_score_prediction": 5.537823636084795
  },
  {
    "image_id": "DevilsGolfCourse_pre",
    "mean_score_prediction": 3.939344495534897
  },
  {
    "image_id": "MammothHotSprings_pre",
    "mean_score_prediction": 4.842050559818745
  },
  {
    "image_id": "BloomingGorse(2)_pre",
    "mean_score_prediction": 5.296081371605396
  },
  {
    "image_id": "AmikeusBeaverDamPM1_pre",
    "mean_score_prediction": 4.7158588618040085
  },
  {
    "image_id": "LuxoDoubleChecker_pre",
    "mean_score_prediction": 5.10029574483633
  },
  {
    "image_id": "BandonSunset(2)_pre",
    "mean_score_prediction": 6.086960472166538
  },
  {
    "image_id": "AhwahneeGreatLounge_pre",
    "mean_score_prediction": 5.258347764611244
  },
  {
    "image_id": "WallDrug_pre",
    "mean_score_prediction": 4.675267696380615
  },
  {
    "image_id": "WillyDesk_pre",
    "mean_score_prediction": 5.887592317536473
  },
  {
    "image_id": "URChapel(1)_pre",
    "mean_score_prediction": 4.713813528418541
  },
  {
    "image_id": "SunsetPoint(1)_pre",
    "mean_score_prediction": 4.902209043502808
  },
  {
    "image_id": "NorthBubble_pre",
    "mean_score_prediction": 4.729524731636047
  },
  {
    "image_id": "BarHarborSunrise_pre",
    "mean_score_prediction": 5.3132007867097855
  },
  {
    "image_id": "Exploratorium(2)_pre",
    "mean_score_prediction": 5.6638611778616905
  },
  {
    "image_id": "LittleRiver_pre",
    "mean_score_prediction": 4.91639456897974
  },
  {
    "image_id": "TupperLake(1)_pre",
    "mean_score_prediction": 5.224244400858879
  },
  {
    "image_id": "MasonLake(2)_pre",
    "mean_score_prediction": 5.040285035967827
  },
  {
    "image_id": "ArtistPalette_pre",
    "mean_score_prediction": 4.95161297172308
  },
  {
    "image_id": "TaughannockFalls_pre",
    "mean_score_prediction": 4.601803652942181
  },
  {
    "image_id": "YosemiteFalls_pre",
    "mean_score_prediction": 5.013870485126972
  },
  {
    "image_id": "RedwoodSunset_pre",
    "mean_score_prediction": 4.789966829121113
  },
  {
    "image_id": "LadyBirdRedwoods_pre",
    "mean_score_prediction": 4.881509751081467
  },
  {
    "image_id": "CemeteryTree(2)_pre",
    "mean_score_prediction": 4.915265932679176
  },
  {
    "image_id": "MirrorLake_pre",
    "mean_score_prediction": 4.5144267082214355
  },
  {
    "image_id": "HooverDam_pre",
    "mean_score_prediction": 5.014296546578407
  },
  {
    "image_id": "BarHarborPresunrise_pre",
    "mean_score_prediction": 5.1143267676234245
  },
  {
    "image_id": "JesseBrownsCabin_pre",
    "mean_score_prediction": 4.8481588661670685
  },
  {
    "image_id": "LabTypewriter_pre",
    "mean_score_prediction": 5.113183960318565
  },
  {
    "image_id": "MtRushmore(1)_pre",
    "mean_score_prediction": 4.748633868992329
  },
  {
    "image_id": "WaffleHouse_pre",
    "mean_score_prediction": 4.867327384650707
  },
  {
    "image_id": "OCanadaNoLights_pre",
    "mean_score_prediction": 5.547234743833542
  },
  {
    "image_id": "LetchworthTeaTable(2)_pre",
    "mean_score_prediction": 5.317468322813511
  },
  {
    "image_id": "HancockKitchenInside_pre",
    "mean_score_prediction": 5.341871917247772
  },
  {
    "image_id": "CanadianFalls_pre",
    "mean_score_prediction": 4.918545454740524
  },
  {
    "image_id": "OldFaithfulInn_pre",
    "mean_score_prediction": 5.166047044098377
  },
  {
    "image_id": "Flamingo_pre",
    "mean_score_prediction": 4.783165723085403
  },
  {
    "image_id": "OCanadaLights_pre",
    "mean_score_prediction": 5.632857542484999
  },
  {
    "image_id": "RoadsEndFireDamage_pre",
    "mean_score_prediction": 4.912455923855305
  },
  {
    "image_id": "RITTiger_pre",
    "mean_score_prediction": 4.670452877879143
  },
  {
    "image_id": "WestBranchAusable(2)_pre",
    "mean_score_prediction": 4.643554732203484
  },
  {
    "image_id": "BandonSunset(1)_pre",
    "mean_score_prediction": 6.26833501458168
  },
  {
    "image_id": "LasVegasStore_pre",
    "mean_score_prediction": 4.547539655119181
  },
  {
    "image_id": "BenJerrys_pre",
    "mean_score_prediction": 5.529772091656923
  },
  {
    "image_id": "URChapel(2)_pre",
    "mean_score_prediction": 5.318016886711121
  },
  {
    "image_id": "SunsetPoint(2)_pre",
    "mean_score_prediction": 5.308567441999912
  },
  {
    "image_id": "RoundStoneBarn_pre",
    "mean_score_prediction": 5.509073067456484
  },
  {
    "image_id": "HancockKitchenOutside_pre",
    "mean_score_prediction": 5.462620824575424
  },
  {
    "image_id": "M3MiddlePond_pre",
    "mean_score_prediction": 5.460015624761581
  },
  {
    "image_id": "Exploratorium(1)_pre",
    "mean_score_prediction": 5.113989517092705
  },
  {
    "image_id": "TupperLake(2)_pre",
    "mean_score_prediction": 5.350640319287777
  },
  {
    "image_id": "ElCapitan_pre",
    "mean_score_prediction": 4.948391504585743
  },
  {
    "image_id": "HDRMark_pre",
    "mean_score_prediction": 5.609394770115614
  },
  {
    "image_id": "BloomingGorse(1)_pre",
    "mean_score_prediction": 5.298347435891628
  },
  {
    "image_id": "SouthBranchKingsRiver_pre",
    "mean_score_prediction": 5.221607483923435
  },
  {
    "image_id": "HalfDomeSunset_pre",
    "mean_score_prediction": 5.086885251104832
  },
  {
    "image_id": "TunnelView(1)_pre",
    "mean_score_prediction": 4.800393432378769
  },
  {
    "image_id": "OtterPoint_pre",
    "mean_score_prediction": 5.288020446896553
  },
  {
    "image_id": "DelicateFlowers_pre",
    "mean_score_prediction": 4.57290306687355
  },
  {
    "image_id": "TheGrotto_pre",
    "mean_score_prediction": 4.868299871683121
  },
  {
    "image_id": "TheNarrows(1)_pre",
    "mean_score_prediction": 5.351073145866394
  },
  {
    "image_id": "Frontier_pre",
    "mean_score_prediction": 4.813003111630678
  },
  {
    "image_id": "SmokyTunnel_pre",
    "mean_score_prediction": 4.40847934782505
  },
  {
    "image_id": "GoldenGate(2)_pre",
    "mean_score_prediction": 5.039037823677063
  }
]
    output4 = [
  {
    "image_id": "WallDrug",
    "mean_score_prediction": 4.907476380467415
  },
  {
    "image_id": "SunsetPoint(2)",
    "mean_score_prediction": 5.663844849914312
  },
  {
    "image_id": "TaughannockFalls",
    "mean_score_prediction": 4.337828632444143
  },
  {
    "image_id": "KingsCanyon",
    "mean_score_prediction": 4.946410194039345
  },
  {
    "image_id": "WillySentinel",
    "mean_score_prediction": 5.563055370002985
  },
  {
    "image_id": "GeneralSherman",
    "mean_score_prediction": 5.311344154179096
  },
  {
    "image_id": "TupperLake(2)",
    "mean_score_prediction": 5.151613458991051
  },
  {
    "image_id": "OCanadaLights",
    "mean_score_prediction": 5.641111675649881
  },
  {
    "image_id": "LabWindow",
    "mean_score_prediction": 5.558731064200401
  },
  {
    "image_id": "HallofFame",
    "mean_score_prediction": 5.999541021883488
  },
  {
    "image_id": "WillyDesk",
    "mean_score_prediction": 5.9529784601181746
  },
  {
    "image_id": "CemeteryTree(1)",
    "mean_score_prediction": 4.721557162702084
  },
  {
    "image_id": "OtterPoint",
    "mean_score_prediction": 5.189788907766342
  },
  {
    "image_id": "BarHarborPresunrise",
    "mean_score_prediction": 4.54036033898592
  },
  {
    "image_id": "Route66Museum",
    "mean_score_prediction": 5.848162945359945
  },
  {
    "image_id": "WestBranchAusable(1)",
    "mean_score_prediction": 4.975213930010796
  },
  {
    "image_id": "RoundBarnInside",
    "mean_score_prediction": 5.114598639309406
  },
  {
    "image_id": "URChapel(1)",
    "mean_score_prediction": 5.050585933029652
  },
  {
    "image_id": "DevilsBathtub",
    "mean_score_prediction": 5.20540663599968
  },
  {
    "image_id": "RoundStoneBarn",
    "mean_score_prediction": 5.180934064090252
  },
  {
    "image_id": "HooverGarage",
    "mean_score_prediction": 4.823288656771183
  },
  {
    "image_id": "BandonSunset(2)",
    "mean_score_prediction": 4.610946454107761
  },
  {
    "image_id": "MiddlePond",
    "mean_score_prediction": 5.036846220493317
  },
  {
    "image_id": "MasonLake(2)",
    "mean_score_prediction": 5.2148974016308784
  },
  {
    "image_id": "Flamingo",
    "mean_score_prediction": 5.111302524805069
  },
  {
    "image_id": "GoldenGate(1)",
    "mean_score_prediction": 5.235531400889158
  },
  {
    "image_id": "Exploratorium(2)",
    "mean_score_prediction": 5.778527166694403
  },
  {
    "image_id": "PeckLake",
    "mean_score_prediction": 5.717144005000591
  },
  {
    "image_id": "McKeesPub",
    "mean_score_prediction": 5.559501089155674
  },
  {
    "image_id": "MammothHotSprings",
    "mean_score_prediction": 4.7370157688856125
  },
  {
    "image_id": "TunnelView(2)",
    "mean_score_prediction": 4.751212865114212
  },
  {
    "image_id": "LabBooth",
    "mean_score_prediction": 5.2089640609920025
  },
  {
    "image_id": "PaulBunyan",
    "mean_score_prediction": 5.345098085701466
  },
  {
    "image_id": "MtRushmore(2)",
    "mean_score_prediction": 5.034677520394325
  },
  {
    "image_id": "BloomingGorse(1)",
    "mean_score_prediction": 5.538611859083176
  },
  {
    "image_id": "BenJerrys",
    "mean_score_prediction": 5.544798728078604
  },
  {
    "image_id": "ElCapitan",
    "mean_score_prediction": 4.892853058874607
  },
  {
    "image_id": "BigfootPass",
    "mean_score_prediction": 4.61651062220335
  },
  {
    "image_id": "MirrorLake",
    "mean_score_prediction": 4.557143330574036
  },
  {
    "image_id": "AirBellowsGap",
    "mean_score_prediction": 4.993483409285545
  },
  {
    "image_id": "LittleRiver",
    "mean_score_prediction": 4.982271328568459
  },
  {
    "image_id": "507",
    "mean_score_prediction": 5.434465732425451
  },
  {
    "image_id": "HancockSeedField",
    "mean_score_prediction": 5.346922658383846
  },
  {
    "image_id": "TheNarrows(1)",
    "mean_score_prediction": 5.236126482486725
  },
  {
    "image_id": "TheGrotto",
    "mean_score_prediction": 5.117239370942116
  },
  {
    "image_id": "LetchworthTeaTable(1)",
    "mean_score_prediction": 5.208363905549049
  },
  {
    "image_id": "RoadsEndFireDamage",
    "mean_score_prediction": 4.930053539574146
  },
  {
    "image_id": "AmikeusBeaverDamPM2",
    "mean_score_prediction": 4.796637684106827
  },
  {
    "image_id": "RedwoodSunset",
    "mean_score_prediction": 4.950362034142017
  },
  {
    "image_id": "Frontier",
    "mean_score_prediction": 5.3199391812086105
  },
  {
    "image_id": "DelicateFlowers",
    "mean_score_prediction": 4.978417754173279
  },
  {
    "image_id": "LadyBirdRedwoods",
    "mean_score_prediction": 5.231082119047642
  },
  {
    "image_id": "MackinacBridge",
    "mean_score_prediction": 5.792182870209217
  },
  {
    "image_id": "M3MiddlePond",
    "mean_score_prediction": 5.3795402981340885
  },
  {
    "image_id": "NorthBubble",
    "mean_score_prediction": 5.246401619166136
  },
  {
    "image_id": "AmikeusBeaverDamPM1",
    "mean_score_prediction": 4.682795651257038
  },
  {
    "image_id": "MtRushmoreFlags",
    "mean_score_prediction": 4.55965730547905
  },
  {
    "image_id": "OldFaithfulInn",
    "mean_score_prediction": 5.581838060170412
  },
  {
    "image_id": "LabTypewriter",
    "mean_score_prediction": 5.058225948363543
  },
  {
    "image_id": "HDRMark",
    "mean_score_prediction": 5.522270705550909
  },
  {
    "image_id": "RITTiger",
    "mean_score_prediction": 4.943303499370813
  },
  {
    "image_id": "LetchworthTeaTable(2)",
    "mean_score_prediction": 5.121067352592945
  },
  {
    "image_id": "TheNarrows(3)",
    "mean_score_prediction": 5.177728123962879
  },
  {
    "image_id": "UpheavalDome",
    "mean_score_prediction": 5.3522849678993225
  },
  {
    "image_id": "BloomingGorse(2)",
    "mean_score_prediction": 5.6223359033465385
  },
  {
    "image_id": "CadesCove",
    "mean_score_prediction": 5.44007234275341
  },
  {
    "image_id": "Peppermill",
    "mean_score_prediction": 4.90664604306221
  },
  {
    "image_id": "TheNarrows(2)",
    "mean_score_prediction": 4.512447014451027
  },
  {
    "image_id": "CanadianFalls",
    "mean_score_prediction": 4.9286525547504425
  },
  {
    "image_id": "MasonLake(1)",
    "mean_score_prediction": 5.113146610558033
  },
  {
    "image_id": "Exploratorium(1)",
    "mean_score_prediction": 5.305305823683739
  },
  {
    "image_id": "GoldenGate(2)",
    "mean_score_prediction": 5.802625373005867
  },
  {
    "image_id": "OCanadaNoLights",
    "mean_score_prediction": 5.551372230052948
  },
  {
    "image_id": "DevilsTower",
    "mean_score_prediction": 4.956215851008892
  },
  {
    "image_id": "HalfDomeSunset",
    "mean_score_prediction": 5.010403156280518
  },
  {
    "image_id": "TunnelView(1)",
    "mean_score_prediction": 4.854714468121529
  },
  {
    "image_id": "SouthBranchKingsRiver",
    "mean_score_prediction": 4.832884684205055
  },
  {
    "image_id": "MtRushmore(1)",
    "mean_score_prediction": 4.796689659357071
  },
  {
    "image_id": "SequoiaRemains",
    "mean_score_prediction": 5.323712639510632
  },
  {
    "image_id": "DelicateArch",
    "mean_score_prediction": 5.185529913753271
  },
  {
    "image_id": "URChapel(2)",
    "mean_score_prediction": 5.107906565070152
  },
  {
    "image_id": "AhwahneeGreatLounge",
    "mean_score_prediction": 5.233778491616249
  },
  {
    "image_id": "BandonSunset(1)",
    "mean_score_prediction": 4.709772378206253
  },
  {
    "image_id": "HancockKitchenOutside",
    "mean_score_prediction": 5.960616886615753
  },
  {
    "image_id": "HooverDam",
    "mean_score_prediction": 4.736658222973347
  },
  {
    "image_id": "WaffleHouse",
    "mean_score_prediction": 5.123516321182251
  },
  {
    "image_id": "GeneralGrant",
    "mean_score_prediction": 5.142636224627495
  },
  {
    "image_id": "HancockKitchenInside",
    "mean_score_prediction": 5.625474497675896
  },
  {
    "image_id": "WestBranchAusable(2)",
    "mean_score_prediction": 4.889512933790684
  },
  {
    "image_id": "BalancedRock",
    "mean_score_prediction": 4.503247067332268
  },
  {
    "image_id": "DevilsGolfCourse",
    "mean_score_prediction": 4.698725588619709
  },
  {
    "image_id": "Zentrum",
    "mean_score_prediction": 4.425833705812693
  },
  {
    "image_id": "NiagaraFalls",
    "mean_score_prediction": 4.779817886650562
  },
  {
    "image_id": "ArtistPalette",
    "mean_score_prediction": 5.330704018473625
  },
  {
    "image_id": "TupperLake(1)",
    "mean_score_prediction": 5.515252567827702
  },
  {
    "image_id": "Petroglyphs",
    "mean_score_prediction": 4.966894768178463
  },
  {
    "image_id": "CemeteryTree(2)",
    "mean_score_prediction": 5.180149391293526
  },
  {
    "image_id": "BarHarborSunrise",
    "mean_score_prediction": 4.949735336005688
  },
  {
    "image_id": "LasVegasStore",
    "mean_score_prediction": 4.998867005109787
  },
  {
    "image_id": "FourCornersStorm",
    "mean_score_prediction": 5.435455679893494
  },
  {
    "image_id": "SunsetPoint(1)",
    "mean_score_prediction": 5.065223775804043
  },
  {
    "image_id": "SmokyTunnel",
    "mean_score_prediction": 4.950747340917587
  },
  {
    "image_id": "LuxoDoubleChecker",
    "mean_score_prediction": 4.963363863527775
  },
  {
    "image_id": "YosemiteFalls",
    "mean_score_prediction": 4.7307218760252
  }
]
    output5 = [
  {
    "image_id": "WallDrug",
    "mean_score_prediction": 5.014258876442909
  },
  {
    "image_id": "SunsetPoint(2)",
    "mean_score_prediction": 5.683080185204744
  },
  {
    "image_id": "TaughannockFalls",
    "mean_score_prediction": 4.312147043645382
  },
  {
    "image_id": "KingsCanyon",
    "mean_score_prediction": 4.817857533693314
  },
  {
    "image_id": "WillySentinel",
    "mean_score_prediction": 5.6590058170259
  },
  {
    "image_id": "GeneralSherman",
    "mean_score_prediction": 5.379643693566322
  },
  {
    "image_id": "TupperLake(2)",
    "mean_score_prediction": 5.093229487538338
  },
  {
    "image_id": "OCanadaLights",
    "mean_score_prediction": 5.623598709702492
  },
  {
    "image_id": "LabWindow",
    "mean_score_prediction": 5.453406911343336
  },
  {
    "image_id": "HallofFame",
    "mean_score_prediction": 5.880660895258188
  },
  {
    "image_id": "WillyDesk",
    "mean_score_prediction": 5.895721158012748
  },
  {
    "image_id": "CemeteryTree(1)",
    "mean_score_prediction": 4.730590485036373
  },
  {
    "image_id": "OtterPoint",
    "mean_score_prediction": 5.221270844340324
  },
  {
    "image_id": "BarHarborPresunrise",
    "mean_score_prediction": 4.4407744333148
  },
  {
    "image_id": "Route66Museum",
    "mean_score_prediction": 5.840929467231035
  },
  {
    "image_id": "WestBranchAusable(1)",
    "mean_score_prediction": 4.966234251856804
  },
  {
    "image_id": "RoundBarnInside",
    "mean_score_prediction": 5.085738852620125
  },
  {
    "image_id": "URChapel(1)",
    "mean_score_prediction": 5.026210993528366
  },
  {
    "image_id": "DevilsBathtub",
    "mean_score_prediction": 5.196614716202021
  },
  {
    "image_id": "RoundStoneBarn",
    "mean_score_prediction": 5.276417173445225
  },
  {
    "image_id": "HooverGarage",
    "mean_score_prediction": 5.162462256848812
  },
  {
    "image_id": "BandonSunset(2)",
    "mean_score_prediction": 4.645876049995422
  },
  {
    "image_id": "MiddlePond",
    "mean_score_prediction": 4.858547061681747
  },
  {
    "image_id": "MasonLake(2)",
    "mean_score_prediction": 5.161871246993542
  },
  {
    "image_id": "Flamingo",
    "mean_score_prediction": 5.0882295072078705
  },
  {
    "image_id": "GoldenGate(1)",
    "mean_score_prediction": 5.268993832170963
  },
  {
    "image_id": "Exploratorium(2)",
    "mean_score_prediction": 5.771660264581442
  },
  {
    "image_id": "PeckLake",
    "mean_score_prediction": 5.749312445521355
  },
  {
    "image_id": "McKeesPub",
    "mean_score_prediction": 5.62398287281394
  },
  {
    "image_id": "MammothHotSprings",
    "mean_score_prediction": 4.769835889339447
  },
  {
    "image_id": "TunnelView(2)",
    "mean_score_prediction": 4.783671833574772
  },
  {
    "image_id": "LabBooth",
    "mean_score_prediction": 5.290387872606516
  },
  {
    "image_id": "PaulBunyan",
    "mean_score_prediction": 5.2055539935827255
  },
  {
    "image_id": "MtRushmore(2)",
    "mean_score_prediction": 5.033203437924385
  },
  {
    "image_id": "BloomingGorse(1)",
    "mean_score_prediction": 5.4802712351083755
  },
  {
    "image_id": "BenJerrys",
    "mean_score_prediction": 5.637765947729349
  },
  {
    "image_id": "ElCapitan",
    "mean_score_prediction": 4.910327486693859
  },
  {
    "image_id": "BigfootPass",
    "mean_score_prediction": 4.65449120849371
  },
  {
    "image_id": "MirrorLake",
    "mean_score_prediction": 4.606832057237625
  },
  {
    "image_id": "AirBellowsGap",
    "mean_score_prediction": 5.099127523601055
  },
  {
    "image_id": "LittleRiver",
    "mean_score_prediction": 4.934138365089893
  },
  {
    "image_id": "507",
    "mean_score_prediction": 5.421639088541269
  },
  {
    "image_id": "HancockSeedField",
    "mean_score_prediction": 5.271799169480801
  },
  {
    "image_id": "TheNarrows(1)",
    "mean_score_prediction": 5.1562559232115746
  },
  {
    "image_id": "TheGrotto",
    "mean_score_prediction": 5.033065646886826
  },
  {
    "image_id": "LetchworthTeaTable(1)",
    "mean_score_prediction": 5.093294970691204
  },
  {
    "image_id": "RoadsEndFireDamage",
    "mean_score_prediction": 4.864856347441673
  },
  {
    "image_id": "AmikeusBeaverDamPM2",
    "mean_score_prediction": 4.656265050172806
  },
  {
    "image_id": "RedwoodSunset",
    "mean_score_prediction": 4.835363060235977
  },
  {
    "image_id": "Frontier",
    "mean_score_prediction": 5.020638056099415
  },
  {
    "image_id": "DelicateFlowers",
    "mean_score_prediction": 4.998823300004005
  },
  {
    "image_id": "LadyBirdRedwoods",
    "mean_score_prediction": 5.094205483794212
  },
  {
    "image_id": "MackinacBridge",
    "mean_score_prediction": 5.799619093537331
  },
  {
    "image_id": "M3MiddlePond",
    "mean_score_prediction": 5.479747027158737
  },
  {
    "image_id": "NorthBubble",
    "mean_score_prediction": 5.185373388230801
  },
  {
    "image_id": "AmikeusBeaverDamPM1",
    "mean_score_prediction": 4.530686683952808
  },
  {
    "image_id": "MtRushmoreFlags",
    "mean_score_prediction": 4.4743741154670715
  },
  {
    "image_id": "OldFaithfulInn",
    "mean_score_prediction": 5.470676049590111
  },
  {
    "image_id": "LabTypewriter",
    "mean_score_prediction": 5.018337033689022
  },
  {
    "image_id": "HDRMark",
    "mean_score_prediction": 5.389440208673477
  },
  {
    "image_id": "RITTiger",
    "mean_score_prediction": 4.934439081698656
  },
  {
    "image_id": "LetchworthTeaTable(2)",
    "mean_score_prediction": 4.9513247311115265
  },
  {
    "image_id": "TheNarrows(3)",
    "mean_score_prediction": 5.2309072986245155
  },
  {
    "image_id": "UpheavalDome",
    "mean_score_prediction": 5.361609309911728
  },
  {
    "image_id": "BloomingGorse(2)",
    "mean_score_prediction": 5.522972993552685
  },
  {
    "image_id": "CadesCove",
    "mean_score_prediction": 5.4721115455031395
  },
  {
    "image_id": "Peppermill",
    "mean_score_prediction": 4.819788537919521
  },
  {
    "image_id": "TheNarrows(2)",
    "mean_score_prediction": 4.399512272328138
  },
  {
    "image_id": "CanadianFalls",
    "mean_score_prediction": 4.789958402514458
  },
  {
    "image_id": "MasonLake(1)",
    "mean_score_prediction": 5.110246129333973
  },
  {
    "image_id": "Exploratorium(1)",
    "mean_score_prediction": 5.225281894207001
  },
  {
    "image_id": "GoldenGate(2)",
    "mean_score_prediction": 5.740479584783316
  },
  {
    "image_id": "OCanadaNoLights",
    "mean_score_prediction": 5.503159694373608
  },
  {
    "image_id": "DevilsTower",
    "mean_score_prediction": 4.9707236886024475
  },
  {
    "image_id": "HalfDomeSunset",
    "mean_score_prediction": 5.043085657060146
  },
  {
    "image_id": "TunnelView(1)",
    "mean_score_prediction": 4.848295636475086
  },
  {
    "image_id": "SouthBranchKingsRiver",
    "mean_score_prediction": 4.7904177233576775
  },
  {
    "image_id": "MtRushmore(1)",
    "mean_score_prediction": 4.7572826743125916
  },
  {
    "image_id": "SequoiaRemains",
    "mean_score_prediction": 5.275208741426468
  },
  {
    "image_id": "DelicateArch",
    "mean_score_prediction": 5.2287504486739635
  },
  {
    "image_id": "URChapel(2)",
    "mean_score_prediction": 5.097635246813297
  },
  {
    "image_id": "AhwahneeGreatLounge",
    "mean_score_prediction": 5.170941643416882
  },
  {
    "image_id": "BandonSunset(1)",
    "mean_score_prediction": 4.865711040794849
  },
  {
    "image_id": "HancockKitchenOutside",
    "mean_score_prediction": 5.901556737720966
  },
  {
    "image_id": "HooverDam",
    "mean_score_prediction": 4.7194922640919685
  },
  {
    "image_id": "WaffleHouse",
    "mean_score_prediction": 5.2560978047549725
  },
  {
    "image_id": "GeneralGrant",
    "mean_score_prediction": 4.913031585514545
  },
  {
    "image_id": "HancockKitchenInside",
    "mean_score_prediction": 5.597076207399368
  },
  {
    "image_id": "WestBranchAusable(2)",
    "mean_score_prediction": 4.931431248784065
  },
  {
    "image_id": "BalancedRock",
    "mean_score_prediction": 4.49051009118557
  },
  {
    "image_id": "DevilsGolfCourse",
    "mean_score_prediction": 4.710384447127581
  },
  {
    "image_id": "Zentrum",
    "mean_score_prediction": 4.780646935105324
  },
  {
    "image_id": "NiagaraFalls",
    "mean_score_prediction": 4.68716012686491
  },
  {
    "image_id": "ArtistPalette",
    "mean_score_prediction": 5.359580162912607
  },
  {
    "image_id": "TupperLake(1)",
    "mean_score_prediction": 5.641960568726063
  },
  {
    "image_id": "Petroglyphs",
    "mean_score_prediction": 5.016902454197407
  },
  {
    "image_id": "CemeteryTree(2)",
    "mean_score_prediction": 5.267533294856548
  },
  {
    "image_id": "BarHarborSunrise",
    "mean_score_prediction": 4.931722350418568
  },
  {
    "image_id": "LasVegasStore",
    "mean_score_prediction": 4.828512065112591
  },
  {
    "image_id": "FourCornersStorm",
    "mean_score_prediction": 5.244228757917881
  },
  {
    "image_id": "SunsetPoint(1)",
    "mean_score_prediction": 4.994406595826149
  },
  {
    "image_id": "SmokyTunnel",
    "mean_score_prediction": 4.811430744826794
  },
  {
    "image_id": "LuxoDoubleChecker",
    "mean_score_prediction": 4.8020031079649925
  },
  {
    "image_id": "YosemiteFalls",
    "mean_score_prediction": 4.750041723251343
  }
]
    output6 = [
  {
    "image_id": "WallDrug",
    "mean_score_prediction": 5.008164897561073
  },
  {
    "image_id": "SunsetPoint(2)",
    "mean_score_prediction": 5.692657262086868
  },
  {
    "image_id": "TaughannockFalls",
    "mean_score_prediction": 4.362137142568827
  },
  {
    "image_id": "KingsCanyon",
    "mean_score_prediction": 4.885212458670139
  },
  {
    "image_id": "WillySentinel",
    "mean_score_prediction": 5.465009726583958
  },
  {
    "image_id": "GeneralSherman",
    "mean_score_prediction": 5.381689190864563
  },
  {
    "image_id": "TupperLake(2)",
    "mean_score_prediction": 5.110593155026436
  },
  {
    "image_id": "OCanadaLights",
    "mean_score_prediction": 5.638568181544542
  },
  {
    "image_id": "LabWindow",
    "mean_score_prediction": 5.4815590642392635
  },
  {
    "image_id": "HallofFame",
    "mean_score_prediction": 5.936848189681768
  },
  {
    "image_id": "WillyDesk",
    "mean_score_prediction": 5.925436310470104
  },
  {
    "image_id": "CemeteryTree(1)",
    "mean_score_prediction": 4.736635901033878
  },
  {
    "image_id": "OtterPoint",
    "mean_score_prediction": 5.277453765273094
  },
  {
    "image_id": "BarHarborPresunrise",
    "mean_score_prediction": 4.468562498688698
  },
  {
    "image_id": "Route66Museum",
    "mean_score_prediction": 5.890871290117502
  },
  {
    "image_id": "WestBranchAusable(1)",
    "mean_score_prediction": 4.973749071359634
  },
  {
    "image_id": "RoundBarnInside",
    "mean_score_prediction": 5.15381284058094
  },
  {
    "image_id": "URChapel(1)",
    "mean_score_prediction": 4.792975544929504
  },
  {
    "image_id": "DevilsBathtub",
    "mean_score_prediction": 5.160772129893303
  },
  {
    "image_id": "RoundStoneBarn",
    "mean_score_prediction": 5.2688256576657295
  },
  {
    "image_id": "HooverGarage",
    "mean_score_prediction": 4.9528399631381035
  },
  {
    "image_id": "BandonSunset(2)",
    "mean_score_prediction": 4.627432122826576
  },
  {
    "image_id": "MiddlePond",
    "mean_score_prediction": 5.010046668350697
  },
  {
    "image_id": "MasonLake(2)",
    "mean_score_prediction": 5.1424926444888115
  },
  {
    "image_id": "Flamingo",
    "mean_score_prediction": 5.079835340380669
  },
  {
    "image_id": "GoldenGate(1)",
    "mean_score_prediction": 5.232754182070494
  },
  {
    "image_id": "Exploratorium(2)",
    "mean_score_prediction": 5.791824653744698
  },
  {
    "image_id": "PeckLake",
    "mean_score_prediction": 5.7124225571751595
  },
  {
    "image_id": "McKeesPub",
    "mean_score_prediction": 5.673691879957914
  },
  {
    "image_id": "MammothHotSprings",
    "mean_score_prediction": 4.775609664618969
  },
  {
    "image_id": "TunnelView(2)",
    "mean_score_prediction": 4.83048690110445
  },
  {
    "image_id": "LabBooth",
    "mean_score_prediction": 5.273452773690224
  },
  {
    "image_id": "PaulBunyan",
    "mean_score_prediction": 5.192277401685715
  },
  {
    "image_id": "MtRushmore(2)",
    "mean_score_prediction": 4.9494554325938225
  },
  {
    "image_id": "BloomingGorse(1)",
    "mean_score_prediction": 5.454587072134018
  },
  {
    "image_id": "BenJerrys",
    "mean_score_prediction": 5.646572235971689
  },
  {
    "image_id": "ElCapitan",
    "mean_score_prediction": 4.856618918478489
  },
  {
    "image_id": "BigfootPass",
    "mean_score_prediction": 4.6153647899627686
  },
  {
    "image_id": "MirrorLake",
    "mean_score_prediction": 4.613171085715294
  },
  {
    "image_id": "AirBellowsGap",
    "mean_score_prediction": 5.157235562801361
  },
  {
    "image_id": "LittleRiver",
    "mean_score_prediction": 4.867659367620945
  },
  {
    "image_id": "507",
    "mean_score_prediction": 5.391859006136656
  },
  {
    "image_id": "HancockSeedField",
    "mean_score_prediction": 5.3175932094454765
  },
  {
    "image_id": "TheNarrows(1)",
    "mean_score_prediction": 5.1643103659152985
  },
  {
    "image_id": "TheGrotto",
    "mean_score_prediction": 5.019156776368618
  },
  {
    "image_id": "LetchworthTeaTable(1)",
    "mean_score_prediction": 5.112160041928291
  },
  {
    "image_id": "RoadsEndFireDamage",
    "mean_score_prediction": 4.831302233040333
  },
  {
    "image_id": "AmikeusBeaverDamPM2",
    "mean_score_prediction": 4.650308392941952
  },
  {
    "image_id": "RedwoodSunset",
    "mean_score_prediction": 4.8802884221076965
  },
  {
    "image_id": "Frontier",
    "mean_score_prediction": 5.009547747671604
  },
  {
    "image_id": "DelicateFlowers",
    "mean_score_prediction": 5.0656632371246815
  },
  {
    "image_id": "LadyBirdRedwoods",
    "mean_score_prediction": 5.140780344605446
  },
  {
    "image_id": "MackinacBridge",
    "mean_score_prediction": 5.819883905351162
  },
  {
    "image_id": "M3MiddlePond",
    "mean_score_prediction": 5.466603171080351
  },
  {
    "image_id": "NorthBubble",
    "mean_score_prediction": 5.224505215883255
  },
  {
    "image_id": "AmikeusBeaverDamPM1",
    "mean_score_prediction": 4.462208494544029
  },
  {
    "image_id": "MtRushmoreFlags",
    "mean_score_prediction": 4.4848960265517235
  },
  {
    "image_id": "OldFaithfulInn",
    "mean_score_prediction": 5.495024975389242
  },
  {
    "image_id": "LabTypewriter",
    "mean_score_prediction": 5.001124478876591
  },
  {
    "image_id": "HDRMark",
    "mean_score_prediction": 5.307026360183954
  },
  {
    "image_id": "RITTiger",
    "mean_score_prediction": 4.92340911924839
  },
  {
    "image_id": "LetchworthTeaTable(2)",
    "mean_score_prediction": 4.873766556382179
  },
  {
    "image_id": "TheNarrows(3)",
    "mean_score_prediction": 5.292634844779968
  },
  {
    "image_id": "UpheavalDome",
    "mean_score_prediction": 5.363691128790379
  },
  {
    "image_id": "BloomingGorse(2)",
    "mean_score_prediction": 5.687753461301327
  },
  {
    "image_id": "CadesCove",
    "mean_score_prediction": 5.42987797409296
  },
  {
    "image_id": "Peppermill",
    "mean_score_prediction": 4.799575582146645
  },
  {
    "image_id": "TheNarrows(2)",
    "mean_score_prediction": 4.440989650785923
  },
  {
    "image_id": "CanadianFalls",
    "mean_score_prediction": 4.802495278418064
  },
  {
    "image_id": "MasonLake(1)",
    "mean_score_prediction": 5.180996045470238
  },
  {
    "image_id": "Exploratorium(1)",
    "mean_score_prediction": 5.188962921500206
  },
  {
    "image_id": "GoldenGate(2)",
    "mean_score_prediction": 5.781427256762981
  },
  {
    "image_id": "OCanadaNoLights",
    "mean_score_prediction": 5.4469200521707535
  },
  {
    "image_id": "DevilsTower",
    "mean_score_prediction": 4.876483507454395
  },
  {
    "image_id": "HalfDomeSunset",
    "mean_score_prediction": 5.123026125133038
  },
  {
    "image_id": "TunnelView(1)",
    "mean_score_prediction": 4.912133693695068
  },
  {
    "image_id": "SouthBranchKingsRiver",
    "mean_score_prediction": 4.7660166919231415
  },
  {
    "image_id": "MtRushmore(1)",
    "mean_score_prediction": 4.728158250451088
  },
  {
    "image_id": "SequoiaRemains",
    "mean_score_prediction": 5.301289148628712
  },
  {
    "image_id": "DelicateArch",
    "mean_score_prediction": 5.200526997447014
  },
  {
    "image_id": "URChapel(2)",
    "mean_score_prediction": 5.052488148212433
  },
  {
    "image_id": "AhwahneeGreatLounge",
    "mean_score_prediction": 5.177405491471291
  },
  {
    "image_id": "BandonSunset(1)",
    "mean_score_prediction": 4.8305580243468285
  },
  {
    "image_id": "HancockKitchenOutside",
    "mean_score_prediction": 5.946976564824581
  },
  {
    "image_id": "HooverDam",
    "mean_score_prediction": 4.712201163172722
  },
  {
    "image_id": "WaffleHouse",
    "mean_score_prediction": 5.2635157108306885
  },
  {
    "image_id": "GeneralGrant",
    "mean_score_prediction": 4.995568826794624
  },
  {
    "image_id": "HancockKitchenInside",
    "mean_score_prediction": 5.639124393463135
  },
  {
    "image_id": "WestBranchAusable(2)",
    "mean_score_prediction": 4.946052275598049
  },
  {
    "image_id": "BalancedRock",
    "mean_score_prediction": 4.463675022125244
  },
  {
    "image_id": "DevilsGolfCourse",
    "mean_score_prediction": 4.709087286144495
  },
  {
    "image_id": "Zentrum",
    "mean_score_prediction": 4.643988512456417
  },
  {
    "image_id": "NiagaraFalls",
    "mean_score_prediction": 4.761895872652531
  },
  {
    "image_id": "ArtistPalette",
    "mean_score_prediction": 5.3429164327681065
  },
  {
    "image_id": "TupperLake(1)",
    "mean_score_prediction": 5.52356793731451
  },
  {
    "image_id": "Petroglyphs",
    "mean_score_prediction": 5.021991543471813
  },
  {
    "image_id": "CemeteryTree(2)",
    "mean_score_prediction": 5.23383566737175
  },
  {
    "image_id": "BarHarborSunrise",
    "mean_score_prediction": 5.001877948641777
  },
  {
    "image_id": "LasVegasStore",
    "mean_score_prediction": 4.843820542097092
  },
  {
    "image_id": "FourCornersStorm",
    "mean_score_prediction": 5.337743677198887
  },
  {
    "image_id": "SunsetPoint(1)",
    "mean_score_prediction": 4.95403864979744
  },
  {
    "image_id": "SmokyTunnel",
    "mean_score_prediction": 4.879784733057022
  },
  {
    "image_id": "LuxoDoubleChecker",
    "mean_score_prediction": 4.801334723830223
  },
  {
    "image_id": "YosemiteFalls",
    "mean_score_prediction": 4.743366375565529
  }
]
    output7 = [
  {
    "image_id": "WallDrug",
    "mean_score_prediction": 4.672092750668526
  },
  {
    "image_id": "SunsetPoint(2)",
    "mean_score_prediction": 5.309578083455563
  },
  {
    "image_id": "TaughannockFalls",
    "mean_score_prediction": 4.586088724434376
  },
  {
    "image_id": "KingsCanyon",
    "mean_score_prediction": 4.9822070226073265
  },
  {
    "image_id": "WillySentinel",
    "mean_score_prediction": 5.5233013443648815
  },
  {
    "image_id": "GeneralSherman",
    "mean_score_prediction": 5.3465805649757385
  },
  {
    "image_id": "TupperLake(2)",
    "mean_score_prediction": 5.353288903832436
  },
  {
    "image_id": "OCanadaLights",
    "mean_score_prediction": 5.621331177651882
  },
  {
    "image_id": "LabWindow",
    "mean_score_prediction": 5.304145351052284
  },
  {
    "image_id": "HallofFame",
    "mean_score_prediction": 5.904010958969593
  },
  {
    "image_id": "WillyDesk",
    "mean_score_prediction": 5.900748761370778
  },
  {
    "image_id": "CemeteryTree(1)",
    "mean_score_prediction": 4.653109036386013
  },
  {
    "image_id": "OtterPoint",
    "mean_score_prediction": 5.290730245411396
  },
  {
    "image_id": "BarHarborPresunrise",
    "mean_score_prediction": 5.113611817359924
  },
  {
    "image_id": "Route66Museum",
    "mean_score_prediction": 5.543144538998604
  },
  {
    "image_id": "WestBranchAusable(1)",
    "mean_score_prediction": 4.705025292932987
  },
  {
    "image_id": "RoundBarnInside",
    "mean_score_prediction": 5.089430138468742
  },
  {
    "image_id": "URChapel(1)",
    "mean_score_prediction": 4.7069979049265385
  },
  {
    "image_id": "DevilsBathtub",
    "mean_score_prediction": 5.4516734555363655
  },
  {
    "image_id": "RoundStoneBarn",
    "mean_score_prediction": 5.509722128510475
  },
  {
    "image_id": "HooverGarage",
    "mean_score_prediction": 4.853814251720905
  },
  {
    "image_id": "BandonSunset(2)",
    "mean_score_prediction": 6.068212889134884
  },
  {
    "image_id": "MiddlePond",
    "mean_score_prediction": 4.669018484652042
  },
  {
    "image_id": "MasonLake(2)",
    "mean_score_prediction": 5.040874682366848
  },
  {
    "image_id": "Flamingo",
    "mean_score_prediction": 4.778527811169624
  },
  {
    "image_id": "GoldenGate(1)",
    "mean_score_prediction": 5.638784982264042
  },
  {
    "image_id": "Exploratorium(2)",
    "mean_score_prediction": 5.661952946335077
  },
  {
    "image_id": "PeckLake",
    "mean_score_prediction": 5.5300192311406136
  },
  {
    "image_id": "McKeesPub",
    "mean_score_prediction": 5.681986026465893
  },
  {
    "image_id": "MammothHotSprings",
    "mean_score_prediction": 4.836504094302654
  },
  {
    "image_id": "TunnelView(2)",
    "mean_score_prediction": 5.086847543716431
  },
  {
    "image_id": "LabBooth",
    "mean_score_prediction": 5.736645393073559
  },
  {
    "image_id": "PaulBunyan",
    "mean_score_prediction": 5.450222574174404
  },
  {
    "image_id": "MtRushmore(2)",
    "mean_score_prediction": 5.14121687412262
  },
  {
    "image_id": "BloomingGorse(1)",
    "mean_score_prediction": 5.306029103696346
  },
  {
    "image_id": "BenJerrys",
    "mean_score_prediction": 5.5283116810023785
  },
  {
    "image_id": "ElCapitan",
    "mean_score_prediction": 4.94685722887516
  },
  {
    "image_id": "BigfootPass",
    "mean_score_prediction": 4.356139309704304
  },
  {
    "image_id": "MirrorLake",
    "mean_score_prediction": 4.521955206990242
  },
  {
    "image_id": "AirBellowsGap",
    "mean_score_prediction": 5.287514463067055
  },
  {
    "image_id": "LittleRiver",
    "mean_score_prediction": 4.915122359991074
  },
  {
    "image_id": "507",
    "mean_score_prediction": 5.18158283084631
  },
  {
    "image_id": "HancockSeedField",
    "mean_score_prediction": 4.862524151802063
  },
  {
    "image_id": "TheNarrows(1)",
    "mean_score_prediction": 5.345277287065983
  },
  {
    "image_id": "TheGrotto",
    "mean_score_prediction": 4.877807088196278
  },
  {
    "image_id": "LetchworthTeaTable(1)",
    "mean_score_prediction": 5.026304304599762
  },
  {
    "image_id": "RoadsEndFireDamage",
    "mean_score_prediction": 4.913227774202824
  },
  {
    "image_id": "AmikeusBeaverDamPM2",
    "mean_score_prediction": 4.75306348502636
  },
  {
    "image_id": "RedwoodSunset",
    "mean_score_prediction": 4.789991103112698
  },
  {
    "image_id": "Frontier",
    "mean_score_prediction": 4.816361963748932
  },
  {
    "image_id": "DelicateFlowers",
    "mean_score_prediction": 4.587897710502148
  },
  {
    "image_id": "LadyBirdRedwoods",
    "mean_score_prediction": 4.889233335852623
  },
  {
    "image_id": "MackinacBridge",
    "mean_score_prediction": 5.514069490134716
  },
  {
    "image_id": "M3MiddlePond",
    "mean_score_prediction": 5.4619577415287495
  },
  {
    "image_id": "NorthBubble",
    "mean_score_prediction": 4.7350920513272285
  },
  {
    "image_id": "AmikeusBeaverDamPM1",
    "mean_score_prediction": 4.713416449725628
  },
  {
    "image_id": "MtRushmoreFlags",
    "mean_score_prediction": 5.0077421218156815
  },
  {
    "image_id": "OldFaithfulInn",
    "mean_score_prediction": 5.172671802341938
  },
  {
    "image_id": "LabTypewriter",
    "mean_score_prediction": 5.10778620839119
  },
  {
    "image_id": "HDRMark",
    "mean_score_prediction": 5.614555511623621
  },
  {
    "image_id": "RITTiger",
    "mean_score_prediction": 4.670745529234409
  },
  {
    "image_id": "LetchworthTeaTable(2)",
    "mean_score_prediction": 5.3374714851379395
  },
  {
    "image_id": "TheNarrows(3)",
    "mean_score_prediction": 4.950577892363071
  },
  {
    "image_id": "JesseBrownsCabin",
    "mean_score_prediction": 4.851165689527988
  },
  {
    "image_id": "UpheavalDome",
    "mean_score_prediction": 4.821387082338333
  },
  {
    "image_id": "BloomingGorse(2)",
    "mean_score_prediction": 5.290609806776047
  },
  {
    "image_id": "CadesCove",
    "mean_score_prediction": 5.288948751986027
  },
  {
    "image_id": "Peppermill",
    "mean_score_prediction": 5.080713793635368
  },
  {
    "image_id": "TheNarrows(2)",
    "mean_score_prediction": 4.1137619242072105
  },
  {
    "image_id": "CanadianFalls",
    "mean_score_prediction": 4.922528199851513
  },
  {
    "image_id": "MasonLake(1)",
    "mean_score_prediction": 4.990933142602444
  },
  {
    "image_id": "Exploratorium(1)",
    "mean_score_prediction": 5.118807680904865
  },
  {
    "image_id": "GoldenGate(2)",
    "mean_score_prediction": 5.053360939025879
  },
  {
    "image_id": "OCanadaNoLights",
    "mean_score_prediction": 5.551447458565235
  },
  {
    "image_id": "DevilsTower",
    "mean_score_prediction": 4.683366194367409
  },
  {
    "image_id": "HalfDomeSunset",
    "mean_score_prediction": 5.1112509816884995
  },
  {
    "image_id": "TunnelView(1)",
    "mean_score_prediction": 4.799077771604061
  },
  {
    "image_id": "SouthBranchKingsRiver",
    "mean_score_prediction": 5.223235733807087
  },
  {
    "image_id": "MtRushmore(1)",
    "mean_score_prediction": 4.745273880660534
  },
  {
    "image_id": "SequoiaRemains",
    "mean_score_prediction": 4.704207696020603
  },
  {
    "image_id": "DelicateArch",
    "mean_score_prediction": 5.030386805534363
  },
  {
    "image_id": "URChapel(2)",
    "mean_score_prediction": 5.317658979445696
  },
  {
    "image_id": "AhwahneeGreatLounge",
    "mean_score_prediction": 5.243354961276054
  },
  {
    "image_id": "BandonSunset(1)",
    "mean_score_prediction": 6.255337227135897
  },
  {
    "image_id": "HancockKitchenOutside",
    "mean_score_prediction": 5.457872122526169
  },
  {
    "image_id": "HooverDam",
    "mean_score_prediction": 5.027331531047821
  },
  {
    "image_id": "WaffleHouse",
    "mean_score_prediction": 4.894663132727146
  },
  {
    "image_id": "GeneralGrant",
    "mean_score_prediction": 4.941694676876068
  },
  {
    "image_id": "HancockKitchenInside",
    "mean_score_prediction": 5.356546096503735
  },
  {
    "image_id": "WestBranchAusable(2)",
    "mean_score_prediction": 4.638536795973778
  },
  {
    "image_id": "BalancedRock",
    "mean_score_prediction": 4.8182061687111855
  },
  {
    "image_id": "DevilsGolfCourse",
    "mean_score_prediction": 3.92198933288455
  },
  {
    "image_id": "Zentrum",
    "mean_score_prediction": 4.493800491094589
  },
  {
    "image_id": "NiagaraFalls",
    "mean_score_prediction": 4.899976760149002
  },
  {
    "image_id": "ArtistPalette",
    "mean_score_prediction": 4.945939600467682
  },
  {
    "image_id": "TupperLake(1)",
    "mean_score_prediction": 5.213843576610088
  },
  {
    "image_id": "Petroglyphs",
    "mean_score_prediction": 4.997818753123283
  },
  {
    "image_id": "CemeteryTree(2)",
    "mean_score_prediction": 4.904869236052036
  },
  {
    "image_id": "BarHarborSunrise",
    "mean_score_prediction": 5.312677964568138
  },
  {
    "image_id": "LasVegasStore",
    "mean_score_prediction": 4.569056116044521
  },
  {
    "image_id": "FourCornersStorm",
    "mean_score_prediction": 5.659073442220688
  },
  {
    "image_id": "SunsetPoint(1)",
    "mean_score_prediction": 4.879854537546635
  },
  {
    "image_id": "SmokyTunnel",
    "mean_score_prediction": 4.3965794295072556
  },
  {
    "image_id": "LuxoDoubleChecker",
    "mean_score_prediction": 5.1014242470264435
  },
  {
    "image_id": "YosemiteFalls",
    "mean_score_prediction": 5.0077463909983635
  }
]
    output8 = [
  {
    "image_id": "371897",
    "mean_score_prediction": 5.598567020148039
  },
  {
    "image_id": "675153",
    "mean_score_prediction": 5.929004788398743
  },
  {
    "image_id": "1440465",
    "mean_score_prediction": 5.238162435591221
  },
  {
    "image_id": "3494059",
    "mean_score_prediction": 6.331102388910949
  },
  {
    "image_id": "764507",
    "mean_score_prediction": 6.354872648604214
  },
  {
    "image_id": "3637013",
    "mean_score_prediction": 5.2959882244467735
  },
  {
    "image_id": "2209751",
    "mean_score_prediction": 5.3979418613016605
  },
  {
    "image_id": "574181",
    "mean_score_prediction": 5.891698848456144
  },
  {
    "image_id": "3996401",
    "mean_score_prediction": 6.134052574634552
  },
  {
    "image_id": "3537322",
    "mean_score_prediction": 5.845956303179264
  },
  {
    "image_id": "2148982",
    "mean_score_prediction": 6.246708437800407
  },
  {
    "image_id": "4489731",
    "mean_score_prediction": 5.8122064135968685
  },
  {
    "image_id": "3367399",
    "mean_score_prediction": 5.406239710748196
  },
  {
    "image_id": "2689611",
    "mean_score_prediction": 5.980732273310423
  },
  {
    "image_id": "1254659",
    "mean_score_prediction": 5.484644457697868
  },
  {
    "image_id": "667626",
    "mean_score_prediction": 5.529547430574894
  },
  {
    "image_id": "4376178",
    "mean_score_prediction": 5.582177005708218
  },
  {
    "image_id": "3734864",
    "mean_score_prediction": 5.017235137522221
  },
  {
    "image_id": "793558",
    "mean_score_prediction": 5.788375955075026
  },
  {
    "image_id": "984950",
    "mean_score_prediction": 5.247685257345438
  },
  {
    "image_id": "1283466",
    "mean_score_prediction": 5.566579714417458
  },
  {
    "image_id": "3160699",
    "mean_score_prediction": 5.828963406383991
  },
  {
    "image_id": "4386588",
    "mean_score_prediction": 5.987102188169956
  },
  {
    "image_id": "960092",
    "mean_score_prediction": 5.4583811312913895
  },
  {
    "image_id": "4280272",
    "mean_score_prediction": 5.417329337447882
  },
  {
    "image_id": "3753939",
    "mean_score_prediction": 5.795193333178759
  },
  {
    "image_id": "4162702",
    "mean_score_prediction": 5.199084661900997
  },
  {
    "image_id": "301246",
    "mean_score_prediction": 5.599133588373661
  },
  {
    "image_id": "2760167",
    "mean_score_prediction": 5.438258305191994
  },
  {
    "image_id": "2784746",
    "mean_score_prediction": 6.142256448045373
  },
  {
    "image_id": "881336",
    "mean_score_prediction": 6.023562448099256
  },
  {
    "image_id": "1243756",
    "mean_score_prediction": 6.240819210186601
  },
  {
    "image_id": "438106",
    "mean_score_prediction": 6.274631050415337
  },
  {
    "image_id": "4414061",
    "mean_score_prediction": 6.265908097848296
  },
  {
    "image_id": "490870",
    "mean_score_prediction": 6.229764401912689
  },
  {
    "image_id": "36979",
    "mean_score_prediction": 5.707424566149712
  },
  {
    "image_id": "3787801",
    "mean_score_prediction": 5.060642391443253
  },
  {
    "image_id": "726414",
    "mean_score_prediction": 6.248739153146744
  },
  {
    "image_id": "4413714",
    "mean_score_prediction": 5.789243698120117
  },
  {
    "image_id": "390369",
    "mean_score_prediction": 5.921703156083822
  },
  {
    "image_id": "1624481",
    "mean_score_prediction": 5.495530806481838
  },
  {
    "image_id": "3662865",
    "mean_score_prediction": 5.443364918231964
  },
  {
    "image_id": "2209317",
    "mean_score_prediction": 5.795364513993263
  },
  {
    "image_id": "4429660",
    "mean_score_prediction": 5.707206692546606
  },
  {
    "image_id": "3001353",
    "mean_score_prediction": 6.426283407956362
  },
  {
    "image_id": "2656351",
    "mean_score_prediction": 5.936116527765989
  },
  {
    "image_id": "3043766",
    "mean_score_prediction": 5.834868632256985
  },
  {
    "image_id": "4135695",
    "mean_score_prediction": 6.396415248513222
  },
  {
    "image_id": "2285664",
    "mean_score_prediction": 5.4416244477033615
  },
  {
    "image_id": "178045",
    "mean_score_prediction": 5.717229478061199
  },
  {
    "image_id": "81641",
    "mean_score_prediction": 5.865032263100147
  },
  {
    "image_id": "353913",
    "mean_score_prediction": 6.153038069605827
  },
  {
    "image_id": "2192573",
    "mean_score_prediction": 6.331151589751244
  },
  {
    "image_id": "3219606",
    "mean_score_prediction": 5.463425777852535
  },
  {
    "image_id": "4199555",
    "mean_score_prediction": 6.133179359138012
  },
  {
    "image_id": "256063",
    "mean_score_prediction": 6.027830636128783
  },
  {
    "image_id": "2317271",
    "mean_score_prediction": 5.2569941729307175
  },
  {
    "image_id": "2069887",
    "mean_score_prediction": 4.818056624382734
  },
  {
    "image_id": "4183120",
    "mean_score_prediction": 6.003412574529648
  },
  {
    "image_id": "1369162",
    "mean_score_prediction": 5.855602020397782
  },
  {
    "image_id": "4307968",
    "mean_score_prediction": 5.442325510084629
  },
  {
    "image_id": "371902",
    "mean_score_prediction": 5.704296834766865
  },
  {
    "image_id": "1920465",
    "mean_score_prediction": 5.053186222910881
  },
  {
    "image_id": "807129",
    "mean_score_prediction": 5.877774514257908
  },
  {
    "image_id": "3012229",
    "mean_score_prediction": 5.813926741480827
  },
  {
    "image_id": "3765589",
    "mean_score_prediction": 5.301263123750687
  },
  {
    "image_id": "1317156",
    "mean_score_prediction": 4.959729619324207
  },
  {
    "image_id": "371903",
    "mean_score_prediction": 6.11148077622056
  },
  {
    "image_id": "1989609",
    "mean_score_prediction": 5.902888752520084
  },
  {
    "image_id": "148284",
    "mean_score_prediction": 6.241413693875074
  },
  {
    "image_id": "2868798",
    "mean_score_prediction": 5.473544493317604
  },
  {
    "image_id": "205842",
    "mean_score_prediction": 6.124047838151455
  },
  {
    "image_id": "3680138",
    "mean_score_prediction": 6.101993782445788
  },
  {
    "image_id": "134206",
    "mean_score_prediction": 5.0144045650959015
  },
  {
    "image_id": "3035057",
    "mean_score_prediction": 5.584641702473164
  },
  {
    "image_id": "4378823",
    "mean_score_prediction": 6.5199292451143265
  },
  {
    "image_id": "3025093",
    "mean_score_prediction": 5.653045259416103
  },
  {
    "image_id": "2806447",
    "mean_score_prediction": 5.2700929790735245
  },
  {
    "image_id": "65567",
    "mean_score_prediction": 6.246271478012204
  },
  {
    "image_id": "854749",
    "mean_score_prediction": 5.455195106565952
  }
]


    total = 0
    for i in output:
        total += i['mean_score_prediction']
    print(len(output))
    print(total / len(output))

    total = 0
    for i in output2:
        total += i['mean_score_prediction']
    print(len(output))
    print(total / len(output))

    total = 0
    for i in output3:
        total += i['mean_score_prediction']
    print(len(output))
    print(total / len(output))

    total = 0
    for i in output4:
        total += i['mean_score_prediction']
    print(len(output))
    print(total / len(output))

    total = 0
    for i in output5:
        total += i['mean_score_prediction']
    print(len(output))
    print(total / len(output))

    total = 0
    for i in output6:
        total += i['mean_score_prediction']
    print(len(output))
    print(total / len(output))

    print("durand luminance")
    total = 0
    for i in output7:
        total += i['mean_score_prediction']
    print(len(output))
    print(total / len(output))

    print("ldr")
    total = 0
    for i in output8:
        total += i['mean_score_prediction']
    print(len(output))
    print(total / len(output))

def back_to_color1(im_hdr, fake):
    im_hdr = im_hdr - np.min(im_hdr)
    im_gray_ = np.sum(im_hdr, axis=2)
    # print_image_details(im_gray_,"im_gray_")
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / (im_gray_ + params.epsilon)
    norm_im[:, :, 1] = im_hdr[:, :, 1] / (im_gray_ + params.epsilon)
    norm_im[:, :, 2] = im_hdr[:, :, 2] / (im_gray_ + params.epsilon)
    output_im = np.power(norm_im, 0.5) * fake
    return to_0_1_range(output_im)

def back_to_color2(im_hdr, fake):
    im_hdr = im_hdr - np.min(im_hdr)
    im_gray_ = hdr_image_util.to_gray(im_hdr / np.max(im_hdr))
    # print_image_details(im_gray_,"im_gray_")
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / (im_gray_ + params.epsilon)
    norm_im[:, :, 1] = im_hdr[:, :, 1] / (im_gray_ + params.epsilon)
    norm_im[:, :, 2] = im_hdr[:, :, 2] / (im_gray_ + params.epsilon)
    output_im = np.power(norm_im, 0.5) * fake
    return to_0_1_range(output_im)

def back_to_color3(im_hdr, fake):
    fake = to_0_1_range(fake)
    # im_hdr = np.max(im_hdr) * to_0_1_range(im_hdr)
    if np.min(im_hdr) < 0:
        im_hdr = im_hdr - np.min(im_hdr)
    im_gray_ = to_gray(im_hdr)
    # print_image_details(im_gray_,"im_gray_")
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / (im_gray_ + params.epsilon)
    norm_im[:, :, 1] = im_hdr[:, :, 1] / (im_gray_ + params.epsilon)
    norm_im[:, :, 2] = im_hdr[:, :, 2] / (im_gray_ + params.epsilon)
    output_im = np.power(norm_im, 0.5) * fake
    output_im = output_im / np.linalg.norm(norm_im)
    return to_0_1_range(output_im)


def back_to_color_dir(input_dir, exr_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for im_name in os.listdir(exr_dir):
        im_pref = os.path.splitext(im_name)[0]
        if im_pref != "JesseBrownsCabin":
            cur_im_exr_path = os.path.join(exr_dir, im_name)
            cur_im_input_dir = os.path.join(input_dir, im_pref+ ".png")
            cur_im_output_dir = os.path.join(output_dir, im_pref+ ".jpg")
            if not os.path.exists(cur_im_output_dir):
                exr_im = imageio.imread(cur_im_exr_path, format='EXR-FI')
                gray_im = imageio.imread(cur_im_input_dir)
                gray_im = gray_im[:,:,None]
                res_im = back_to_color3(exr_im, gray_im)
                imageio.imwrite(cur_im_output_dir, res_im)

def to_gray_test():
    x = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data/ldr_data/ldr_data/im_96.bmp")
    hdr_image_util.print_image_details(x, "x")
    z = np.dot(x[..., :3], [0.299, 0.587, 0.114])
    y = np.zeros_like(x)
    x1 = 0.299 * x[:, :, 0] + 0.5870 * x[:, :, 1] + 0.1140 * x[:, :, 2]
    y[:, :, 0] = x1
    y[:, :, 1] = x1
    y[:, :, 2] = x1
    hdr_image_util.print_image_details(x1, "x1")
    hdr_image_util.print_image_details(z, "z")
    plt.subplot(2,1,1)
    plt.imshow(x1, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.imshow(z, cmap='gray')
    plt.show()

def gather_all_architectures(arch_dir, output_path, epoch, date, im_number):
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, epoch)
        old_name = im_number + "_epoch_" + epoch + "_gray.png"
        cur_output_path = os.path.join(output_path, epoch, im_number)
        output_name = date + "_" + arch_name + ".png"
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        if os.path.exists(os.path.join(im_path, old_name)):
            shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
        # os.rename(os.path.join(im_path, old_name), os.path.join(output_path, output_name))

def gather_all_architectures_accuracy(arch_dir, output_path, epoch, date):
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "accuracy")
        old_name = "accuracy epoch = " + epoch + ".png"
        cur_output_path = os.path.join(output_path, "accuracy_" + epoch)
        output_name = date + "_" + arch_name + ".png"
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        if os.path.exists(os.path.join(im_path, old_name)):
            shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
        # os.rename(os.path.join(im_path, old_name), os.path.join(output_path, output_name))

def patchD():
    from torchsummary import summary
    import models.Discriminator
    netD = models.Discriminator.NLayerDiscriminator(1)
    summary(netD, (1, 256, 256))
    print(netD)

if __name__ == '__main__':
    patchD()
    # epochs = ['96']
    # for ep in epochs:
    #     gather_all_architectures("/Users/yaelvinker/Documents/university/lab/02_08/fix_exp",
    #                          "/Users/yaelvinker/Documents/university/lab/02_08/new_arch_summary", ep, "02_08", "1")
    im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data_generator/hard_images/belgium.hdr", format='HDR-FI')
    gray_im = hdr_image_util.to_gray(im)
    gray_im_temp = hdr_image_util.reshape_im(gray_im, 128, 128)
    brightness_factor = hdr_image_util.get_brightness_factor(gray_im_temp)
    print("belgium ", brightness_factor* 255 * 0.1)
    im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data_generator/hard_images/cathedral.hdr", format='HDR-FI')
    gray_im = hdr_image_util.to_gray(im)
    gray_im_temp = hdr_image_util.reshape_im(gray_im, 128, 128)
    brightness_factor = hdr_image_util.get_brightness_factor(gray_im_temp)
    print("cathedral ", brightness_factor * 255 * 0.1)
    im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data_generator/hard_images/synagogue.hdr", format='HDR-FI')
    gray_im = hdr_image_util.to_gray(im)
    gray_im_temp = hdr_image_util.reshape_im(gray_im, 128, 128)
    brightness_factor = hdr_image_util.get_brightness_factor(gray_im_temp)
    print("synagogue ", brightness_factor* 255 * 0.1)


    b = np.load(os.path.join("/Users/yaelvinker/PycharmProjects/lab/data_generator/b1984.npy"), allow_pickle=True)[()]
    val =np.asarray(list(b.values()))

    val = val * 255 * 0.1
    print(np.std(val))
    print(np.mean(val))
    print(np.max(val))
    print(np.median(val))

    a = np.asarray([721.9743549692752,
        3148.3651122359115,
        5551.411194663835,
        41.522882630711905,
        16586.49082955642,
        130.39035891834058,
        789.6127192856875,
        3086.3298816154415,
        1260.429918120939,
        2454.995761544146,
        452.29022691852634])
    print()
    print(np.std(a))
    print(np.mean(a))
    print(np.max(a))
    print(np.median(a))


# if __name__ == '__main__':
#     import numpy as np
#
#     # to_gray_test()
#     # back_to_color_dir("/Users/yaelvinker/Documents/university/lab/open_exr_images_tests/01_10/110_clip_11/",
#     #                   "/Users/yaelvinker/Documents/university/lab/open_exr_images_tests/01_29/exr_format_fixed_size",
#     #                   "/Users/yaelvinker/Documents/university/lab/open_exr_images_tests/01_10/110_clip_11_color")
#
#     # parse_nima_output()
#     im_exr = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/data/synagogue.hdr", format='HDR-FI')
#     im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/data/1_epoch_496_gray.png")
#     # im = imageio.imread("/Users/yaelvinker/Documents/university/lab/open_exr_images_tests/01_30/small_lr_instance_norm_g_pyramid_half_lr_decay_G_unet_original_unet_depth_4_filters_32_frame_1/105_gray_1_clip_11_05/507.png")
#     # im = im / np.max(im)
#     im = im[:, :, None]
    # im1 = (back_to_color1(im_exr, im) * 255).astype('uint8')
    # hdr_image_util.print_image_details(im1, "im1")
    # im2 = (back_to_color2(im_exr, im) * 255).astype('uint8')
    # hdr_image_util.print_image_details(im2, "im2")
    # im3 = (back_to_color3(im_exr, im) * 255).astype('uint8')
    # plt.subplot(2,1,1)
    # plt.imshow(im_exr)
    # plt.subplot(2, 1, 2)
    # plt.imshow(im3)
    # plt.show()

    # hdr_image_util.print_image_details(im3, "im3")
    # hdr_image_util.print_image_details(im, "im")
    # imageio.imwrite("/Users/yaelvinker/Documents/MATLAB/best_model/tmqi_check/original.jpg", im1)
    # imageio.imwrite("/Users/yaelvinker/Documents/MATLAB/best_model/tmqi_check/differen_gray.jpg", im2)
    # imageio.imwrite("/Users/yaelvinker/Documents/MATLAB/best_model/tmqi_check/with_norm.jpg", im3)
    # plt.subplot(2,2,1)
    # plt.imshow(im1)
    # plt.subplot(2, 2, 2)
    # plt.imshow(im2)
    # plt.subplot(2, 2, 3)
    # plt.imshow(np.squeeze(im), cmap='gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(im3)
    # plt.show()
    # im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_0006_20160726_105942_902.dng",
    #                     format="RAW-FI")
    # im = np.log(im + 1)
    # im = im / np.max(im)
    # plt.imshow(im)
    # plt.show()
    # im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data/ldr_data/ldr_data/im_96.jpg")
    # im = im / np.max(im)
    # im = (im - 0.5) / 0.5
    # hdr_image_util.print_image_details(im, "1")
    # im_exp = np.exp(im) - 1
    # hdr_image_util.print_image_details(im_exp, "2")
    # im_end = im_exp / np.max(im_exp)
    # hdr_image_util.print_image_details(im_end, "3")
    # im = (im_end - 0.5) / 0.5
    # hdr_image_util.print_image_details(im, "4")
    # f_factor_test()
    # exr_to_hdr()
    # our_custom_ssim_test()
    # ssim_test()
    # a = torch.tensor([1, 0, -1, -3])
    # b = a <= 0
    # print(b.sum())
    # print(b.count(True))
    # print(len([a <= 0]))
    # data = np.load("/Users/yaelvinker/PycharmProjects/lab/data/ldr_npy/ldr_npy/im_96_one_dim.npy", allow_pickle=True)
    # img1 = data[()]["input_image"]
    # img2 = img1 * 0.5
    # # img1 = (img1 - img1.min()) / (img1.std())
    # factor = float(2 ** 8 - 1.)
    # # factor = 1
    # img1 = factor * (img1 - img1.min()) / (img1.max() - img1.min())
    # img2 = factor * (img2 - img2.min()) / (img2.max() - img2.min())
    # # img2 = (img2 - img2.min()) / (img2.std())
    # # torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    # window = create_window(11,1).type_as(img1)
    # window = window / window.sum()
    # img1 = img1.unsqueeze(0)
    # mu1 = F.conv2d(img1, window, padding=11 // 2, groups=1)
    #
    # var_ = img1 * img1 - 2 * mu1 * img1 + mu1 * mu1
    # # var_ = var_ * var_
    # var_ = F.conv2d(var_, window, padding=11 // 2, groups=1)
    #
    # mu1_sq = mu1.pow(2)
    # sigma1_sq = F.conv2d(img1 * img1, window, padding=11 // 2, groups=1) - mu1_sq
    # std1 = (sigma1_sq).pow(0.5)
    # hdr_image_utils.print_tensor_details(std1, "std1")
    # norm_im = (img1 - mu1) / std1
    # norm_im = torch.div(torch.add(img1, -mu1), std1)
    # hdr_image_utils.print_tensor_details(var_, "var_")
    # hdr_image_utils.print_tensor_details(norm_im, "norm_im")
    # loss = torch.nn.MSELoss(reduction='mean')
    #
    # img2 = img1 * 0.5
    # mu1 = F.conv2d(img2, window, padding=11 // 2, groups=1)
    #
    # var_ = img2 * img2 - 2 * mu1 * img1 + mu1 * mu1
    # # var_ = var_ * var_
    # var_ = F.conv2d(var_, window, padding=11 // 2, groups=1)
    #
    # mu1_sq = mu1.pow(2)
    # sigma1_sq = F.conv2d(img2 * img2, window, padding=11 // 2, groups=1) - mu1_sq
    # std1 = (sigma1_sq).pow(0.5)
    # hdr_image_utils.print_tensor_details(std1, "std1")
    # norm_im2 = (img2 - mu1) / std1
    # hdr_image_utils.print_tensor_details(var_, "var_")
    # hdr_image_utils.print_tensor_details(norm_im2, "norm_im")
    # # loss = torch.nn.MSELoss(reduction='mean')
    #
    # # norm_im2 = norm_im * 0.5
    # print(loss(norm_im, norm_im2))
    # print((norm_im - norm_im2).pow(2).mean())
    # print(ssim.ssim(img1, img1 * 0.5))
    # hdr_image_utils.print_tensor_details(norm_im, "norm_im")
    # hdr_image_utils.print_tensor_details(img1, "img1")
    # plt.subplot(1,3,1)
    # plt.imshow(to_numpy_display(norm_im[0,:,:,:]), cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(to_numpy_display(img1[0, :, :, :]), cmap='gray')
    # plt.subplot(1, 3, 3)
    # im = (img1 - img1.mean()) / img1.std()
    # hdr_image_utils.print_tensor_details(im, "im")
    # plt.imshow(to_numpy_display(im[0, :, :, :]), cmap='gray')
    #
    # plt.show()
    # mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    #
    # mu1_sq = mu1.pow(2)
    # mu2_sq = mu2.pow(2)
    # mu1_mu2 = mu1 * mu2
    #
    # sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    # sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    #
    # sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    #
    # C1 = 0.01 ** 2
    # C2 = 0.03 ** 2
    #
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # if size_average:
    #     return ssim_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)

    # print(im.shape)
    # first_row = im[0, :]
    # new_first_row = np.tile(first_row, (45, 1))
    # new_im = np.vstack((new_first_row, im))
    # last_row = im[-1, :]
    # new_last_row = np.tile(last_row, (45, 1))
    # new_im = np.vstack((new_im, new_last_row))
    # plt.imshow(new_im)
    # plt.show()
    #
    # left_col = new_im[:, 0:1]
    # new_left_col = np.tile(left_col, (1, 45))
    # new_im = np.hstack((new_im, new_left_col))
    #
    # right_col = new_im[:, -2:-1]
    # new_right_col = np.tile(right_col, (1, 45))
    # new_im = np.hstack((new_right_col, new_im))
    # print(new_im.shape)
    # plt.imshow(new_im)
    # plt.show()

    # print(first_row.shape)
    # new_first_rows = np.tile(first_row, (40, 1))
    # new_last_rows = np.tile(last_row, (40, 1))
    # new_im = np.copy(im)
    # new_im = np.vstack((new_first_rows, new_im))
    # new_im = np.vstack((new_im, new_last_rows))
    # new_left_row = np.tile(new_im[:, 0], (1, 40))
    # print(new_left_row.shape)
    # print(new_im.shape)
    # new_im = np.hstack(([new_im[:, -1]], new_im))
    # print(new_im.shape)
    # plt.subplot(2, 1, 1)
    # plt.imshow(im, cmap='gray')
    # plt.subplot(2,1,2)
    # plt.imshow(new_im, cmap='gray')
    # plt.show()
    # n_img = cv2.imread("/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/belgium.hdr", cv2.IMREAD_ANYDEPTH) #reads image data
    # # n_img = cv2.cvtColor(n_img, cv2.COLOR_BGR2GRAY)
    # print(n_img.shape)
    # print(np.max(n_img))
    # print(np.min(n_img))
    # # Apply log transform.
    # # c = 255 / (np.log(1 + np.max(n_img)))
    # # n_img = c * np.log(1 + n_img)
    #
    # # histr = cv2.calcHist([n_img], [0], None)
    # # n_img = n_img.astype('uint8')
    # print(np.unique(n_img).shape[0])
    # bins = np.unique(n_img).shape[0]
    # hist, bin_edges = np.histogram(n_img, bins=bins)
    # print(hist, bin_edges)
    # plt.subplot(2, 1, 1)
    # plt.bar(bin_edges[:-1], hist, width=1.5,color='#0504aa',alpha=0.7)
    # plt.hist(n_img.ravel(),density=True, bins=256,color='#0504aa')  # calculating histogram
    # plt.plot(histr)
    # plt.subplot(2,1,2)
    # plt.imshow(n_img, cmap='gray')
    # plt.show()

    # im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/belgium.hdr",  format="HDR-FI")
    # print(im.dtype)
    # im_ldr = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data/ldr_data/ldr_data/im_97.bmp", as_gray=True)
    # im_ldr = im_ldr / np.max(im_ldr)
    # ax = plt.hist(im.ravel())
    # plt.show()
    # im = im / np.max(im)
    # ax = plt.hist(im.ravel())
    # plt.show()
    # plt.imshow(im_ldr)
    # plt.show()
    # print(im_ldr.dtype)
    # print(np.max(im_ldr))
    # histogram, bin_edges = np.histogram(im_ldr, bins=256, range=(0, 255))
    # plt.figure()
    # plt.title("Grayscale Histogram")
    # plt.xlabel("grayscale value")
    # plt.ylabel("pixels")
    # plt.xlim([0, 255])  # <- named arguments do not work here
    #
    # plt.plot(bin_edges[0:-1], histogram)  # <- or here
    # plt.show()
    # print()
    # np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    # rng = np.random.RandomState(10)  # deterministic random data
    # a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    # _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
    # plt.title("Histogram with 'auto' bins")
    # plt.text(0.5, 1.0, "Histogram with 'auto' bins")
    # plt.show()
    # model_test()
    # tmqi_test("/Users/yaelvinker/Documents/university/lab/matlab_input_niqe/belgium_res", "/Users/yaelvinker/Documents/university/lab/matlab_input_niqe/original.hdr")
    # im = imageio.imread("1 .dng", format="RAW-FI")
    # print(im.shape)
    # print(im.dtype)

    # transform_custom = transforms.Compose([
    #     transforms_.Scale(params.input_size),
    #     # transforms_.CenterCrop(params.input_size),
    #     transforms_.ToTensor(),
    #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # data_root = get_data_root()
    # print(data_root)
    # data_root_2 = "data/dng_data"
    # dataset1 = HdrImageFolder.HdrImageFolder(root=data_root,
    #                                          transform=transform_custom)
    # dataset2 = HdrImageFolder.HdrImageFolder(root=data_root_2,
    #                                          transform=transform_custom)
    #
    # dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=4,
    #                                           shuffle=False, num_workers=params.workers)
    # dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=3,
    #                                           shuffle=False, num_workers=1)
    # batch1 = (next(iter(dataloader1)))[0]
    # batch2 = (next(iter(dataloader2)))[0]
    # print_batch_details(batch1, "batch1 hdr")
    # print_batch_details(batch2, "batch2 dng")

    # im1 = imageio.imread(os.path.join(data_root_2, "dng_data","dng_image.dng"), format="RAW-FI").astype('float32')
    # im2 = imageio.imread(os.path.join(data_root_2, "dng_data","im2.dng"), format="RAW-FI")
    # im2 = imageio.imread(os.path.join(data_root_2, "dng_data","dng_image.dng"), format="RAW-FI")
    # hdr_image_utils.print_image_details(im1, "im1 dng")
    # hdr_image_utils.print_image_details(im2, "im2 dng")
    #
    # im3 = im2 + 1
    # hdr_image_utils.print_image_details(im3, "im3 dng")
