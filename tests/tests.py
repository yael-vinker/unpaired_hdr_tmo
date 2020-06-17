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
from torch import nn
from data_generator import create_dng_npy_data
import tranforms
import utils.data_loader_util as data_loader_util


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

    hdr_im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/2.hdr", format="HDR-FI")
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


def run_ssim(b1, b2):
    our_custom_sigma_loss = ssim.OUR_SIGMA_SSIM(window_size=5)
    our_ssim_loss = ssim.OUR_CUSTOM_SSIM(window_size=5)
    ssim_loss_p = ssim.OUR_CUSTOM_SSIM_PYRAMID(window_size=5,
                                               pyramid_weight_list=torch.FloatTensor(
                                                   [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))

    print("\nour_ssim_loss")
    b2a = b2 / b2.max()
    print(our_ssim_loss(b1, b1))
    print(our_ssim_loss(b2a, b1))
    print("\nour_sigma_loss")
    print(our_custom_sigma_loss(b1, b1))

    print(our_custom_sigma_loss(b1, b2))
    print("\nssim")
    print(1 - ssim.ssim(b2, b1))
    print("\npyramid")
    print(ssim_loss_p(b1, b1))
    print(ssim_loss_p(b2, b1))



def our_custom_ssim_test():
    data1 = np.load("/Users/yaelvinker/PycharmProjects/lab/data/factorised_data_original_range/flicker_use_factorise_data_1_factor_coeff_0.1_use_normalization_0/wrap/4696332335.npy", allow_pickle=True)
    data = np.load("/Users/yaelvinker/PycharmProjects/lab/data/factorised_data_original_range/hdrplus_use_factorise_data_0_factor_coeff_1000_use_normalization_1/synagogue.npy", allow_pickle=True)
    # print(datta[()].keys())
    img1 = data1[()]["input_image"]

    img2 = data1[()]["display_image"]
    im2 = hdr_image_util.to_gray_tensor(img2)
    # im2 = im2[None, :, :]
    # hdr_image_util.display_tensor(im2, 'gray')
    # plt.show()
    img1 = img1 / 255
    print(img1.max())
    print(img1.min())
    img2 = 20000*img1 + 5
    # img2 = img2 / img2.max()

    # plt.subplot(2, 1, 1)
    # hdr_image_util.display_tensor(img1, "gray")
    # plt.subplot(2, 1, 2)
    # hdr_image_util.display_tensor(img2, "gray")
    # plt.show()
    # img2 = img2 / img2.max()
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
    run_ssim(im_tone_mapped_tensor_tensor_b, im_tone_mapped_tensor_tensor_b2)

    hdr_im = imageio.imread(
        "/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/1.hdr",
        format="HDR-FI")
    hdr_im = to_gray(hdr_im)
    hdr_im_tensor = torch.from_numpy(hdr_im)
    hdr_im_tensor = hdr_im_tensor[None, :, :]
    hdr_im_tensor_b = torch.zeros([2, 1, hdr_im_tensor.shape[1], hdr_im_tensor.shape[2]])
    hdr_im_tensor_b[0, :] = hdr_im_tensor
    hdr_im_tensor_b[1, :] = hdr_im_tensor
    t_m_hdr = np.log(hdr_im / np.max(hdr_im) * 1000 + 1)
    # t_m_hdr = t_m_hdr / t_m_hdr.max()
    t_m_hdr = torch.from_numpy(t_m_hdr)
    t_m_hdr = t_m_hdr[None, :, :]

    t_m_hdr_tensor_b = torch.zeros([2, 1, t_m_hdr.shape[1], t_m_hdr.shape[2]])
    t_m_hdr_tensor_b[0, :] = t_m_hdr
    t_m_hdr_tensor_b[1, :] = t_m_hdr
    run_ssim(hdr_im_tensor_b, t_m_hdr_tensor_b)
    # plt.subplot(2,1,1)
    # hdr_image_util.display_tensor(hdr_im_tensor, "gray")
    # plt.subplot(2,1,2)
    # hdr_image_util.display_tensor(t_m_hdr, "gray")
    # plt.show()
    # print()
    # print("res")
    # print(our_ssim_loss(t_m_hdr_tensor_b, t_m_hdr_tensor_b))
    # print(our_ssim_loss(hdr_im_tensor_b, t_m_hdr_tensor_b))
    # print("ssim")
    # print(1 - ssim.ssim(hdr_im_tensor_b, t_m_hdr_tensor_b))
    # print("pyramid")
    # ssim_loss_p = ssim.OUR_CUSTOM_SSIM_PYRAMID(window_size=5,
    #                                            pyramid_weight_list=torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
    # print(t_m_hdr_tensor_b.max(), t_m_hdr_tensor_b.min())
    # print(ssim_loss_p(hdr_im_tensor_b, hdr_im_tensor_b))
    # print()
    # print(ssim_loss_p(hdr_im_tensor_b, t_m_hdr_tensor_b))


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
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "model_results", epoch)
        old_name = im_number + "_epoch_" + epoch + "_gray_source.png"
        cur_output_path = os.path.join(output_path, epoch, im_number + "_gray_source")
        output_name = date + "_" + arch_name + ".png"
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        if os.path.exists(os.path.join(im_path, old_name)):
            shutil.copy(os.path.join(im_path, old_name), os.path.join(cur_output_path, output_name))
        # os.rename(os.path.join(im_path, old_name), os.path.join(output_path, output_name))


def gather_all_architectures2(arch_dir, output_path, epoch, date, im_name):
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "test_images_format_factorised_1", epoch)
        old_name_color = os.path.join(im_path, im_name + ".jpg")
        cur_output_path = os.path.join(output_path, epoch, im_name + "_color")
        output_name_color = date + "_" + arch_name + ".jpg"
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        if os.path.exists(os.path.join(im_path, old_name_color)):
            shutil.copy(os.path.join(im_path, old_name_color), os.path.join(cur_output_path, output_name_color))
        # os.rename(os.path.join(im_path, old_name), os.path.join(output_path, output_name))



def gather_all_architectures_accuracy(arch_dir, output_path, epoch, date):
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "accuracy")
        # im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "loss_plot")
        old_name = "accuracy epoch = " + epoch + ".png"
        # old_name = "summary epoch_=_" + epoch + "all.png"
        cur_output_path = os.path.join(output_path, "accuracy_" + epoch)
        # cur_output_path = os.path.join(output_path, "loss_" + epoch)
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

def f_gamma_test(im_path):
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    print(rgb_img.shape)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.min(rgb_img)
    gray_im = hdr_image_util.to_gray(rgb_img)

    gray_im_temp = hdr_image_util.reshape_im(gray_im, 256, 256)
    brightness_factor = hdr_image_util.get_brightness_factor(gray_im_temp) * 255
    print(brightness_factor)
    gray_im_gamma = (gray_im / np.max(gray_im)) ** (1 / (1 + np.log10(brightness_factor)))
    hdr_image_util.print_image_details(gray_im_gamma, "gamma")
    plt.figure()
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.imshow(gray_im / np.max(gray_im), cmap='gray')
    gray_im_prev = (gray_im / np.max(gray_im)) * brightness_factor
    gray_im_log = np.log(gray_im_prev + 1)

    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(gray_im_log / np.max(gray_im_log), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(gray_im_gamma, cmap="gray")
    plt.show()
    return rgb_img, gray_im_gamma


def struct_loss_res(path):
    import sys
    import inspect
    import os
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    import torch.optim as optim
    import tranforms as my_transforms

    ssim_loss = ssim.OUR_CUSTOM_SSIM(window_size=5, use_c3=False,
                                     apply_sig_mu_ssim=False)

    hdr_im = imageio.imread(path, format="HDR-FI").astype('float32')
    if np.min(hdr_im) < 0:
        hdr_im = hdr_im + np.abs(np.min(hdr_im))
    hdr_im = hdr_image_util.to_gray(hdr_im)
    hdr_im = my_transforms.image_transform_no_norm(hdr_im)
    print(hdr_im.shape, hdr_im.max(), hdr_im.min())
    hdr_im = hdr_im[None, :, :, :]
    hdr_im_numpy = hdr_im[0].permute(1, 2, 0).detach().cpu().numpy()
    # im = (im - np.min(im)) / (np.max(im) - np.min(im))
    # im = (im * 255).astype('uint8')
    # imageio.imwrite("/Users/yaelvinker/PycharmProjects/lab/utils/original_im.png", im)

    img2 = torch.FloatTensor(hdr_im.size()).uniform_(0, 1)
    img1 = Variable(hdr_im, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)

    optimizer = optim.Adam([img2], lr=0.01)

    i = 0
    img_list = []
    img_list2 = []
    ssim_value = ssim_loss(img1, img2).item()
    print("Initial ssim:", ssim_value)
    while ssim_value > 0.0002:
        if i % 1000 == 0:
            im = img2[0].permute(1, 2, 0).detach().cpu().numpy()
            sub = im - hdr_im_numpy
            print("sub",sub.shape, sub.max(), sub.min())
            data = {'sub': sub}
            np.save("/Users/yaelvinker/PycharmProjects/lab/utils/sub.npy", data)
            imageio.imwrite("/Users/yaelvinker/PycharmProjects/lab/utils/sub.png", im)
            print(ssim_value)
            print("im",im.shape, im.max(), im.min())
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            im = (im * 255).astype('uint8')
            img_list2.append(im)
            imageio.imwrite("/Users/yaelvinker/PycharmProjects/lab/utils/res.png", im)

        optimizer.zero_grad()
        ssim_out = ssim_loss(img1, img2)
        ssim_value = ssim_out.item()
        ssim_out.backward()
        optimizer.step()
        i += 1
    imageio.mimsave("/Users/yaelvinker/PycharmProjects/lab/utils/res.gif", img_list2, fps=5)

def sub_test():
    import tranforms as my_transforms
    hdr_im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr", format="HDR-FI").astype('float32')
    hdr_im = hdr_image_util.to_gray(hdr_im)

    hdr_im = my_transforms.image_transform_no_norm(hdr_im)

    hdr_im = hdr_im[None, :, :, :]
    hdr_im_numpy = np.squeeze(hdr_im[0].permute(1, 2, 0).detach().cpu().numpy())
    hdr_im_numpy = (hdr_im_numpy - np.min(hdr_im_numpy)) / (np.max(hdr_im_numpy) - np.min(hdr_im_numpy))
    print("input", hdr_im_numpy.min(), hdr_im_numpy.mean(), hdr_im_numpy.max())

    plt.subplot(3,2,1)
    plt.axis("off")
    plt.title("input")
    plt.imshow(hdr_im_numpy, cmap='gray')
    plt.subplot(3, 2, 2)
    plt.hist(hdr_im_numpy.ravel(), 256, [hdr_im_numpy.min(), hdr_im_numpy.max()])

    data = np.load("/Users/yaelvinker/Downloads/output.npy", allow_pickle=True)
    input_im = np.squeeze(data[()]["sub"])
    print("output", input_im.min(), input_im.mean(), input_im.max())
    # input_im = np.clip(input_im,hdr_im_numpy.min(), 256)
    input_im = (input_im - np.min(input_im)) / (np.max(input_im) - np.min(input_im))

    plt.subplot(3,2,3)
    plt.axis("off")
    plt.title("output")
    plt.imshow(input_im, cmap='gray')
    plt.subplot(3, 2, 4)
    plt.hist(input_im.ravel(), 256, [input_im.min(), input_im.max()])


    res = (hdr_im_numpy) / (input_im + 0.0001)
    # res = (hdr_im_numpy) - (input_im)
    (unique, counts) = np.unique(res, return_counts=True)
    print((unique[counts>1]))
    print((counts[unique>1]))
    frequencies = np.asarray((unique, counts)).T
    print(frequencies.shape)
    # frequencies = np.sort(frequencies)[:2]

    print(np.sort(frequencies)[0])
    print(np.sort(frequencies)[-1])
    print("div",res.min(), res.mean(), res.max())
    res = np.clip(res, -    0.175,1)
    # res = (res - np.min(res)) / (np.max(res) - np.min(res))
    # im = (im * 255).astype('uint8')
    plt.subplot(3,2,5)
    plt.axis("off")
    plt.title("sub")
    plt.imshow(res, cmap='gray')
    plt.subplot(3, 2, 6)
    plt.hist(res.ravel(), 256, [res.min(), res.max()])
    plt.show()

def create_window(window_size=5, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    print(window)
    return window

def normalization_test():
    data = np.load("/Users/yaelvinker/PycharmProjects/lab/data/factorised_data_original_range/hdrplus_gamma1_use_factorise_data_1_factor_coeff_1.0_use_normalization_0/hdrplus_gamma_use_factorise_data_1_factor_coeff_1.0_use_normalization_0/synagogue.npy", allow_pickle=True)
    # input_im = data[()]["input_image"]
    # color_im = data[()]["display_image"]
    im_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/WillyDesk.exr")
    rgb_img, gray_im_log = create_dng_npy_data.hdr_preprocess(im_path, 1,
                                                              1,
                                                              1, reshape=True,
                                                              window_tone_map=0)
    rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
    input_im = gray_im_log
    gray_original_im = hdr_image_util.to_gray_tensor(rgb_img)
    gray_original_im_norm = gray_original_im / gray_original_im.max()
    print(torch.isnan(gray_original_im_norm).any())
    wind_size = 5
    # input_im = data_loader_util.add_frame_to_im(input_im)
    # gray_original_im_norm = data_loader_util.add_frame_to_im(gray_original_im_norm)
    m = nn.ZeroPad2d(wind_size // 2)
    input_im = m(input_im)
    gray_original_im_norm = m(gray_original_im_norm)
    hdr_image_util.print_tensor_details(input_im, "input_im")
    hdr_image_util.print_tensor_details(gray_original_im_norm, "gray_original_im_norm")

    b = torch.unsqueeze(input_im, dim=0)
    b_wind = b[:, :, 0:5, 0:5]
    a = torch.unsqueeze(gray_original_im_norm, dim=0)
    a_wind = a[:, :, 0:5, 0:5]

    # a = torch.rand((1, 1, 20, 20))
    windows = a.unfold(dimension=2, size=5, step=1)
    windows = windows.unfold(dimension=3, size=5, step=1)
    windows = windows.reshape(windows.shape[0], windows.shape[1],
                              windows.shape[2], windows.shape[3],
                              wind_size * wind_size)


    # window = create_window()
    window = torch.ones((1,1,5,5))
    window = window / window.sum()
    mu1 = F.conv2d(a, window, padding=0, groups=1)
    print(torch.isnan(mu1).any())
    mu2 = F.conv2d(b, window, padding=0, groups=1)
    print(torch.isnan(mu2).any())
    # print("mu1", mu1[0,0,100,100], torch.mean(a_wind))
    # print("mu2", mu2[0,0,100,100], torch.mean(b_wind))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(a * a, window, padding=0, groups=1) - mu1_sq
    print(torch.isnan(sigma1_sq).any())
    sigma2_sq = F.conv2d(b * b, window, padding=0, groups=1) - mu2_sq
    print(torch.isnan(sigma2_sq).any())
    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon, 0.5)
    print(torch.isnan(std1).any())
    std2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)) + params.epsilon, 0.5)
    print(torch.isnan(std2).any())
    norm_std2 = std2 / (torch.max(mu2, torch.zeros_like(sigma1_sq)) + params.epsilon)
    # std2_normalise = std2 / torch.max(sigma1_sq, torch.zeros_like(sigma1_sq) + params.epsilon)
    # # std2_normalise = torch.pow(std2_normalise, 0.8)
    # std2_normalise = torch.pow((std2 / mu2), 0.8)

    compressed_std = torch.pow(norm_std2, 0.8)
    print("\ntarget mu", mu2[0, 0, 100, 100])
    print("target std", compressed_std[0, 0, 100, 100])
    print(torch.isnan(compressed_std).any())
    # print("\nstd1", std1[0,0,100,100], torch.std(a_wind))
    # print("compressed_std", compressed_std[0,0,100,100], torch.std(b_wind))


    mu1 = mu1.unsqueeze(dim=4)
    std1 = std1.unsqueeze(dim=4)
    mu1 = mu1.expand(-1, -1, -1, -1, wind_size * wind_size)
    std1 = std1.expand(-1, -1, -1, -1, wind_size * wind_size)
    windows = windows - mu1
    windows = windows / (std1 + params.epsilon)
    print(torch.isnan(windows).any())

    # print("\nmu1", torch.mean(windows[0, 0, 0, 0]))
    # print("std1", torch.std(windows[0, 0, 0, 0]))

    mu2 = mu2.unsqueeze(dim=4)
    compressed_std = compressed_std.unsqueeze(dim=4)
    mu2 = mu2.expand(-1, -1, -1, -1, 25)
    compressed_std = compressed_std.expand(-1, -1, -1, -1, 25)
    windows = windows * compressed_std
    windows = windows + mu2
    # print("\nmu1", torch.mean(windows[0, 0, 100,100]))
    # print("std1", torch.std(windows[0, 0, 100,100]))
    print(windows.shape)
    print("\nres mu", torch.mean(windows[0, 0, 100, 100]))
    print("res std", torch.std(windows[0, 0, 100, 100]))
    windows = windows.permute(0, 4, 2, 3, 1)
    # print(windows.shape)
    windows = windows.squeeze(dim=4)
    # print(windows.shape)
    # hdr_image_util.print_tensor_details(windows, "b")
    print(torch.isnan(windows).any())


def get_window_and_set_device(wind_size, a, b):
    from torch import nn
    # m = nn.ZeroPad2d(wind_size // 2)
    # a, b = m(a), m(b)
    if a.dim() < 4:
        a = torch.unsqueeze(a, dim=0)
    if b.dim() < 4:
        b = torch.unsqueeze(b, dim=0)
    window = torch.ones((1, 1, wind_size, wind_size))
    window = window / window.sum()
    if a.is_cuda:
        window = window.cuda(a.get_device())
        if not b.is_cuda:
            b = b.cuda(a.get_device())
    b = b.type_as(a)
    window = window.type_as(a)
    return a, b, window


def get_mu_and_std(x, window):
    mu = F.conv2d(x, window, padding=0, groups=1)
    mu_sq = mu.pow(2)
    sigma_sq = F.conv2d(x * x, window, padding=0, groups=1) - mu_sq
    std = torch.pow(torch.max(sigma_sq, torch.zeros_like(sigma_sq)) + params.epsilon, 0.5)
    return mu, std


def get_mu(x, window, wind_size):
    mu = F.conv2d(x, window, padding=wind_size // 2, groups=1)
    mu = mu.unsqueeze(dim=4)
    mu = mu.expand(-1, -1, -1, -1, wind_size * wind_size)
    return mu


def get_im_as_windows(a, wind_size):
    m = nn.ZeroPad2d(wind_size // 2)
    a = m(a)
    windows = a.unfold(dimension=2, size=5, step=1)
    windows = windows.unfold(dimension=3, size=5, step=1)
    windows = windows.reshape(windows.shape[0], windows.shape[1],
                              windows.shape[2], windows.shape[3],
                              wind_size * wind_size)
    return windows


def get_std(windows, mu1, wind_size):
    x_minus_mu = windows - mu1
    x_minus_mu = x_minus_mu.squeeze(dim=1)
    x_minus_mu = x_minus_mu.permute(0, 3, 1, 2)
    wind_a = torch.ones((1, wind_size * wind_size, 1, 1))
    if windows.is_cuda:
        wind_a = wind_a.cuda(windows.get_device())
    wind_a = wind_a / wind_a.sum()
    mu_x_minus_mu_sq = F.conv2d(x_minus_mu * x_minus_mu, wind_a, padding=0, groups=1)
    std1 = torch.pow(torch.max(mu_x_minus_mu_sq, torch.zeros_like(mu_x_minus_mu_sq)) + params.epsilon, 0.5)
    std1 = std1.expand(-1, wind_size * wind_size, -1, -1)
    std1 = std1.permute(0, 2, 3, 1)
    std1 = std1.unsqueeze(dim=1)
    return std1


def get_std2(x, mu1, wind_size, window):
    window = window / window.sum()
    mu1_sq = mu1.pow(2)
    sigma1_sq = F.conv2d(x * x, window, padding=wind_size // 2, groups=1) - mu1_sq
    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon, 0.5)
    return std1


def get_mu2(x, window, wind_size):
    window = window / window.sum()
    mu1 = F.conv2d(x, window, padding=wind_size // 2, groups=1)
    return mu1


def fix_shape(windows):
    windows = windows.permute(0, 4, 2, 3, 1)
    windows = windows.squeeze(dim=4)
    return windows


def custom_ssim_a(wind_size, a, b, std_norm_factor):
    """
    compute L2[(a - mu_a) / std_a, (b - mu_b) / std_b)] where a is G(gamma(hdr)

    :param wind_size:
    :param a: original_im
    :param b: target_im
    :param std_norm_factor
    :return:
    """
    img1_c, img2_c = a.clone(), b.clone()
    a, b, window = get_window_and_set_device(wind_size, a, b)
    img1 = get_im_as_windows(a, wind_size)
    img2 = get_im_as_windows(b, wind_size)
    # print("\na_original : mu", torch.mean(windows[0, 0, 5, 5]), " res std", torch.std(windows[0, 0, 5, 5]))

    # mu1 = get_mu(a, window, wind_size)
    # mu2 = get_mu(b, window, wind_size)
    #
    # std1 = get_std(img1, mu1, wind_size)
    # std2 = get_std(img2, mu2, wind_size)

    mu1 = torch.mean(img1, dim=4).unsqueeze(dim=4).expand(-1, -1, -1, -1, wind_size * wind_size)
    mu2 = torch.mean(img2, dim=4).unsqueeze(dim=4).expand(-1, -1, -1, -1, wind_size * wind_size)

    std1 = torch.std(img1, dim=4).unsqueeze(dim=4).expand(-1, -1, -1, -1, wind_size * wind_size)
    std2 = torch.std(img2, dim=4).unsqueeze(dim=4).expand(-1, -1, -1, -1, wind_size * wind_size)

    # print_mu(mu1, "my_mu", img1_c)
    # print_mu(std1, "my_std", img1_c)
    # print_mu(mu2, "my_mu", img2_c)
    # print_mu(std2, "my_std", img2_c)

    # eps = (std1).min() * 0.00001
    # print(std1 + eps)
    # print(img1.shape)
    # print(mu1.shape)
    img1 = (img1 - mu1)
    print("std1 min[%.4f] mean[%.4f] max[%.4f]" % (std1.min(), std1.mean(), std1.max()))
    print("img1 min[%.4f] mean[%.4f] max[%.4f]" % (img1.min(), img1.mean(), img1.max()))
    std1 = torch.max(std1, torch.zeros_like(std1)) + params.epsilon
    img1 = img1 / (std1)
    # print_mu(std1, "my_std1_std", img1_c)
    # print_mu(img1, "my_img1_std", img1_c)

    # eps = (std2).min() * 0.0001
    # print((std2).max())
    std2 = torch.max(std2, torch.zeros_like(std2)) + params.epsilon
    # print(torch.isnan(std2).any())
    # print(img2[0, 0, 1009, 1730])
    # print(mu2[0, 0, 1009, 1730])
    img2 = (img2 - mu2)
    print("std2 min[%.4f] mean[%.4f] max[%.4f]" % (std2.min(), std2.mean(), std2.max()))
    print("img2 min[%.4f] mean[%.4f] max[%.4f]" % (img2.min(), img2.mean(), img2.max()))

    # print_mu(img2, "my_img2_std", img2_c)
    # print(std2[0, 0, 1009, 1730])
    img2 = img2 / (std2)
    # print("my",0/0.0001)
    # print_mu(img2, "my_img2_std", img2_c)
    # print(img2[0,0,1009,1730])
    # print((torch.isnan(img2[0,0,:,:,0])).nonzero())
    mse_loss = torch.nn.MSELoss()
    # img1, img2 = fix_shape(img1), fix_shape(img2)
    # print((torch.isnan(img2[0, 0, :, :, 0])).nonzero())
    # print((torch.isnan(img1[0, 0, :, :, 0])).nonzero())
    res = mse_loss(img1, img2)
    # print(res)
    # print((torch.isnan(res[0, 0, :, :, 0])).nonzero())
    return mse_loss(img1, img2), (std1), mu1


def custom_ssim_b(wind_size, a, b, std_norm_factor):
    """
    compute L2[(a-mu_a),(b-mu_b)*s)] where a is G(gamma(hdr)
    :param wind_size:
    :param a: original_im
    :param b: target_im
    :param std_norm_factor
    :return:
    """
    mse_loss = torch.nn.MSELoss()
    a, b, window = get_window_and_set_device(wind_size, a, b)
    windows = get_im_as_windows(a, wind_size)
    windows_b = get_im_as_windows(b, wind_size)
    print("\na_original : mu", torch.mean(windows[0, 0, 100, 100]), " res std", torch.std(windows[0, 0, 100, 100]))

    mu1 = get_mu(a, window, wind_size)
    mu2 = get_mu(b, window, wind_size)

    std2 = get_std(windows_b, mu2, wind_size)

    windows = windows - mu1
    print("\na : res mu", torch.mean(windows[0, 0, 100, 100]), " res std", torch.std(windows[0, 0, 100, 100]))

    windows_b = windows_b - mu2
    print(windows.shape, windows_b.shape)
    windows_b = std_norm_factor * windows_b * (mu1 / (std2 + params.epsilon))
    print(windows.shape, windows_b.shape)
    print("\nb : res mu", torch.mean(windows_b[0, 0, 100, 100])," res std", torch.std(windows_b[0, 0, 100, 100]))
    print(windows.shape, windows_b.shape)
    windows, windows_b = fix_shape(windows), fix_shape(windows_b)
    print(windows.shape, windows_b.shape)
    return mse_loss(windows, windows_b)


def custom_ssim_c(wind_size, a, b, std_norm_factor):
    """
    compute L2[(a-mu_a),(b-mu_b)*s)] where a is G(gamma(hdr)
    :param wind_size:
    :param a: original_im
    :param b: target_im
    :param std_norm_factor
    :return:
    """
    mse_loss = torch.nn.MSELoss()
    a, b, window = get_window_and_set_device(wind_size, a, b)
    windows = get_im_as_windows(a, wind_size)
    windows_b = get_im_as_windows(b, wind_size)
    print("\na_original : mu", torch.mean(windows[0, 0, 100, 100]), " res std", torch.std(windows[0, 0, 100, 100]))

    mu1 = get_mu(a, window, wind_size)
    mu2 = get_mu(b, window, wind_size)

    windows = windows - mu1
    print("\na : res mu", torch.mean(windows[0, 0, 100, 100]), " res std", torch.std(windows[0, 0, 100, 100]))

    windows_b = windows_b - mu2
    windows_b = std_norm_factor * windows_b * (mu1 / (mu2 + params.epsilon))
    print("\nb : res mu", torch.mean(windows_b[0, 0, 100, 100])," res std", torch.std(windows_b[0, 0, 100, 100]))
    windows, windows_b = fix_shape(windows), fix_shape(windows_b)
    print(windows.shape, windows_b.shape)
    return mse_loss(windows, windows_b)



def get_real_std(im, wind_size=5):
    im = im.unsqueeze(dim=0)
    res = np.zeros(im.shape)
    res_mu = np.zeros(im.shape)
    m = nn.ZeroPad2d(wind_size // 2)
    im = m(im)
    im = im.unfold(dimension=2, size=5, step=1)
    im = im.unfold(dimension=3, size=5, step=1)
    im = im.reshape(im.shape[0], im.shape[1],
                              im.shape[2], im.shape[3],
                              wind_size * wind_size)
    res_s = torch.std(im, dim=4)
    print(res_s.shape)
    print(res_s[0,0,0,1086])
    print(im.shape)
    print(im[0,0,0,1086])
    print("mean",im[0, 0, 0, 1086].mean(), "std",im[0, 0, 0, 1086].std())
    # for i in range(im.shape[2]):
    #     for j in range(im.shape[3]):
    #         res[0, 0, i, j] = (im[0, 0, i, j].std())
    #         res_mu[0, 0, i, j] = (im[0, 0, i, j].mean())
    # print("mean", res_mu[0, 0, 0, 1086], "std", res[0, 0, 0, 1086])
    # print(res.shape)
    return torch.std(im, dim=4).numpy(), torch.mean(im, dim=4).numpy()
    # print(im.shape)



def new_ssim_test():
    # im_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/WillyDesk.exr")
    im_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr")
    rgb_img, gray_im_log = create_dng_npy_data.hdr_preprocess(im_path, 1,
                                                              1,
                                                              1, reshape=True,
                                                              window_tone_map=0)
    rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
    input_im = gray_im_log
    gray_original_im = hdr_image_util.to_gray_tensor(rgb_img)
    gray_original_im_norm = gray_original_im / gray_original_im.max()
    wind_size = 5
    # resl_std, real_mu = get_real_std(input_im)
    input_im = input_im.unsqueeze(dim=0)
    input_im = torch.cat([input_im], dim=0)
    our_ssim_loss = ssim.OUR_CUSTOM_SSIM(wind_size, struct_method="reg_ssim")

    # print(input_im.shape)
    gray_original_im_norm = gray_original_im_norm.unsqueeze(dim=0)
    gray_original_im_norm = torch.cat([gray_original_im_norm], dim=0)
    window = torch.ones((1, 1, wind_size, wind_size))
    # res, std_my, mu_my = custom_ssim_a(wind_size, input_im, gray_original_im_norm, 1)
    # print("my", res)
    print("======================================")

    # res, res_a = ssim.our_custom_ssim_3(input_im, input_im, window, wind_size, channel=1, mse_loss=torch.nn.MSELoss())
    # print("my_from_ssim", res)
    #
    # res, res_b = our_ssim_loss(input_im, input_im)
    # print(res_b.shape)
    # print("other_original", res)
    # sub = res_a - res_b
    # plt.imshow(sub[0,0].detach().numpy(), cmap='gray')
    # plt.show()



    print("======================================")

    res, res_a, res2, res_b = ssim.our_custom_ssim_3(input_im, gray_original_im_norm, window, wind_size, channel=1, mse_loss=torch.nn.MSELoss())
    print(res_a[0,0,566, 701])
    print("my_from_ssim", res)
    res1 = our_ssim_loss(input_im, gray_original_im_norm)
    print(res_b[0,0,566,701])
    print("other_original", res2)
    print("other_original", res1)
    plt.show()
    sub = res_a - res_b
    plt.imshow(sub[0, 0].detach().numpy(), cmap='gray')
    plt.show()



    # std_my, mu_my = std_my.numpy()[:,:,:,:,0], mu_my.numpy()[:,:,:,:,0]
    # std_other, mu_other = std_other.numpy()[:,:,:,:,0], mu_other.numpy()[:,:,:,:,0]
    # # std_my = my_floor(std_my, precision=4)
    # # std_other = my_floor(std_other, precision=4)
    # # print(resl_std.shape)
    # # print(std_my.shape)
    # # print(np.sort(std_my[(resl_std != std_my)])[::-1].shape)
    # print("================= my =====================")
    # # print(np.transpose((resl_std[0,0] != std_my[0,0]).nonzero()))
    # print(np.sum((resl_std != std_my))/(4*1*1024*2048*25),"\n")
    # # print(np.sort(std_my[(resl_std != std_my)])[::-1][0])
    # print()
    # sub_my = np.abs(resl_std[0,0] - std_my[0,0])
    # sub_other = np.abs(resl_std[0,0] - std_other[0,0])
    # a = np.unravel_index(sub_my.argmax(), sub_my.shape)
    # # print(a)
    # print("======= std ========")
    # print("real  std", resl_std[0,0][a], "mu", real_mu[0,0][a])
    # print("my", a, "std", std_my[0,0][a], sub_my[a], "mu", mu_my[0,0][a])
    #
    # a = np.unravel_index(sub_other.argmax(), sub_other.shape)
    # print("other", a, "std",std_other[0,0][a], sub_other[a],"mu", mu_other[0,0][a])
    # print()
    # sub_my = np.abs(real_mu[0, 0] - mu_my[0, 0])
    # sub_other = np.abs(real_mu[0, 0] - mu_other[0, 0])
    # a = np.unravel_index(sub_my.argmax(), sub_my.shape)
    # # print(a)
    # print("======= mu ========")
    # print("real", real_mu[0, 0][a])
    # print("my", a,  mu_my[0,0][a], sub_my[a])
    #
    # a = np.unravel_index(sub_other.argmax(), sub_other.shape)
    # print("other", a, mu_other[0,0][a], sub_other[a])
    # # print("real", resl_std[0, 0, 0, 2])
    # # print("std_my", std_my[0, 0, 0, 2])
    # # print("std_other", std_other[0, 0, 0, 2])
    #
    #
    # # print("real",np.sort(resl_std[(resl_std != std_my)])[::-1][0])
    # # print()
    # # print(np.sort(std_my[(resl_std != std_my)])[::-1][-1])
    # # print("real",np.sort(resl_std[(resl_std != std_my)])[::-1][-1])
    #
    # print("================= other =====================")
    # print(np.transpose((resl_std[0, 0] != std_other[0, 0]).nonzero()))
    # print(np.sum((resl_std != std_other)) / (4 * 1 * 1024 * 2048 * 25))
    # print()
    # print(np.sort(std_other[(resl_std != std_other)])[::-1][0])
    # print(np.sort(std_other[(resl_std != std_other)])[::-1][-1])
    # # print(np.sort(resl_std[(resl_std != std_other)])[::-1][0,0,0,0,0])

    print()


def my_floor(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)


def our_custom_ssim2(img1, img2, window, window_size, channel, mse_loss):
    window = window / window.sum()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq

    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)), 0.5)
    std2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)), 0.5)

    # print("std1 min[%f] mean[%f] max[%f]" % (std1.min(), std1.mean(), std1.max()))
    # print("img1 min[%f] mean[%f] max[%f]" % (img1.min(), img1.mean(), img1.max()))
    # print("std2 min[%f] mean[%f] max[%f]" % (std2.min(), std2.mean(), std2.max()))
    # print("img2 min[%f] mean[%f] max[%f]" % (img2.min(), img2.mean(), img2.max()))

    mu1 = mu1.unsqueeze(dim=4).expand(-1, -1, -1, -1, window_size * window_size)
    mu2 = mu2.unsqueeze(dim=4).expand(-1, -1, -1, -1, window_size * window_size)

    std1 = std1.unsqueeze(dim=4).expand(-1, -1, -1, -1, window_size * window_size)
    std2 = std2.unsqueeze(dim=4).expand(-1, -1, -1, -1, window_size * window_size)

    img1 = get_im_as_windows(img1, window_size)
    img2 = get_im_as_windows(img2, window_size)
    img1 = (img1 - mu1)
    img1 = img1 / (std1 + params.epsilon2)
    img2 = (img2 - mu2)
    img2 = (img2) / (std2 + params.epsilon2)
    return mse_loss(img1, img2)






def print_mu(mu, title, x):
    print("\n-------- %s --------" % title)
    print(mu.shape)
    print("min[%.4f] max[%.4f] mean[%.4f]" % (mu.min(), mu.max(), mu.mean()))
    real_wind = x[0, 0, 100-2: 100+3, 100-2: 100+3]

    a = mu[0, 0, 100, 100]
    if mu.dim() > 4:
        a = a[0]
    if "std" in title:
        print("wind[100, 100] -- real[%.4f] custom[%.4f]" % (torch.std(real_wind), a))
    else:
        print("wind[100, 100] -- real[%.4f] custom[%.4f]" % (torch.mean(real_wind), a))

    a = mu[0, 0, 2, 2]
    if mu.dim() > 4:
        a = a[0]
    real_wind = x[0, 0, 0: 5, 0: 5]
    if "std" in title:
        print("wind[2, 2] -- real[%.4f] custom[%.4f]" % (torch.std(real_wind), a))
    else:
        print("wind[2, 2] -- real[%.4f] custom[%.4f]" % (torch.mean(real_wind), a))

    a = mu[0, 0, 0, 0]
    if mu.dim() > 4:
        a = a[0]
    print("wind [0, 0]  [%.4f]" % a)


def mu_std_test():
    im_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/WillyDesk.exr")
    rgb_img, gray_im_log = create_dng_npy_data.hdr_preprocess(im_path, 1,
                                                              1,
                                                              1, reshape=True,
                                                              window_tone_map=0)
    rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
    input_im = gray_im_log
    gray_original_im = hdr_image_util.to_gray_tensor(rgb_img)
    gray_original_im_norm = gray_original_im / gray_original_im.max()
    wind_size = 5
    window = torch.ones((1, 1, wind_size, wind_size))
    input_im2, gray_original_im_norm, window2 = get_window_and_set_device(wind_size, input_im, gray_original_im_norm)
    windows = get_im_as_windows(input_im2, wind_size)

    input_im = input_im.unsqueeze(dim=1)
    print("input_im2",input_im2.shape)
    print(input_im.shape)
    my_mu = get_mu(input_im2, window2, wind_size)
    print_mu(my_mu, "my_mu", input_im)
    other_mu = get_mu2(input_im, window, wind_size)
    print_mu(other_mu, "other_mu", input_im)

    my_std = get_std(windows, my_mu, wind_size)
    print_mu(my_std, "my_std", input_im)
    other_std = get_std2(input_im, other_mu, wind_size, window)
    print_mu(other_std, "other_std", input_im)


def struct_loss_formula_test():
    mse_loss = torch.nn.MSELoss()
    x = torch.rand((5))
    # y = torch.log(x)
    y = torch.rand((5))

    mu1, mu2 = torch.mean(x), torch.mean(y)
    std1, std2 = torch.std(x), torch.std(y)

    print("x\n", x)
    print("mu[%f] std[%f]" % (mu1, std1))
    print("y\n", y)
    print("mu[%f] std[%f]" % (mu2, std2))

    x_a = (x - mu1) / std1
    y_b = (y - mu2) / std2
    res_a = mse_loss(x_a, y_b)
    sigma12 = np.cov((x - mu1).numpy(), (y - mu2).numpy(), bias=True)[0][1]
    res_b_bias_true = 2 - 2*(sigma12 / (std1 * std2))
    sigma12 = np.cov(x.numpy(), y.numpy(), bias=False)[0][1]
    res_b_bias_false = 2 - 2 * (sigma12 / (std1 * std2))
    print("\noption a ", res_a)
    print("option b bias=True", res_b_bias_true)
    print("option b bias=False", res_b_bias_false)


def struct_loss_formula_test2():
    # mse_loss = torch.nn.MSELoss()
    x = np.random.rand(25)
    x_1 = torch.from_numpy(x)
    y = np.log(x)
    # y = x
    # y = np.random.rand(5)
    y_1 = torch.from_numpy(y)

    mu1, mu2 = np.mean(x), np.mean(y)
    mu1_torch, mu2_torch = torch.mean(x_1), torch.mean(y_1)
    std1, std2 = np.std(x), np.std(y)
    std1_torch, std2_torch = torch.std(x_1), torch.std(y_1)

    mu1_sq = mu1_torch.pow(2)
    mu2_sq = mu2_torch.pow(2)
    sigma1_sq = torch.mean(x_1 * x_1) - mu1_sq
    sigma2_sq = torch.mean(y_1 * y_1) - mu2_sq
    std1_torch2, std2_torch2 = torch.pow(sigma1_sq, 0.5), torch.pow(sigma2_sq, 0.5)

    print("x \n", x)
    print("numpy mu[%f] std[%f]" % (mu1, std1))
    print("torch mu[%f] std[%f]" % (mu1_torch, std1_torch))
    print("torch2 mu[%f] std[%f]" % (mu1_torch, std1_torch2))
    print("y numpy\n", y)
    print("mu[%f] std[%f]" % (mu2, std2))
    print("torch mu[%f] std[%f]" % (mu2_torch, std2_torch))
    print("torch2 mu[%f] std[%f]" % (mu2_torch, std2_torch2))

    x_a = (x - mu1) / std1
    y_b = (y - mu2) / std2
    res_a = np.mean((x_a-y_b) ** 2)
    sigma12 = np.cov(x, y, bias=True)[0][1]
    res_b_bias_true = 2 - 2 * (sigma12 / (std1 * std2))
    # sigma12 = np.cov(x.numpy(), y.numpy(), bias=False)[0][1]
    res_b_bias_false = 2 - 2 * (sigma12 / (std1 * std2))
    print("\noption a ", res_a)
    print("option b bias=True", res_b_bias_true)
    # print("option b bias=False", res_b_bias_false)


def f_test():
    root = "/Users/yaelvinker/PycharmProjects/lab/utils/temp_data"
    original_hdr_images = []
    counter = 1
    for img_name in os.listdir(root):
        im_path = os.path.join(root, img_name)
        print(img_name)
        rgb_img, gray_im_log = create_dng_npy_data.hdr_preprocess(im_path, True,
                                                                  True,
                                                                  1, reshape=False,
                                                                  window_tone_map=False,
                                                                  calculate_f=True)
        rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
        # gray_im_log = data_loader_util.add_frame_to_im(gray_im_log)
        # original_hdr_images.append({'im_name': str(counter),
        #                             'im_hdr_original': rgb_img,
        #                             'im_log_normalize_tensor': gray_im_log,
        #                             'epoch': 0})
        # if counter == 1 or counter == 2:
        #     self.get_test_image_special_factor(im_path, original_hdr_images, counter)

        # plt.imshow(gray_im_log.clone().permute(1, 2, 0).detach().cpu().squeeze().numpy(), cmap='gray')
        # title = "max %.4f min %.4f mean %.4f" % (gray_im_log.max().item(),gray_im_log.min().item(), gray_im_log.mean().item())
        # plt.title(title, fontSize=8)
        # plt.show()
        counter += 1
    return original_hdr_images

def gather_all_architectures3(arch_dir, output_path, epoch):
    from shutil import copyfile
    import shutil
    # copyfile(src, dst)
    for arch_name in os.listdir(arch_dir):
        im_path = os.path.join(os.path.abspath(arch_dir), arch_name, "hdr_test_format_factorised_1", epoch)
        cur_output_path = os.path.join(output_path, arch_name)
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        for im_name_ext in os.listdir(im_path):
            if os.path.splitext(im_name_ext)[1] == ".jpg":
                shutil.copy(os.path.join(im_path, im_name_ext), os.path.join(cur_output_path, im_name_ext))


def parse_text_file(file_path):
    with open(file_path) as fp:
        line_path = fp.readline()
        cnt = 0
        while line_path:
            file_name = os.path.basename(line_path)
            print(file_name)
            line_number = fp.readline()
            print(line_number)
            line_factor = float(fp.readline()[18:])
            print(line_factor)
            line_path = fp.readline()
            cnt += 3

def nima_parse():
    total_our = 0
    total_paris = 0
    total_durand = 0
    total_fattal = 0
    res_our = [
      {
        "image_id": "BigfootPass_stretch",
        "mean_score_prediction": 4.910272464156151
      },
      {
        "image_id": "BandonSunset(2)_stretch",
        "mean_score_prediction": 5.087489753961563
      },
      {
        "image_id": "RITTiger_stretch",
        "mean_score_prediction": 4.6551591493189335
      },
      {
        "image_id": "LabWindow_stretch",
        "mean_score_prediction": 5.447415832430124
      },
      {
        "image_id": "TunnelView(2)_stretch",
        "mean_score_prediction": 4.5853050127625465
      },
      {
        "image_id": "TheGrotto_stretch",
        "mean_score_prediction": 4.843858294188976
      },
      {
        "image_id": "BarHarborPresunrise_stretch",
        "mean_score_prediction": 4.840325199067593
      },
      {
        "image_id": "LuxoDoubleChecker_stretch",
        "mean_score_prediction": 5.016093768179417
      },
      {
        "image_id": "OCanadaNoLights_stretch",
        "mean_score_prediction": 5.615224987268448
      },
      {
        "image_id": "SunsetPoint(1)_stretch",
        "mean_score_prediction": 4.386003360152245
      },
      {
        "image_id": "LittleRiver_stretch",
        "mean_score_prediction": 4.826734788715839
      },
      {
        "image_id": "LetchworthTeaTable(2)_stretch",
        "mean_score_prediction": 4.818279214203358
      },
      {
        "image_id": "JesseBrownsCabin_stretch",
        "mean_score_prediction": 4.662121616303921
      },
      {
        "image_id": "RedwoodSunset_stretch",
        "mean_score_prediction": 4.429075911641121
      },
      {
        "image_id": "BarHarborSunrise_stretch",
        "mean_score_prediction": 5.256455957889557
      },
      {
        "image_id": "BalancedRock_stretch",
        "mean_score_prediction": 4.549564868211746
      },
      {
        "image_id": "WallDrug_stretch",
        "mean_score_prediction": 4.708578124642372
      },
      {
        "image_id": "HancockKitchenInside_stretch",
        "mean_score_prediction": 5.609829604625702
      },
      {
        "image_id": "SmokyTunnel_stretch",
        "mean_score_prediction": 4.623952902853489
      },
      {
        "image_id": "MtRushmore(1)_stretch",
        "mean_score_prediction": 4.499650701880455
      },
      {
        "image_id": "WillySentinel_stretch",
        "mean_score_prediction": 5.211701080203056
      },
      {
        "image_id": "AmikeusBeaverDamPM2_stretch",
        "mean_score_prediction": 4.56564674526453
      },
      {
        "image_id": "MackinacBridge_stretch",
        "mean_score_prediction": 5.2605311423540115
      },
      {
        "image_id": "OCanadaLights_stretch",
        "mean_score_prediction": 5.506892077624798
      },
      {
        "image_id": "DelicateArch_stretch",
        "mean_score_prediction": 4.793216463178396
      },
      {
        "image_id": "BloomingGorse(1)_stretch",
        "mean_score_prediction": 5.244456797838211
      },
      {
        "image_id": "RoadsEndFireDamage_stretch",
        "mean_score_prediction": 5.046627260744572
      },
      {
        "image_id": "Zentrum_stretch",
        "mean_score_prediction": 4.452933043241501
      },
      {
        "image_id": "Exploratorium(2)_stretch",
        "mean_score_prediction": 5.505869567394257
      },
      {
        "image_id": "TupperLake(2)_stretch",
        "mean_score_prediction": 4.118054032325745
      },
      {
        "image_id": "507_stretch",
        "mean_score_prediction": 4.886837936937809
      },
      {
        "image_id": "KingsCanyon_stretch",
        "mean_score_prediction": 5.06132534891367
      },
      {
        "image_id": "ElCapitan_stretch",
        "mean_score_prediction": 4.674701817333698
      },
      {
        "image_id": "MiddlePond_stretch",
        "mean_score_prediction": 4.830470830202103
      },
      {
        "image_id": "Route66Museum_stretch",
        "mean_score_prediction": 5.636515229940414
      },
      {
        "image_id": "Peppermill_stretch",
        "mean_score_prediction": 5.165686219930649
      },
      {
        "image_id": "CemeteryTree(1)_stretch",
        "mean_score_prediction": 4.741307616233826
      },
      {
        "image_id": "WillyDesk_stretch",
        "mean_score_prediction": 5.61846908275038
      },
      {
        "image_id": "HooverGarage_stretch",
        "mean_score_prediction": 5.1094381138682365
      },
      {
        "image_id": "OtterPoint_stretch",
        "mean_score_prediction": 5.093752548098564
      },
      {
        "image_id": "URChapel(1)_stretch",
        "mean_score_prediction": 4.972276620566845
      },
      {
        "image_id": "AmikeusBeaverDamPM1_stretch",
        "mean_score_prediction": 5.025993816554546
      },
      {
        "image_id": "MammothHotSprings_stretch",
        "mean_score_prediction": 4.271407634019852
      },
      {
        "image_id": "TaughannockFalls_stretch",
        "mean_score_prediction": 4.686946965754032
      },
      {
        "image_id": "HooverDam_stretch",
        "mean_score_prediction": 4.77139875292778
      },
      {
        "image_id": "TheNarrows(2)_stretch",
        "mean_score_prediction": 4.386139698326588
      },
      {
        "image_id": "OldFaithfulInn_stretch",
        "mean_score_prediction": 5.202473238110542
      },
      {
        "image_id": "LabBooth_stretch",
        "mean_score_prediction": 5.517950791865587
      },
      {
        "image_id": "FourCornersStorm_stretch",
        "mean_score_prediction": 4.904638633131981
      },
      {
        "image_id": "DevilsGolfCourse_stretch",
        "mean_score_prediction": 4.87747473269701
      },
      {
        "image_id": "GoldenGate(1)_stretch",
        "mean_score_prediction": 5.667317543178797
      },
      {
        "image_id": "DevilsTower_stretch",
        "mean_score_prediction": 4.640577495098114
      },
      {
        "image_id": "SequoiaRemains_stretch",
        "mean_score_prediction": 5.089017242193222
      },
      {
        "image_id": "CanadianFalls_stretch",
        "mean_score_prediction": 5.034867092967033
      },
      {
        "image_id": "HalfDomeSunset_stretch",
        "mean_score_prediction": 4.738789223134518
      },
      {
        "image_id": "WestBranchAusable(1)_stretch",
        "mean_score_prediction": 4.915320605039597
      },
      {
        "image_id": "TheNarrows(3)_stretch",
        "mean_score_prediction": 4.711455427110195
      },
      {
        "image_id": "HallofFame_stretch",
        "mean_score_prediction": 5.446520194411278
      },
      {
        "image_id": "MasonLake(2)_stretch",
        "mean_score_prediction": 4.54228600114584
      },
      {
        "image_id": "UpheavalDome_stretch",
        "mean_score_prediction": 5.128633365035057
      },
      {
        "image_id": "GeneralSherman_stretch",
        "mean_score_prediction": 4.662207789719105
      },
      {
        "image_id": "CemeteryTree(2)_stretch",
        "mean_score_prediction": 5.199855953454971
      },
      {
        "image_id": "MtRushmoreFlags_stretch",
        "mean_score_prediction": 4.430353961884975
      },
      {
        "image_id": "LasVegasStore_stretch",
        "mean_score_prediction": 4.627242997288704
      },
      {
        "image_id": "Petroglyphs_stretch",
        "mean_score_prediction": 4.503813907504082
      },
      {
        "image_id": "AhwahneeGreatLounge_stretch",
        "mean_score_prediction": 5.396134667098522
      },
      {
        "image_id": "YosemiteFalls_stretch",
        "mean_score_prediction": 3.9489012509584427
      },
      {
        "image_id": "TheNarrows(1)_stretch",
        "mean_score_prediction": 4.9711132645606995
      },
      {
        "image_id": "McKeesPub_stretch",
        "mean_score_prediction": 5.620997443795204
      },
      {
        "image_id": "URChapel(2)_stretch",
        "mean_score_prediction": 5.331093482673168
      },
      {
        "image_id": "PeckLake_stretch",
        "mean_score_prediction": 5.164077430963516
      },
      {
        "image_id": "LabTypewriter_stretch",
        "mean_score_prediction": 5.161290384829044
      },
      {
        "image_id": "LadyBirdRedwoods_stretch",
        "mean_score_prediction": 5.169030211865902
      },
      {
        "image_id": "GoldenGate(2)_stretch",
        "mean_score_prediction": 5.28877680003643
      },
      {
        "image_id": "Frontier_stretch",
        "mean_score_prediction": 4.735200472176075
      },
      {
        "image_id": "HancockSeedField_stretch",
        "mean_score_prediction": 5.052204936742783
      },
      {
        "image_id": "PaulBunyan_stretch",
        "mean_score_prediction": 5.57262384518981
      },
      {
        "image_id": "MirrorLake_stretch",
        "mean_score_prediction": 4.7347743064165115
      },
      {
        "image_id": "DelicateFlowers_stretch",
        "mean_score_prediction": 5.261063940823078
      },
      {
        "image_id": "MasonLake(1)_stretch",
        "mean_score_prediction": 4.497589819133282
      },
      {
        "image_id": "Flamingo_stretch",
        "mean_score_prediction": 4.954344131052494
      },
      {
        "image_id": "DevilsBathtub_stretch",
        "mean_score_prediction": 5.06635469943285
      },
      {
        "image_id": "NorthBubble_stretch",
        "mean_score_prediction": 5.3454990312457085
      },
      {
        "image_id": "GeneralGrant_stretch",
        "mean_score_prediction": 4.868488408625126
      },
      {
        "image_id": "WestBranchAusable(2)_stretch",
        "mean_score_prediction": 5.302547976374626
      },
      {
        "image_id": "HancockKitchenOutside_stretch",
        "mean_score_prediction": 5.715906269848347
      },
      {
        "image_id": "WaffleHouse_stretch",
        "mean_score_prediction": 5.011009402573109
      },
      {
        "image_id": "SouthBranchKingsRiver_stretch",
        "mean_score_prediction": 5.045771427452564
      },
      {
        "image_id": "TunnelView(1)_stretch",
        "mean_score_prediction": 4.449309788644314
      },
      {
        "image_id": "BandonSunset(1)_stretch",
        "mean_score_prediction": 5.169345751404762
      },
      {
        "image_id": "NiagaraFalls_stretch",
        "mean_score_prediction": 4.219521977007389
      },
      {
        "image_id": "HDRMark_stretch",
        "mean_score_prediction": 5.642747048288584
      },
      {
        "image_id": "M3MiddlePond_stretch",
        "mean_score_prediction": 5.263796716928482
      },
      {
        "image_id": "AirBellowsGap_stretch",
        "mean_score_prediction": 5.090854771435261
      },
      {
        "image_id": "SunsetPoint(2)_stretch",
        "mean_score_prediction": 5.180557869374752
      },
      {
        "image_id": "LetchworthTeaTable(1)_stretch",
        "mean_score_prediction": 5.367028549313545
      },
      {
        "image_id": "MtRushmore(2)_stretch",
        "mean_score_prediction": 5.032781712710857
      },
      {
        "image_id": "BenJerrys_stretch",
        "mean_score_prediction": 5.543326653540134
      },
      {
        "image_id": "Exploratorium(1)_stretch",
        "mean_score_prediction": 5.087195977568626
      },
      {
        "image_id": "TupperLake(1)_stretch",
        "mean_score_prediction": 5.088444873690605
      },
      {
        "image_id": "CadesCove_stretch",
        "mean_score_prediction": 5.239006325602531
      },
      {
        "image_id": "BloomingGorse(2)_stretch",
        "mean_score_prediction": 5.380359023809433
      },
      {
        "image_id": "ArtistPalette_stretch",
        "mean_score_prediction": 4.943384945392609
      },
      {
        "image_id": "RoundBarnInside_stretch",
        "mean_score_prediction": 4.833623290061951
      },
      {
        "image_id": "RoundStoneBarn_stretch",
        "mean_score_prediction": 4.995761834084988
      }
    ]
    res_paris = [
      {
        "image_id": "BloomingGorse(2)_lum_",
        "mean_score_prediction": 5.407394118607044
      },
      {
        "image_id": "ElCapitan_lum_",
        "mean_score_prediction": 4.4151280000805855
      },
      {
        "image_id": "MasonLake(2)_lum_",
        "mean_score_prediction": 4.716822788119316
      },
      {
        "image_id": "Zentrum_lum_",
        "mean_score_prediction": 4.0245350524783134
      },
      {
        "image_id": "TupperLake(2)_lum_",
        "mean_score_prediction": 4.715946823358536
      },
      {
        "image_id": "DelicateFlowers_lum_",
        "mean_score_prediction": 4.593161463737488
      },
      {
        "image_id": "Route66Museum_lum_",
        "mean_score_prediction": 5.638169039040804
      },
      {
        "image_id": "DevilsGolfCourse_lum_",
        "mean_score_prediction": 4.802206341177225
      },
      {
        "image_id": "OldFaithfulInn_lum_",
        "mean_score_prediction": 5.135502681136131
      },
      {
        "image_id": "SmokyTunnel_lum_",
        "mean_score_prediction": 4.386697977781296
      },
      {
        "image_id": "ArtistPalette_lum_",
        "mean_score_prediction": 4.970069721341133
      },
      {
        "image_id": "MiddlePond_lum_",
        "mean_score_prediction": 4.736552909016609
      },
      {
        "image_id": "McKeesPub_lum_",
        "mean_score_prediction": 5.508610971271992
      },
      {
        "image_id": "CemeteryTree(2)_lum_",
        "mean_score_prediction": 5.078375704586506
      },
      {
        "image_id": "GoldenGate(2)_lum_",
        "mean_score_prediction": 5.142636828124523
      },
      {
        "image_id": "AmikeusBeaverDamPM2_lum_",
        "mean_score_prediction": 4.620217278599739
      },
      {
        "image_id": "HooverDam_lum_",
        "mean_score_prediction": 4.555589333176613
      },
      {
        "image_id": "CanadianFalls_lum_",
        "mean_score_prediction": 4.87135373800993
      },
      {
        "image_id": "Flamingo_lum_",
        "mean_score_prediction": 4.978277914226055
      },
      {
        "image_id": "URChapel(1)_lum_",
        "mean_score_prediction": 4.726935341954231
      },
      {
        "image_id": "Exploratorium(1)_lum_",
        "mean_score_prediction": 5.176354251801968
      },
      {
        "image_id": "LabTypewriter_lum_",
        "mean_score_prediction": 5.2096941620111465
      },
      {
        "image_id": "HallofFame_lum_",
        "mean_score_prediction": 5.276641935110092
      },
      {
        "image_id": "LetchworthTeaTable(2)_lum_",
        "mean_score_prediction": 4.9885831996798515
      },
      {
        "image_id": "WestBranchAusable(2)_lum_",
        "mean_score_prediction": 4.786043785512447
      },
      {
        "image_id": "BandonSunset(1)_lum_",
        "mean_score_prediction": 5.472332701086998
      },
      {
        "image_id": "MtRushmoreFlags_lum_",
        "mean_score_prediction": 4.789027914404869
      },
      {
        "image_id": "LadyBirdRedwoods_lum_",
        "mean_score_prediction": 4.893787123262882
      },
      {
        "image_id": "DelicateArch_lum_",
        "mean_score_prediction": 4.685097593814135
      },
      {
        "image_id": "MackinacBridge_lum_",
        "mean_score_prediction": 5.659099861979485
      },
      {
        "image_id": "YosemiteFalls_lum_",
        "mean_score_prediction": 4.664725691080093
      },
      {
        "image_id": "TaughannockFalls_lum_",
        "mean_score_prediction": 4.311989150941372
      },
      {
        "image_id": "GeneralGrant_lum_",
        "mean_score_prediction": 4.83008049428463
      },
      {
        "image_id": "CadesCove_lum_",
        "mean_score_prediction": 5.121896669268608
      },
      {
        "image_id": "PeckLake_lum_",
        "mean_score_prediction": 5.146081902086735
      },
      {
        "image_id": "TunnelView(2)_lum_",
        "mean_score_prediction": 4.536968596279621
      },
      {
        "image_id": "LittleRiver_lum_",
        "mean_score_prediction": 4.728951618075371
      },
      {
        "image_id": "SunsetPoint(1)_lum_",
        "mean_score_prediction": 4.458802245557308
      },
      {
        "image_id": "MtRushmore(1)_lum_",
        "mean_score_prediction": 4.655571460723877
      },
      {
        "image_id": "HancockSeedField_lum_",
        "mean_score_prediction": 4.890395380556583
      },
      {
        "image_id": "TheNarrows(1)_lum_",
        "mean_score_prediction": 5.316064462065697
      },
      {
        "image_id": "MammothHotSprings_lum_",
        "mean_score_prediction": 4.349729433655739
      },
      {
        "image_id": "HancockKitchenOutside_lum_",
        "mean_score_prediction": 4.847123399376869
      },
      {
        "image_id": "RITTiger_lum_",
        "mean_score_prediction": 4.203291170299053
      },
      {
        "image_id": "WillySentinel_lum_",
        "mean_score_prediction": 5.429656453430653
      },
      {
        "image_id": "BandonSunset(2)_lum_",
        "mean_score_prediction": 5.4590276926755905
      },
      {
        "image_id": "507_lum_log",
        "mean_score_prediction": 5.198446616530418
      },
      {
        "image_id": "HooverGarage_lum_",
        "mean_score_prediction": 4.92924040555954
      },
      {
        "image_id": "NorthBubble_lum_",
        "mean_score_prediction": 4.763640619814396
      },
      {
        "image_id": "LasVegasStore_lum_",
        "mean_score_prediction": 4.431070066988468
      },
      {
        "image_id": "AmikeusBeaverDamPM1_lum_",
        "mean_score_prediction": 4.684409864246845
      },
      {
        "image_id": "LuxoDoubleChecker_lum_",
        "mean_score_prediction": 5.154505662620068
      },
      {
        "image_id": "FourCornersStorm_lum_",
        "mean_score_prediction": 5.113797158002853
      },
      {
        "image_id": "Petroglyphs_lum_",
        "mean_score_prediction": 4.763365134596825
      },
      {
        "image_id": "BarHarborPresunrise_lum_",
        "mean_score_prediction": 5.040670409798622
      },
      {
        "image_id": "RedwoodSunset_lum_",
        "mean_score_prediction": 4.295915398746729
      },
      {
        "image_id": "BigfootPass_lum_",
        "mean_score_prediction": 4.538838222622871
      },
      {
        "image_id": "DevilsTower_lum_",
        "mean_score_prediction": 4.518814384937286
      },
      {
        "image_id": "RoundBarnInside_lum_",
        "mean_score_prediction": 4.6646294593811035
      },
      {
        "image_id": "SunsetPoint(2)_lum_",
        "mean_score_prediction": 5.261204622685909
      },
      {
        "image_id": "TunnelView(1)_lum_",
        "mean_score_prediction": 4.927998147904873
      },
      {
        "image_id": "DevilsBathtub_lum_",
        "mean_score_prediction": 4.78872287273407
      },
      {
        "image_id": "OCanadaLights_lum_",
        "mean_score_prediction": 5.240374781191349
      },
      {
        "image_id": "AirBellowsGap_lum_",
        "mean_score_prediction": 5.273740723729134
      },
      {
        "image_id": "LabBooth_lum_",
        "mean_score_prediction": 5.636465575546026
      },
      {
        "image_id": "Frontier_lum_",
        "mean_score_prediction": 5.015954934060574
      },
      {
        "image_id": "WallDrug_lum_",
        "mean_score_prediction": 4.583051014691591
      },
      {
        "image_id": "TheNarrows(2)_lum_",
        "mean_score_prediction": 4.272046659141779
      },
      {
        "image_id": "HDRMark_lum_",
        "mean_score_prediction": 5.663585849106312
      },
      {
        "image_id": "M3MiddlePond_lum_",
        "mean_score_prediction": 5.379407852888107
      },
      {
        "image_id": "KingsCanyon_lum_",
        "mean_score_prediction": 5.26406966149807
      },
      {
        "image_id": "TheGrotto_lum_",
        "mean_score_prediction": 4.867266699671745
      },
      {
        "image_id": "SouthBranchKingsRiver_lum_",
        "mean_score_prediction": 4.84047120064497
      },
      {
        "image_id": "MtRushmore(2)_lum_",
        "mean_score_prediction": 5.029243089258671
      },
      {
        "image_id": "WillyDesk_lum_",
        "mean_score_prediction": 5.54546095430851
      },
      {
        "image_id": "MasonLake(1)_lum_",
        "mean_score_prediction": 4.896173253655434
      },
      {
        "image_id": "WaffleHouse_lum_",
        "mean_score_prediction": 5.172905087471008
      },
      {
        "image_id": "BloomingGorse(1)_lum_",
        "mean_score_prediction": 5.055926121771336
      },
      {
        "image_id": "NiagaraFalls_lum_",
        "mean_score_prediction": 4.367504935711622
      },
      {
        "image_id": "PaulBunyan_lum_",
        "mean_score_prediction": 5.169115290045738
      },
      {
        "image_id": "HancockKitchenInside_lum_",
        "mean_score_prediction": 5.5934728011488914
      },
      {
        "image_id": "BarHarborSunrise_lum_",
        "mean_score_prediction": 5.084147229790688
      },
      {
        "image_id": "AhwahneeGreatLounge_lum_",
        "mean_score_prediction": 5.041946694254875
      },
      {
        "image_id": "TupperLake(1)_lum_",
        "mean_score_prediction": 5.026339337229729
      },
      {
        "image_id": "RoundStoneBarn_lum_",
        "mean_score_prediction": 4.9899784699082375
      },
      {
        "image_id": "RoadsEndFireDamage_lum_",
        "mean_score_prediction": 4.8299175426363945
      },
      {
        "image_id": "OtterPoint_lum_",
        "mean_score_prediction": 5.493659183382988
      },
      {
        "image_id": "SequoiaRemains_lum_",
        "mean_score_prediction": 4.974684588611126
      },
      {
        "image_id": "JesseBrownsCabin_lum_",
        "mean_score_prediction": 4.9609265178442
      },
      {
        "image_id": "OCanadaNoLights_lum_",
        "mean_score_prediction": 5.172299027442932
      },
      {
        "image_id": "Exploratorium(2)_lum_",
        "mean_score_prediction": 5.247821480035782
      },
      {
        "image_id": "URChapel(2)_lum_",
        "mean_score_prediction": 5.291870035231113
      },
      {
        "image_id": "MirrorLake_lum_",
        "mean_score_prediction": 4.440368063747883
      },
      {
        "image_id": "HalfDomeSunset_lum_",
        "mean_score_prediction": 4.4867307767271996
      },
      {
        "image_id": "GeneralSherman_lum_",
        "mean_score_prediction": 5.340761609375477
      },
      {
        "image_id": "Peppermill_lum_",
        "mean_score_prediction": 4.940840296447277
      },
      {
        "image_id": "GoldenGate(1)_lum_",
        "mean_score_prediction": 5.535938881337643
      },
      {
        "image_id": "CemeteryTree(1)_lum_",
        "mean_score_prediction": 4.487001173198223
      },
      {
        "image_id": "WestBranchAusable(1)_lum_",
        "mean_score_prediction": 5.1205666065216064
      },
      {
        "image_id": "LabWindow_lum_",
        "mean_score_prediction": 5.436596937477589
      },
      {
        "image_id": "TheNarrows(3)_lum_",
        "mean_score_prediction": 4.79359532892704
      },
      {
        "image_id": "BalancedRock_lum_",
        "mean_score_prediction": 4.320385277271271
      },
      {
        "image_id": "BenJerrys_lum_",
        "mean_score_prediction": 5.391407422721386
      },
      {
        "image_id": "LetchworthTeaTable(1)_lum_",
        "mean_score_prediction": 5.156724721193314
      },
      {
        "image_id": "UpheavalDome_lum_",
        "mean_score_prediction": 4.666123487055302
      }
    ]
    res_durand = [
      {
        "image_id": "HancockSeedField_prega",
        "mean_score_prediction": 4.857281863689423
      },
      {
        "image_id": "MiddlePond_prega",
        "mean_score_prediction": 4.671005934476852
      },
      {
        "image_id": "WillySentinel_prega",
        "mean_score_prediction": 5.487097047269344
      },
      {
        "image_id": "GoldenGate(2)_prega",
        "mean_score_prediction": 5.023489557206631
      },
      {
        "image_id": "TheNarrows(2)_prega",
        "mean_score_prediction": 4.1181652173399925
      },
      {
        "image_id": "TaughannockFalls_prega",
        "mean_score_prediction": 4.557517468929291
      },
      {
        "image_id": "OCanadaNoLights_prega",
        "mean_score_prediction": 5.5277558490633965
      },
      {
        "image_id": "RedwoodSunset_prega",
        "mean_score_prediction": 4.804565519094467
      },
      {
        "image_id": "MackinacBridge_prega",
        "mean_score_prediction": 5.512624517083168
      },
      {
        "image_id": "MasonLake(2)_prega",
        "mean_score_prediction": 5.051805682480335
      },
      {
        "image_id": "WillyDesk_prega",
        "mean_score_prediction": 5.869776843115687
      },
      {
        "image_id": "OCanadaLights_prega",
        "mean_score_prediction": 5.6486014649271965
      },
      {
        "image_id": "FourCornersStorm_prega",
        "mean_score_prediction": 5.717407152056694
      },
      {
        "image_id": "BigfootPass_prega",
        "mean_score_prediction": 4.376370541751385
      },
      {
        "image_id": "Petroglyphs_prega",
        "mean_score_prediction": 5.033145762979984
      },
      {
        "image_id": "GeneralSherman_prega",
        "mean_score_prediction": 5.338321104645729
      },
      {
        "image_id": "OldFaithfulInn_prega",
        "mean_score_prediction": 5.122139245271683
      },
      {
        "image_id": "TupperLake(2)_prega",
        "mean_score_prediction": 5.345233052968979
      },
      {
        "image_id": "URChapel(1)_prega",
        "mean_score_prediction": 4.664691399782896
      },
      {
        "image_id": "SouthBranchKingsRiver_prega",
        "mean_score_prediction": 5.223618507385254
      },
      {
        "image_id": "AmikeusBeaverDamPM2_prega",
        "mean_score_prediction": 4.692936755716801
      },
      {
        "image_id": "HooverDam_prega",
        "mean_score_prediction": 4.960517473518848
      },
      {
        "image_id": "LetchworthTeaTable(2)_prega",
        "mean_score_prediction": 5.306493259966373
      },
      {
        "image_id": "Route66Museum_prega",
        "mean_score_prediction": 5.569097209721804
      },
      {
        "image_id": "WestBranchAusable(1)_prega",
        "mean_score_prediction": 4.7173444256186485
      },
      {
        "image_id": "HalfDomeSunset_prega",
        "mean_score_prediction": 5.161509998142719
      },
      {
        "image_id": "HancockKitchenInside_prega",
        "mean_score_prediction": 5.394907429814339
      },
      {
        "image_id": "LabBooth_prega",
        "mean_score_prediction": 5.732944473624229
      },
      {
        "image_id": "NiagaraFalls_prega",
        "mean_score_prediction": 4.901542648673058
      },
      {
        "image_id": "HancockKitchenOutside_prega",
        "mean_score_prediction": 5.483973488211632
      },
      {
        "image_id": "TunnelView(1)_prega",
        "mean_score_prediction": 4.810581915080547
      },
      {
        "image_id": "DevilsBathtub_prega",
        "mean_score_prediction": 5.463548336178064
      },
      {
        "image_id": "DevilsTower_prega",
        "mean_score_prediction": 4.695876270532608
      },
      {
        "image_id": "AirBellowsGap_prega",
        "mean_score_prediction": 5.350692205131054
      },
      {
        "image_id": "Exploratorium(1)_prega",
        "mean_score_prediction": 5.096042223274708
      },
      {
        "image_id": "DevilsGolfCourse_prega",
        "mean_score_prediction": 3.976993076503277
      },
      {
        "image_id": "SunsetPoint(1)_prega",
        "mean_score_prediction": 4.83707332611084
      },
      {
        "image_id": "BandonSunset(2)_prega",
        "mean_score_prediction": 6.019012480974197
      },
      {
        "image_id": "BarHarborSunrise_prega",
        "mean_score_prediction": 5.313042506575584
      },
      {
        "image_id": "SmokyTunnel_prega",
        "mean_score_prediction": 4.440691232681274
      },
      {
        "image_id": "MtRushmore(2)_prega",
        "mean_score_prediction": 5.161970883607864
      },
      {
        "image_id": "BloomingGorse(2)_prega",
        "mean_score_prediction": 5.281324356794357
      },
      {
        "image_id": "CanadianFalls_prega",
        "mean_score_prediction": 4.932386860251427
      },
      {
        "image_id": "ElCapitan_prega",
        "mean_score_prediction": 4.995063953101635
      },
      {
        "image_id": "CemeteryTree(2)_prega",
        "mean_score_prediction": 4.91847562789917
      },
      {
        "image_id": "ArtistPalette_prega",
        "mean_score_prediction": 4.959597282111645
      },
      {
        "image_id": "SunsetPoint(2)_prega",
        "mean_score_prediction": 5.302737317979336
      },
      {
        "image_id": "RoundStoneBarn_prega",
        "mean_score_prediction": 5.5039903447031975
      },
      {
        "image_id": "BandonSunset(1)_prega",
        "mean_score_prediction": 6.303754758089781
      },
      {
        "image_id": "MammothHotSprings_prega",
        "mean_score_prediction": 4.793526902794838
      },
      {
        "image_id": "RoadsEndFireDamage_prega",
        "mean_score_prediction": 4.917258761823177
      },
      {
        "image_id": "Exploratorium(2)_prega",
        "mean_score_prediction": 5.643790230154991
      },
      {
        "image_id": "LabWindow_prega",
        "mean_score_prediction": 5.256700776517391
      },
      {
        "image_id": "WallDrug_prega",
        "mean_score_prediction": 4.686232298612595
      },
      {
        "image_id": "LuxoDoubleChecker_prega",
        "mean_score_prediction": 5.049651429057121
      },
      {
        "image_id": "BarHarborPresunrise_prega",
        "mean_score_prediction": 5.144376926124096
      },
      {
        "image_id": "CemeteryTree(1)_prega",
        "mean_score_prediction": 4.655736334621906
      },
      {
        "image_id": "AhwahneeGreatLounge_prega",
        "mean_score_prediction": 5.222804434597492
      },
      {
        "image_id": "GeneralGrant_prega",
        "mean_score_prediction": 4.963466838002205
      },
      {
        "image_id": "DelicateArch_prega",
        "mean_score_prediction": 5.0641870200634
      },
      {
        "image_id": "BenJerrys_prega",
        "mean_score_prediction": 5.529011461883783
      },
      {
        "image_id": "BloomingGorse(1)_prega",
        "mean_score_prediction": 5.261529587209225
      },
      {
        "image_id": "DelicateFlowers_prega",
        "mean_score_prediction": 4.590045988559723
      },
      {
        "image_id": "MtRushmore(1)_prega",
        "mean_score_prediction": 4.783702470362186
      },
      {
        "image_id": "RoundBarnInside_prega",
        "mean_score_prediction": 5.080167934298515
      },
      {
        "image_id": "McKeesPub_prega",
        "mean_score_prediction": 5.668652825057507
      },
      {
        "image_id": "CadesCove_prega",
        "mean_score_prediction": 5.264417976140976
      },
      {
        "image_id": "LabTypewriter_prega",
        "mean_score_prediction": 5.153118960559368
      },
      {
        "image_id": "WestBranchAusable(2)_prega",
        "mean_score_prediction": 4.679041802883148
      },
      {
        "image_id": "TheNarrows(3)_prega",
        "mean_score_prediction": 4.980162866413593
      },
      {
        "image_id": "KingsCanyon_prega",
        "mean_score_prediction": 4.9680876061320305
      },
      {
        "image_id": "TunnelView(2)_prega",
        "mean_score_prediction": 5.085823640227318
      },
      {
        "image_id": "LadyBirdRedwoods_prega",
        "mean_score_prediction": 4.908042564988136
      },
      {
        "image_id": "LasVegasStore_prega",
        "mean_score_prediction": 4.563727539032698
      },
      {
        "image_id": "OtterPoint_prega",
        "mean_score_prediction": 5.31946911662817
      },
      {
        "image_id": "UpheavalDome_prega",
        "mean_score_prediction": 4.736116543412209
      },
      {
        "image_id": "PaulBunyan_prega",
        "mean_score_prediction": 5.467858202755451
      },
      {
        "image_id": "Zentrum_prega",
        "mean_score_prediction": 4.456929311156273
      },
      {
        "image_id": "TheGrotto_prega",
        "mean_score_prediction": 4.825108923017979
      },
      {
        "image_id": "TupperLake(1)_prega",
        "mean_score_prediction": 5.247153453528881
      },
      {
        "image_id": "URChapel(2)_prega",
        "mean_score_prediction": 5.316319726407528
      },
      {
        "image_id": "SequoiaRemains_prega",
        "mean_score_prediction": 4.693625256419182
      },
      {
        "image_id": "BalancedRock_prega",
        "mean_score_prediction": 4.828374832868576
      },
      {
        "image_id": "HooverGarage_prega",
        "mean_score_prediction": 4.84820069372654
      },
      {
        "image_id": "LetchworthTeaTable(1)_prega",
        "mean_score_prediction": 5.047427199780941
      },
      {
        "image_id": "MtRushmoreFlags_prega",
        "mean_score_prediction": 5.012848488986492
      },
      {
        "image_id": "Frontier_prega",
        "mean_score_prediction": 4.807708907872438
      },
      {
        "image_id": "RITTiger_prega",
        "mean_score_prediction": 4.681416116654873
      },
      {
        "image_id": "JesseBrownsCabin_prega",
        "mean_score_prediction": 4.857893265783787
      },
      {
        "image_id": "HDRMark_prega",
        "mean_score_prediction": 5.628144923597574
      },
      {
        "image_id": "LittleRiver_prega",
        "mean_score_prediction": 4.911485120654106
      },
      {
        "image_id": "Peppermill_prega",
        "mean_score_prediction": 5.055906221270561
      },
      {
        "image_id": "WaffleHouse_prega",
        "mean_score_prediction": 4.7375897616147995
      },
      {
        "image_id": "GoldenGate(1)_prega",
        "mean_score_prediction": 5.680515207350254
      },
      {
        "image_id": "TheNarrows(1)_prega",
        "mean_score_prediction": 5.315245099365711
      },
      {
        "image_id": "HallofFame_prega",
        "mean_score_prediction": 5.8956397995352745
      },
      {
        "image_id": "Flamingo_prega",
        "mean_score_prediction": 4.757349535822868
      },
      {
        "image_id": "M3MiddlePond_prega",
        "mean_score_prediction": 5.454489912837744
      },
      {
        "image_id": "MasonLake(1)_prega",
        "mean_score_prediction": 4.996395908296108
      },
      {
        "image_id": "PeckLake_prega",
        "mean_score_prediction": 5.558609090745449
      },
      {
        "image_id": "YosemiteFalls_prega",
        "mean_score_prediction": 5.0117697566747665
      },
      {
        "image_id": "507_prega",
        "mean_score_prediction": 5.186304025352001
      },
      {
        "image_id": "AmikeusBeaverDamPM1_prega",
        "mean_score_prediction": 4.752292796969414
      },
      {
        "image_id": "MirrorLake_prega",
        "mean_score_prediction": 4.489426843822002
      },
      {
        "image_id": "NorthBubble_prega",
        "mean_score_prediction": 4.7252189591526985
      }
    ]
    res_fattal = [
      {
        "image_id": "TaughannockFalls_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "UpheavalDome_pregamma_1_fattal",
        "mean_score_prediction": 5.035621993243694
      },
      {
        "image_id": "SunsetPoint(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "BandonSunset(2)_pregamma_1_fattal",
        "mean_score_prediction": 5.849164567887783
      },
      {
        "image_id": "WestBranchAusable(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.105016678571701
      },
      {
        "image_id": "NorthBubble_pregamma_1_fattal",
        "mean_score_prediction": 4.9362708032131195
      },
      {
        "image_id": "HancockKitchenOutside_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "WestBranchAusable(2)_pregamma_1_fattal",
        "mean_score_prediction": 4.9699321165680885
      },
      {
        "image_id": "OCanadaNoLights_pregamma_1_fattal",
        "mean_score_prediction": 5.511814013123512
      },
      {
        "image_id": "Flamingo_pregamma_1_fattal",
        "mean_score_prediction": 5.0423852652311325
      },
      {
        "image_id": "BarHarborSunrise_pregamma_1_fattal",
        "mean_score_prediction": 5.36103243380785
      },
      {
        "image_id": "TheGrotto_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "LabTypewriter_pregamma_1_fattal",
        "mean_score_prediction": 5.224584463983774
      },
      {
        "image_id": "RedwoodSunset_pregamma_1_fattal",
        "mean_score_prediction": 4.693586505949497
      },
      {
        "image_id": "Petroglyphs_pregamma_1_fattal",
        "mean_score_prediction": 4.7487930208444595
      },
      {
        "image_id": "PaulBunyan_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "BandonSunset(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.756908677518368
      },
      {
        "image_id": "HDRMark_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "SunsetPoint(2)_pregamma_1_fattal",
        "mean_score_prediction": 4.9880534410476685
      },
      {
        "image_id": "RITTiger_pregamma_1_fattal",
        "mean_score_prediction": 4.759332068264484
      },
      {
        "image_id": "BigfootPass_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "LasVegasStore_pregamma_1_fattal",
        "mean_score_prediction": 4.869000367820263
      },
      {
        "image_id": "LetchworthTeaTable(2)_pregamma_1_fattal",
        "mean_score_prediction": 4.899363569915295
      },
      {
        "image_id": "DevilsGolfCourse_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "HancockKitchenInside_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "JesseBrownsCabin_pregamma_1_fattal",
        "mean_score_prediction": 4.867237374186516
      },
      {
        "image_id": "AirBellowsGap_pregamma_1_fattal",
        "mean_score_prediction": 5.2859117686748505
      },
      {
        "image_id": "DevilsTower_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "Exploratorium(2)_pregamma_1_fattal",
        "mean_score_prediction": 5.319010145962238
      },
      {
        "image_id": "MtRushmoreFlags_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "OCanadaLights_pregamma_1_fattal",
        "mean_score_prediction": 5.5748913660645485
      },
      {
        "image_id": "McKeesPub_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "ArtistPalette_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "BenJerrys_pregamma_1_fattal",
        "mean_score_prediction": 5.561049833893776
      },
      {
        "image_id": "HalfDomeSunset_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "BloomingGorse(2)_pregamma_1_fattal",
        "mean_score_prediction": 5.296816103160381
      },
      {
        "image_id": "SequoiaRemains_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "MiddlePond_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "KingsCanyon_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "AmikeusBeaverDamPM2_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "DelicateFlowers_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "Exploratorium(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "NiagaraFalls_pregamma_1_fattal",
        "mean_score_prediction": 4.948031045496464
      },
      {
        "image_id": "DevilsBathtub_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "BloomingGorse(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "HooverDam_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "GeneralSherman_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "RoundStoneBarn_pregamma_1_fattal",
        "mean_score_prediction": 5.184201188385487
      },
      {
        "image_id": "ElCapitan_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "PeckLake_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "LetchworthTeaTable(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.378513753414154
      },
      {
        "image_id": "MammothHotSprings_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "LabWindow_pregamma_1_fattal",
        "mean_score_prediction": 5.606766484677792
      },
      {
        "image_id": "AhwahneeGreatLounge_pregamma_1_fattal",
        "mean_score_prediction": 5.229725703597069
      },
      {
        "image_id": "WallDrug_pregamma_1_fattal",
        "mean_score_prediction": 4.637685902416706
      },
      {
        "image_id": "CemeteryTree(2)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "GoldenGate(2)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "TheNarrows(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "Route66Museum_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "RoadsEndFireDamage_pregamma_1_fattal",
        "mean_score_prediction": 4.944935970008373
      },
      {
        "image_id": "MasonLake(2)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "OldFaithfulInn_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "TupperLake(2)_pregamma_1_fattal",
        "mean_score_prediction": 4.85210756957531
      },
      {
        "image_id": "URChapel(2)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321834385395
      },
      {
        "image_id": "Zentrum_pregamma_1_fattal",
        "mean_score_prediction": 4.677542708814144
      },
      {
        "image_id": "MtRushmore(1)_pregamma_1_fattal",
        "mean_score_prediction": 4.925967745482922
      },
      {
        "image_id": "Frontier_pregamma_1_fattal",
        "mean_score_prediction": 5.090229131281376
      },
      {
        "image_id": "HancockSeedField_pregamma_1_fattal",
        "mean_score_prediction": 4.949999183416367
      },
      {
        "image_id": "WillyDesk_pregamma_1_fattal",
        "mean_score_prediction": 5.992689839564264
      },
      {
        "image_id": "YosemiteFalls_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "MackinacBridge_pregamma_1_fattal",
        "mean_score_prediction": 5.673282042145729
      },
      {
        "image_id": "MtRushmore(2)_pregamma_1_fattal",
        "mean_score_prediction": 4.985613025724888
      },
      {
        "image_id": "SouthBranchKingsRiver_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "URChapel(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "LuxoDoubleChecker_pregamma_1_fattal",
        "mean_score_prediction": 5.030950032174587
      },
      {
        "image_id": "FourCornersStorm_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "HallofFame_pregamma_1_fattal",
        "mean_score_prediction": 5.73553791642189
      },
      {
        "image_id": "GeneralGrant_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "BalancedRock_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "LabBooth_pregamma_1_fattal",
        "mean_score_prediction": 5.662463936954737
      },
      {
        "image_id": "BarHarborPresunrise_pregamma_1_fattal",
        "mean_score_prediction": 5.09300871193409
      },
      {
        "image_id": "TheNarrows(2)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "CemeteryTree(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "GoldenGate(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.513229735195637
      },
      {
        "image_id": "507_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "TupperLake(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.255471237003803
      },
      {
        "image_id": "MasonLake(1)_pregamma_1_fattal",
        "mean_score_prediction": 4.9373434111475945
      },
      {
        "image_id": "CadesCove_pregamma_1_fattal",
        "mean_score_prediction": 5.2101441621780396
      },
      {
        "image_id": "LadyBirdRedwoods_pregamma_1_fattal",
        "mean_score_prediction": 4.87543486058712
      },
      {
        "image_id": "RoundBarnInside_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "TunnelView(1)_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "TheNarrows(3)_pregamma_1_fattal",
        "mean_score_prediction": 4.735572729259729
      },
      {
        "image_id": "HooverGarage_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "M3MiddlePond_pregamma_1_fattal",
        "mean_score_prediction": 5.4419418387115
      },
      {
        "image_id": "MirrorLake_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "WaffleHouse_pregamma_1_fattal",
        "mean_score_prediction": 5.091461978852749
      },
      {
        "image_id": "AmikeusBeaverDamPM1_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "CanadianFalls_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "WillySentinel_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "SmokyTunnel_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "Peppermill_pregamma_1_fattal",
        "mean_score_prediction": 5.215767774730921
      },
      {
        "image_id": "DelicateArch_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "LittleRiver_pregamma_1_fattal",
        "mean_score_prediction": 4.76761344820261
      },
      {
        "image_id": "OtterPoint_pregamma_1_fattal",
        "mean_score_prediction": 5.790321454405785
      },
      {
        "image_id": "TunnelView(2)_pregamma_1_fattal",
        "mean_score_prediction": 4.649043120443821
      }
    ]
    res_out_now = [
      {
        "image_id": "BigfootPass_stretch",
        "mean_score_prediction": 4.842783220112324
      },
      {
        "image_id": "BandonSunset(2)_stretch",
        "mean_score_prediction": 5.063626520335674
      },
      {
        "image_id": "RITTiger_stretch",
        "mean_score_prediction": 4.656033284962177
      },
      {
        "image_id": "LabWindow_stretch",
        "mean_score_prediction": 5.516355063766241
      },
      {
        "image_id": "TunnelView(2)_stretch",
        "mean_score_prediction": 4.627959735691547
      },
      {
        "image_id": "TheGrotto_stretch",
        "mean_score_prediction": 4.966006621718407
      },
      {
        "image_id": "BarHarborPresunrise_stretch",
        "mean_score_prediction": 4.91495868563652
      },
      {
        "image_id": "LuxoDoubleChecker_stretch",
        "mean_score_prediction": 4.847224213182926
      },
      {
        "image_id": "OCanadaNoLights_stretch",
        "mean_score_prediction": 5.386347621679306
      },
      {
        "image_id": "SunsetPoint(1)_stretch",
        "mean_score_prediction": 4.369607239961624
      },
      {
        "image_id": "LittleRiver_stretch",
        "mean_score_prediction": 4.7656629383563995
      },
      {
        "image_id": "LetchworthTeaTable(2)_stretch",
        "mean_score_prediction": 5.067666359245777
      },
      {
        "image_id": "JesseBrownsCabin_stretch",
        "mean_score_prediction": 4.589898958802223
      },
      {
        "image_id": "RedwoodSunset_stretch",
        "mean_score_prediction": 4.402441538870335
      },
      {
        "image_id": "BarHarborSunrise_stretch",
        "mean_score_prediction": 5.035418026149273
      },
      {
        "image_id": "BalancedRock_stretch",
        "mean_score_prediction": 4.339556537568569
      },
      {
        "image_id": "WallDrug_stretch",
        "mean_score_prediction": 4.586448296904564
      },
      {
        "image_id": "HancockKitchenInside_stretch",
        "mean_score_prediction": 5.577774345874786
      },
      {
        "image_id": "SmokyTunnel_stretch",
        "mean_score_prediction": 4.502622313797474
      },
      {
        "image_id": "MtRushmore(1)_stretch",
        "mean_score_prediction": 4.7301649153232574
      },
      {
        "image_id": "WillySentinel_stretch",
        "mean_score_prediction": 5.215237677097321
      },
      {
        "image_id": "AmikeusBeaverDamPM2_stretch",
        "mean_score_prediction": 4.6328220665454865
      },
      {
        "image_id": "MackinacBridge_stretch",
        "mean_score_prediction": 5.58113045245409
      },
      {
        "image_id": "OCanadaLights_stretch",
        "mean_score_prediction": 5.39261831343174
      },
      {
        "image_id": "DelicateArch_stretch",
        "mean_score_prediction": 5.0609101466834545
      },
      {
        "image_id": "BloomingGorse(1)_stretch",
        "mean_score_prediction": 5.416340090334415
      },
      {
        "image_id": "RoadsEndFireDamage_stretch",
        "mean_score_prediction": 5.013679251074791
      },
      {
        "image_id": "Zentrum_stretch",
        "mean_score_prediction": 4.698490887880325
      },
      {
        "image_id": "Exploratorium(2)_stretch",
        "mean_score_prediction": 5.552924778312445
      },
      {
        "image_id": "TupperLake(2)_stretch",
        "mean_score_prediction": 4.54874649643898
      },
      {
        "image_id": "507_stretch",
        "mean_score_prediction": 5.130307286977768
      },
      {
        "image_id": "KingsCanyon_stretch",
        "mean_score_prediction": 5.225757107138634
      },
      {
        "image_id": "ElCapitan_stretch",
        "mean_score_prediction": 4.838681310415268
      },
      {
        "image_id": "MiddlePond_stretch",
        "mean_score_prediction": 4.708195373415947
      },
      {
        "image_id": "Route66Museum_stretch",
        "mean_score_prediction": 5.64142632484436
      },
      {
        "image_id": "Peppermill_stretch",
        "mean_score_prediction": 5.166706301271915
      },
      {
        "image_id": "CemeteryTree(1)_stretch",
        "mean_score_prediction": 4.673730336129665
      },
      {
        "image_id": "WillyDesk_stretch",
        "mean_score_prediction": 5.690113541670144
      },
      {
        "image_id": "HooverGarage_stretch",
        "mean_score_prediction": 5.274801045656204
      },
      {
        "image_id": "OtterPoint_stretch",
        "mean_score_prediction": 5.144595317542553
      },
      {
        "image_id": "URChapel(1)_stretch",
        "mean_score_prediction": 4.669674299657345
      },
      {
        "image_id": "AmikeusBeaverDamPM1_stretch",
        "mean_score_prediction": 4.96259880810976
      },
      {
        "image_id": "MammothHotSprings_stretch",
        "mean_score_prediction": 4.6738278940320015
      },
      {
        "image_id": "TaughannockFalls_stretch",
        "mean_score_prediction": 4.877666667103767
      },
      {
        "image_id": "HooverDam_stretch",
        "mean_score_prediction": 4.628577567636967
      },
      {
        "image_id": "TheNarrows(2)_stretch",
        "mean_score_prediction": 4.321621969342232
      },
      {
        "image_id": "OldFaithfulInn_stretch",
        "mean_score_prediction": 5.28136432915926
      },
      {
        "image_id": "LabBooth_stretch",
        "mean_score_prediction": 5.359720770269632
      },
      {
        "image_id": "FourCornersStorm_stretch",
        "mean_score_prediction": 4.773315943777561
      },
      {
        "image_id": "DevilsGolfCourse_stretch",
        "mean_score_prediction": 4.769339367747307
      },
      {
        "image_id": "GoldenGate(1)_stretch",
        "mean_score_prediction": 5.8168815821409225
      },
      {
        "image_id": "DevilsTower_stretch",
        "mean_score_prediction": 4.633798144757748
      },
      {
        "image_id": "SequoiaRemains_stretch",
        "mean_score_prediction": 5.0894976034760475
      },
      {
        "image_id": "CanadianFalls_stretch",
        "mean_score_prediction": 5.075924597680569
      },
      {
        "image_id": "HalfDomeSunset_stretch",
        "mean_score_prediction": 4.759166464209557
      },
      {
        "image_id": "WestBranchAusable(1)_stretch",
        "mean_score_prediction": 5.196432314813137
      },
      {
        "image_id": "TheNarrows(3)_stretch",
        "mean_score_prediction": 4.863989092409611
      },
      {
        "image_id": "HallofFame_stretch",
        "mean_score_prediction": 5.425362206995487
      },
      {
        "image_id": "MasonLake(2)_stretch",
        "mean_score_prediction": 4.682864680886269
      },
      {
        "image_id": "UpheavalDome_stretch",
        "mean_score_prediction": 5.158583045005798
      },
      {
        "image_id": "GeneralSherman_stretch",
        "mean_score_prediction": 4.82992697507143
      },
      {
        "image_id": "CemeteryTree(2)_stretch",
        "mean_score_prediction": 5.232885032892227
      },
      {
        "image_id": "MtRushmoreFlags_stretch",
        "mean_score_prediction": 4.673472568392754
      },
      {
        "image_id": "LasVegasStore_stretch",
        "mean_score_prediction": 4.687221489846706
      },
      {
        "image_id": "Petroglyphs_stretch",
        "mean_score_prediction": 4.456067271530628
      },
      {
        "image_id": "AhwahneeGreatLounge_stretch",
        "mean_score_prediction": 5.3506006225943565
      },
      {
        "image_id": "YosemiteFalls_stretch",
        "mean_score_prediction": 4.201913133263588
      },
      {
        "image_id": "TheNarrows(1)_stretch",
        "mean_score_prediction": 5.043702267110348
      },
      {
        "image_id": "McKeesPub_stretch",
        "mean_score_prediction": 5.5744349509477615
      },
      {
        "image_id": "URChapel(2)_stretch",
        "mean_score_prediction": 5.358393341302872
      },
      {
        "image_id": "PeckLake_stretch",
        "mean_score_prediction": 5.305830217897892
      },
      {
        "image_id": "LabTypewriter_stretch",
        "mean_score_prediction": 4.844313368201256
      },
      {
        "image_id": "LadyBirdRedwoods_stretch",
        "mean_score_prediction": 5.293241731822491
      },
      {
        "image_id": "GoldenGate(2)_stretch",
        "mean_score_prediction": 5.464354641735554
      },
      {
        "image_id": "Frontier_stretch",
        "mean_score_prediction": 4.779062107205391
      },
      {
        "image_id": "HancockSeedField_stretch",
        "mean_score_prediction": 5.037194050848484
      },
      {
        "image_id": "PaulBunyan_stretch",
        "mean_score_prediction": 5.755334362387657
      },
      {
        "image_id": "MirrorLake_stretch",
        "mean_score_prediction": 4.680452153086662
      },
      {
        "image_id": "DelicateFlowers_stretch",
        "mean_score_prediction": 4.9824651554226875
      },
      {
        "image_id": "MasonLake(1)_stretch",
        "mean_score_prediction": 4.716859966516495
      },
      {
        "image_id": "Flamingo_stretch",
        "mean_score_prediction": 4.835286878049374
      },
      {
        "image_id": "DevilsBathtub_stretch",
        "mean_score_prediction": 5.191950805485249
      },
      {
        "image_id": "NorthBubble_stretch",
        "mean_score_prediction": 5.1212586015462875
      },
      {
        "image_id": "GeneralGrant_stretch",
        "mean_score_prediction": 5.066743761301041
      },
      {
        "image_id": "WestBranchAusable(2)_stretch",
        "mean_score_prediction": 5.049321755766869
      },
      {
        "image_id": "HancockKitchenOutside_stretch",
        "mean_score_prediction": 5.621790491044521
      },
      {
        "image_id": "WaffleHouse_stretch",
        "mean_score_prediction": 4.768981050699949
      },
      {
        "image_id": "SouthBranchKingsRiver_stretch",
        "mean_score_prediction": 4.929770745337009
      },
      {
        "image_id": "TunnelView(1)_stretch",
        "mean_score_prediction": 4.827316850423813
      },
      {
        "image_id": "BandonSunset(1)_stretch",
        "mean_score_prediction": 5.39508830010891
      },
      {
        "image_id": "NiagaraFalls_stretch",
        "mean_score_prediction": 4.590092092752457
      },
      {
        "image_id": "HDRMark_stretch",
        "mean_score_prediction": 5.7113246619701385
      },
      {
        "image_id": "M3MiddlePond_stretch",
        "mean_score_prediction": 5.279757723212242
      },
      {
        "image_id": "AirBellowsGap_stretch",
        "mean_score_prediction": 5.03221245855093
      },
      {
        "image_id": "SunsetPoint(2)_stretch",
        "mean_score_prediction": 5.11978055536747
      },
      {
        "image_id": "LetchworthTeaTable(1)_stretch",
        "mean_score_prediction": 5.1515200808644295
      },
      {
        "image_id": "MtRushmore(2)_stretch",
        "mean_score_prediction": 5.118643581867218
      },
      {
        "image_id": "BenJerrys_stretch",
        "mean_score_prediction": 5.600836105644703
      },
      {
        "image_id": "Exploratorium(1)_stretch",
        "mean_score_prediction": 5.139277800917625
      },
      {
        "image_id": "TupperLake(1)_stretch",
        "mean_score_prediction": 5.137056402862072
      },
      {
        "image_id": "CadesCove_stretch",
        "mean_score_prediction": 5.333934783935547
      },
      {
        "image_id": "BloomingGorse(2)_stretch",
        "mean_score_prediction": 5.437535189092159
      },
      {
        "image_id": "ArtistPalette_stretch",
        "mean_score_prediction": 4.991634398698807
      },
      {
        "image_id": "RoundBarnInside_stretch",
        "mean_score_prediction": 4.921816937625408
      },
      {
        "image_id": "RoundStoneBarn_stretch",
        "mean_score_prediction": 5.1462137773633
      }
    ]
    for m in res_out_now:
        total_our += m["mean_score_prediction"]
    print("total_our",total_our / len(res_our))
    for m in res_paris:
        total_paris += m["mean_score_prediction"]
    print("total_paris",total_paris / len(res_paris))
    for m in res_durand:
        total_durand += m["mean_score_prediction"]
    print("total_durand",total_durand / len(res_durand))
    for m in res_fattal:
        total_fattal += m["mean_score_prediction"]
    print("total_fattal",total_fattal / len(res_fattal))
    # res_our_07 = [
    #   {
    #     "image_id": "BigfootPass_stretch",
    #     "mean_score_prediction": 4.869696505367756
    #   },
    #   {
    #     "image_id": "BandonSunset(2)_stretch",
    #     "mean_score_prediction": 4.7798991575837135
    #   },
    #   {
    #     "image_id": "RITTiger_stretch",
    #     "mean_score_prediction": 4.726388718932867
    #   },
    #   {
    #     "image_id": "LabWindow_stretch",
    #     "mean_score_prediction": 5.461375527083874
    #   },
    #   {
    #     "image_id": "TunnelView(2)_stretch",
    #     "mean_score_prediction": 4.7245852798223495
    #   },
    #   {
    #     "image_id": "TheGrotto_stretch",
    #     "mean_score_prediction": 4.903433211147785
    #   },
    #   {
    #     "image_id": "BarHarborPresunrise_stretch",
    #     "mean_score_prediction": 4.828572049736977
    #   },
    #   {
    #     "image_id": "LuxoDoubleChecker_stretch",
    #     "mean_score_prediction": 4.834713488817215
    #   },
    #   {
    #     "image_id": "OCanadaNoLights_stretch",
    #     "mean_score_prediction": 5.381347671151161
    #   },
    #   {
    #     "image_id": "SunsetPoint(1)_stretch",
    #     "mean_score_prediction": 4.411631308495998
    #   },
    #   {
    #     "image_id": "LittleRiver_stretch",
    #     "mean_score_prediction": 4.842702433466911
    #   },
    #   {
    #     "image_id": "LetchworthTeaTable(2)_stretch",
    #     "mean_score_prediction": 5.023076176643372
    #   },
    #   {
    #     "image_id": "JesseBrownsCabin_stretch",
    #     "mean_score_prediction": 4.4881055280566216
    #   },
    #   {
    #     "image_id": "RedwoodSunset_stretch",
    #     "mean_score_prediction": 4.397984221577644
    #   },
    #   {
    #     "image_id": "BarHarborSunrise_stretch",
    #     "mean_score_prediction": 5.084485150873661
    #   },
    #   {
    #     "image_id": "BalancedRock_stretch",
    #     "mean_score_prediction": 4.414441175758839
    #   },
    #   {
    #     "image_id": "WallDrug_stretch",
    #     "mean_score_prediction": 4.63860747218132
    #   },
    #   {
    #     "image_id": "HancockKitchenInside_stretch",
    #     "mean_score_prediction": 5.463131219148636
    #   },
    #   {
    #     "image_id": "SmokyTunnel_stretch",
    #     "mean_score_prediction": 4.68088848143816
    #   },
    #   {
    #     "image_id": "MtRushmore(1)_stretch",
    #     "mean_score_prediction": 4.585983075201511
    #   },
    #   {
    #     "image_id": "WillySentinel_stretch",
    #     "mean_score_prediction": 5.261893153190613
    #   },
    #   {
    #     "image_id": "AmikeusBeaverDamPM2_stretch",
    #     "mean_score_prediction": 4.638513803482056
    #   },
    #   {
    #     "image_id": "MackinacBridge_stretch",
    #     "mean_score_prediction": 5.547711431980133
    #   },
    #   {
    #     "image_id": "OCanadaLights_stretch",
    #     "mean_score_prediction": 5.417403489351273
    #   },
    #   {
    #     "image_id": "DelicateArch_stretch",
    #     "mean_score_prediction": 5.077257018536329
    #   },
    #   {
    #     "image_id": "BloomingGorse(1)_stretch",
    #     "mean_score_prediction": 5.3071786016225815
    #   },
    #   {
    #     "image_id": "RoadsEndFireDamage_stretch",
    #     "mean_score_prediction": 4.859067790210247
    #   },
    #   {
    #     "image_id": "Zentrum_stretch",
    #     "mean_score_prediction": 4.500365376472473
    #   },
    #   {
    #     "image_id": "Exploratorium(2)_stretch",
    #     "mean_score_prediction": 5.611076764762402
    #   },
    #   {
    #     "image_id": "TupperLake(2)_stretch",
    #     "mean_score_prediction": 4.233887050300837
    #   },
    #   {
    #     "image_id": "507_stretch",
    #     "mean_score_prediction": 4.965331494808197
    #   },
    #   {
    #     "image_id": "KingsCanyon_stretch",
    #     "mean_score_prediction": 5.229543782770634
    #   },
    #   {
    #     "image_id": "ElCapitan_stretch",
    #     "mean_score_prediction": 4.770279325544834
    #   },
    #   {
    #     "image_id": "MiddlePond_stretch",
    #     "mean_score_prediction": 4.741365455091
    #   },
    #   {
    #     "image_id": "Route66Museum_stretch",
    #     "mean_score_prediction": 5.650251630693674
    #   },
    #   {
    #     "image_id": "Peppermill_stretch",
    #     "mean_score_prediction": 5.055528752505779
    #   },
    #   {
    #     "image_id": "CemeteryTree(1)_stretch",
    #     "mean_score_prediction": 4.700432613492012
    #   },
    #   {
    #     "image_id": "WillyDesk_stretch",
    #     "mean_score_prediction": 5.640575643628836
    #   },
    #   {
    #     "image_id": "HooverGarage_stretch",
    #     "mean_score_prediction": 5.085663169622421
    #   },
    #   {
    #     "image_id": "OtterPoint_stretch",
    #     "mean_score_prediction": 5.1238997131586075
    #   },
    #   {
    #     "image_id": "URChapel(1)_stretch",
    #     "mean_score_prediction": 4.840491749346256
    #   },
    #   {
    #     "image_id": "AmikeusBeaverDamPM1_stretch",
    #     "mean_score_prediction": 4.852872379124165
    #   },
    #   {
    #     "image_id": "MammothHotSprings_stretch",
    #     "mean_score_prediction": 4.6664155796170235
    #   },
    #   {
    #     "image_id": "TaughannockFalls_stretch",
    #     "mean_score_prediction": 4.6323337852954865
    #   },
    #   {
    #     "image_id": "HooverDam_stretch",
    #     "mean_score_prediction": 4.880640275776386
    #   },
    #   {
    #     "image_id": "TheNarrows(2)_stretch",
    #     "mean_score_prediction": 4.351555306464434
    #   },
    #   {
    #     "image_id": "OldFaithfulInn_stretch",
    #     "mean_score_prediction": 5.189954772591591
    #   },
    #   {
    #     "image_id": "LabBooth_stretch",
    #     "mean_score_prediction": 5.452932350337505
    #   },
    #   {
    #     "image_id": "FourCornersStorm_stretch",
    #     "mean_score_prediction": 4.9054944813251495
    #   },
    #   {
    #     "image_id": "DevilsGolfCourse_stretch",
    #     "mean_score_prediction": 4.834501497447491
    #   },
    #   {
    #     "image_id": "GoldenGate(1)_stretch",
    #     "mean_score_prediction": 5.688965991139412
    #   },
    #   {
    #     "image_id": "DevilsTower_stretch",
    #     "mean_score_prediction": 4.559813447296619
    #   },
    #   {
    #     "image_id": "SequoiaRemains_stretch",
    #     "mean_score_prediction": 5.151796959340572
    #   },
    #   {
    #     "image_id": "CanadianFalls_stretch",
    #     "mean_score_prediction": 5.00480405241251
    #   },
    #   {
    #     "image_id": "HalfDomeSunset_stretch",
    #     "mean_score_prediction": 4.514128789305687
    #   },
    #   {
    #     "image_id": "WestBranchAusable(1)_stretch",
    #     "mean_score_prediction": 5.124950282275677
    #   },
    #   {
    #     "image_id": "TheNarrows(3)_stretch",
    #     "mean_score_prediction": 4.860723435878754
    #   },
    #   {
    #     "image_id": "HallofFame_stretch",
    #     "mean_score_prediction": 5.418196476995945
    #   },
    #   {
    #     "image_id": "MasonLake(2)_stretch",
    #     "mean_score_prediction": 4.7178990095853806
    #   },
    #   {
    #     "image_id": "UpheavalDome_stretch",
    #     "mean_score_prediction": 5.066671408712864
    #   },
    #   {
    #     "image_id": "GeneralSherman_stretch",
    #     "mean_score_prediction": 4.789495900273323
    #   },
    #   {
    #     "image_id": "CemeteryTree(2)_stretch",
    #     "mean_score_prediction": 5.300351180136204
    #   },
    #   {
    #     "image_id": "MtRushmoreFlags_stretch",
    #     "mean_score_prediction": 4.722156248986721
    #   },
    #   {
    #     "image_id": "LasVegasStore_stretch",
    #     "mean_score_prediction": 4.5998477302491665
    #   },
    #   {
    #     "image_id": "Petroglyphs_stretch",
    #     "mean_score_prediction": 4.533873379230499
    #   },
    #   {
    #     "image_id": "AhwahneeGreatLounge_stretch",
    #     "mean_score_prediction": 5.382563263177872
    #   },
    #   {
    #     "image_id": "YosemiteFalls_stretch",
    #     "mean_score_prediction": 3.9234379194676876
    #   },
    #   {
    #     "image_id": "TheNarrows(1)_stretch",
    #     "mean_score_prediction": 4.984784975647926
    #   },
    #   {
    #     "image_id": "McKeesPub_stretch",
    #     "mean_score_prediction": 5.581548377871513
    #   },
    #   {
    #     "image_id": "URChapel(2)_stretch",
    #     "mean_score_prediction": 5.421009682118893
    #   },
    #   {
    #     "image_id": "PeckLake_stretch",
    #     "mean_score_prediction": 5.184118323028088
    #   },
    #   {
    #     "image_id": "LabTypewriter_stretch",
    #     "mean_score_prediction": 5.051527392119169
    #   },
    #   {
    #     "image_id": "LadyBirdRedwoods_stretch",
    #     "mean_score_prediction": 5.253595985472202
    #   },
    #   {
    #     "image_id": "GoldenGate(2)_stretch",
    #     "mean_score_prediction": 5.3688333332538605
    #   },
    #   {
    #     "image_id": "Frontier_stretch",
    #     "mean_score_prediction": 4.852098792791367
    #   },
    #   {
    #     "image_id": "HancockSeedField_stretch",
    #     "mean_score_prediction": 5.209536388516426
    #   },
    #   {
    #     "image_id": "PaulBunyan_stretch",
    #     "mean_score_prediction": 5.760845962911844
    #   },
    #   {
    #     "image_id": "MirrorLake_stretch",
    #     "mean_score_prediction": 4.630173087120056
    #   },
    #   {
    #     "image_id": "DelicateFlowers_stretch",
    #     "mean_score_prediction": 4.936475038528442
    #   },
    #   {
    #     "image_id": "MasonLake(1)_stretch",
    #     "mean_score_prediction": 4.62193188816309
    #   },
    #   {
    #     "image_id": "Flamingo_stretch",
    #     "mean_score_prediction": 4.870118588209152
    #   },
    #   {
    #     "image_id": "DevilsBathtub_stretch",
    #     "mean_score_prediction": 5.136362552642822
    #   },
    #   {
    #     "image_id": "NorthBubble_stretch",
    #     "mean_score_prediction": 5.258288159966469
    #   },
    #   {
    #     "image_id": "GeneralGrant_stretch",
    #     "mean_score_prediction": 4.978314697742462
    #   },
    #   {
    #     "image_id": "WestBranchAusable(2)_stretch",
    #     "mean_score_prediction": 5.084199383854866
    #   },
    #   {
    #     "image_id": "HancockKitchenOutside_stretch",
    #     "mean_score_prediction": 5.616473384201527
    #   },
    #   {
    #     "image_id": "WaffleHouse_stretch",
    #     "mean_score_prediction": 4.982051771134138
    #   },
    #   {
    #     "image_id": "SouthBranchKingsRiver_stretch",
    #     "mean_score_prediction": 4.922020323574543
    #   },
    #   {
    #     "image_id": "TunnelView(1)_stretch",
    #     "mean_score_prediction": 4.715621501207352
    #   },
    #   {
    #     "image_id": "BandonSunset(1)_stretch",
    #     "mean_score_prediction": 5.128907583653927
    #   },
    #   {
    #     "image_id": "NiagaraFalls_stretch",
    #     "mean_score_prediction": 4.265179164707661
    #   },
    #   {
    #     "image_id": "HDRMark_stretch",
    #     "mean_score_prediction": 5.727101027965546
    #   },
    #   {
    #     "image_id": "M3MiddlePond_stretch",
    #     "mean_score_prediction": 5.40487465262413
    #   },
    #   {
    #     "image_id": "AirBellowsGap_stretch",
    #     "mean_score_prediction": 5.126161232590675
    #   },
    #   {
    #     "image_id": "SunsetPoint(2)_stretch",
    #     "mean_score_prediction": 5.374274544417858
    #   },
    #   {
    #     "image_id": "LetchworthTeaTable(1)_stretch",
    #     "mean_score_prediction": 5.258333273231983
    #   },
    #   {
    #     "image_id": "MtRushmore(2)_stretch",
    #     "mean_score_prediction": 5.076945073902607
    #   },
    #   {
    #     "image_id": "BenJerrys_stretch",
    #     "mean_score_prediction": 5.47840379178524
    #   },
    #   {
    #     "image_id": "Exploratorium(1)_stretch",
    #     "mean_score_prediction": 5.159188702702522
    #   },
    #   {
    #     "image_id": "TupperLake(1)_stretch",
    #     "mean_score_prediction": 5.027751944959164
    #   },
    #   {
    #     "image_id": "CadesCove_stretch",
    #     "mean_score_prediction": 5.190940275788307
    #   },
    #   {
    #     "image_id": "BloomingGorse(2)_stretch",
    #     "mean_score_prediction": 5.459664463996887
    #   },
    #   {
    #     "image_id": "ArtistPalette_stretch",
    #     "mean_score_prediction": 5.148826267570257
    #   },
    #   {
    #     "image_id": "RoundBarnInside_stretch",
    #     "mean_score_prediction": 4.935379087924957
    #   },
    #   {
    #     "image_id": "RoundStoneBarn_stretch",
    #     "mean_score_prediction": 5.285031504929066
    #   }
    # ]
    res_our_07 = [
  {
    "image_id": "BigfootPass_stretch",
    "mean_score_prediction": 4.867377080023289
  },
  {
    "image_id": "BandonSunset(2)_stretch",
    "mean_score_prediction": 5.097372680902481
  },
  {
    "image_id": "RITTiger_stretch",
    "mean_score_prediction": 4.560207024216652
  },
  {
    "image_id": "LabWindow_stretch",
    "mean_score_prediction": 5.457372784614563
  },
  {
    "image_id": "TunnelView(2)_stretch",
    "mean_score_prediction": 4.5695726945996284
  },
  {
    "image_id": "TheGrotto_stretch",
    "mean_score_prediction": 4.933337830007076
  },
  {
    "image_id": "BarHarborPresunrise_stretch",
    "mean_score_prediction": 4.833443179726601
  },
  {
    "image_id": "LuxoDoubleChecker_stretch",
    "mean_score_prediction": 4.741984091699123
  },
  {
    "image_id": "OCanadaNoLights_stretch",
    "mean_score_prediction": 5.346851997077465
  },
  {
    "image_id": "SunsetPoint(1)_stretch",
    "mean_score_prediction": 4.335497424006462
  },
  {
    "image_id": "LittleRiver_stretch",
    "mean_score_prediction": 4.835381872951984
  },
  {
    "image_id": "LetchworthTeaTable(2)_stretch",
    "mean_score_prediction": 4.933672599494457
  },
  {
    "image_id": "JesseBrownsCabin_stretch",
    "mean_score_prediction": 4.598006144165993
  },
  {
    "image_id": "RedwoodSunset_stretch",
    "mean_score_prediction": 4.490410700440407
  },
  {
    "image_id": "BarHarborSunrise_stretch",
    "mean_score_prediction": 5.030347414314747
  },
  {
    "image_id": "BalancedRock_stretch",
    "mean_score_prediction": 4.302130304276943
  },
  {
    "image_id": "WallDrug_stretch",
    "mean_score_prediction": 4.621121674776077
  },
  {
    "image_id": "HancockKitchenInside_stretch",
    "mean_score_prediction": 5.5488167107105255
  },
  {
    "image_id": "SmokyTunnel_stretch",
    "mean_score_prediction": 4.519486345350742
  },
  {
    "image_id": "MtRushmore(1)_stretch",
    "mean_score_prediction": 4.665761329233646
  },
  {
    "image_id": "WillySentinel_stretch",
    "mean_score_prediction": 5.1385084465146065
  },
  {
    "image_id": "AmikeusBeaverDamPM2_stretch",
    "mean_score_prediction": 4.713713183999062
  },
  {
    "image_id": "MackinacBridge_stretch",
    "mean_score_prediction": 5.520897693932056
  },
  {
    "image_id": "OCanadaLights_stretch",
    "mean_score_prediction": 5.380183503031731
  },
  {
    "image_id": "DelicateArch_stretch",
    "mean_score_prediction": 4.9885015189647675
  },
  {
    "image_id": "BloomingGorse(1)_stretch",
    "mean_score_prediction": 5.488612696528435
  },
  {
    "image_id": "RoadsEndFireDamage_stretch",
    "mean_score_prediction": 4.918337158858776
  },
  {
    "image_id": "Zentrum_stretch",
    "mean_score_prediction": 4.6199072152376175
  },
  {
    "image_id": "Exploratorium(2)_stretch",
    "mean_score_prediction": 5.569386683404446
  },
  {
    "image_id": "TupperLake(2)_stretch",
    "mean_score_prediction": 4.490179654210806
  },
  {
    "image_id": "507_stretch",
    "mean_score_prediction": 5.027866296470165
  },
  {
    "image_id": "KingsCanyon_stretch",
    "mean_score_prediction": 5.154842518270016
  },
  {
    "image_id": "ElCapitan_stretch",
    "mean_score_prediction": 4.808259002864361
  },
  {
    "image_id": "MiddlePond_stretch",
    "mean_score_prediction": 4.704550474882126
  },
  {
    "image_id": "Route66Museum_stretch",
    "mean_score_prediction": 5.58030442148447
  },
  {
    "image_id": "Peppermill_stretch",
    "mean_score_prediction": 5.135001793503761
  },
  {
    "image_id": "CemeteryTree(1)_stretch",
    "mean_score_prediction": 4.634232118725777
  },
  {
    "image_id": "WillyDesk_stretch",
    "mean_score_prediction": 5.670642728917301
  },
  {
    "image_id": "HooverGarage_stretch",
    "mean_score_prediction": 5.1886866092681885
  },
  {
    "image_id": "OtterPoint_stretch",
    "mean_score_prediction": 5.175363749265671
  },
  {
    "image_id": "URChapel(1)_stretch",
    "mean_score_prediction": 4.756485067307949
  },
  {
    "image_id": "AmikeusBeaverDamPM1_stretch",
    "mean_score_prediction": 4.778787702322006
  },
  {
    "image_id": "MammothHotSprings_stretch",
    "mean_score_prediction": 4.561395063996315
  },
  {
    "image_id": "TaughannockFalls_stretch",
    "mean_score_prediction": 4.687634639441967
  },
  {
    "image_id": "HooverDam_stretch",
    "mean_score_prediction": 4.476631663739681
  },
  {
    "image_id": "TheNarrows(2)_stretch",
    "mean_score_prediction": 4.215166922658682
  },
  {
    "image_id": "OldFaithfulInn_stretch",
    "mean_score_prediction": 5.220120653510094
  },
  {
    "image_id": "LabBooth_stretch",
    "mean_score_prediction": 5.387400388717651
  },
  {
    "image_id": "FourCornersStorm_stretch",
    "mean_score_prediction": 4.7359780594706535
  },
  {
    "image_id": "DevilsGolfCourse_stretch",
    "mean_score_prediction": 4.740666478872299
  },
  {
    "image_id": "GoldenGate(1)_stretch",
    "mean_score_prediction": 5.681077253073454
  },
  {
    "image_id": "DevilsTower_stretch",
    "mean_score_prediction": 4.641382202506065
  },
  {
    "image_id": "SequoiaRemains_stretch",
    "mean_score_prediction": 5.142278902232647
  },
  {
    "image_id": "CanadianFalls_stretch",
    "mean_score_prediction": 5.141233965754509
  },
  {
    "image_id": "HalfDomeSunset_stretch",
    "mean_score_prediction": 4.7379439398646355
  },
  {
    "image_id": "WestBranchAusable(1)_stretch",
    "mean_score_prediction": 5.271523855626583
  },
  {
    "image_id": "TheNarrows(3)_stretch",
    "mean_score_prediction": 4.880181781947613
  },
  {
    "image_id": "HallofFame_stretch",
    "mean_score_prediction": 5.406075403094292
  },
  {
    "image_id": "MasonLake(2)_stretch",
    "mean_score_prediction": 4.604138411581516
  },
  {
    "image_id": "UpheavalDome_stretch",
    "mean_score_prediction": 4.951359905302525
  },
  {
    "image_id": "GeneralSherman_stretch",
    "mean_score_prediction": 4.876208208501339
  },
  {
    "image_id": "CemeteryTree(2)_stretch",
    "mean_score_prediction": 5.217479392886162
  },
  {
    "image_id": "MtRushmoreFlags_stretch",
    "mean_score_prediction": 4.574889153242111
  },
  {
    "image_id": "LasVegasStore_stretch",
    "mean_score_prediction": 4.615850396454334
  },
  {
    "image_id": "Petroglyphs_stretch",
    "mean_score_prediction": 4.387754395604134
  },
  {
    "image_id": "AhwahneeGreatLounge_stretch",
    "mean_score_prediction": 5.301517844200134
  },
  {
    "image_id": "YosemiteFalls_stretch",
    "mean_score_prediction": 3.9959295094013214
  },
  {
    "image_id": "TheNarrows(1)_stretch",
    "mean_score_prediction": 5.092390209436417
  },
  {
    "image_id": "McKeesPub_stretch",
    "mean_score_prediction": 5.607812836766243
  },
  {
    "image_id": "URChapel(2)_stretch",
    "mean_score_prediction": 5.2905223816633224
  },
  {
    "image_id": "PeckLake_stretch",
    "mean_score_prediction": 5.274919182062149
  },
  {
    "image_id": "LabTypewriter_stretch",
    "mean_score_prediction": 4.8563841581344604
  },
  {
    "image_id": "LadyBirdRedwoods_stretch",
    "mean_score_prediction": 5.070159457623959
  },
  {
    "image_id": "GoldenGate(2)_stretch",
    "mean_score_prediction": 5.313610628247261
  },
  {
    "image_id": "Frontier_stretch",
    "mean_score_prediction": 4.614729080349207
  },
  {
    "image_id": "HancockSeedField_stretch",
    "mean_score_prediction": 5.153928428888321
  },
  {
    "image_id": "PaulBunyan_stretch",
    "mean_score_prediction": 5.6842902936041355
  },
  {
    "image_id": "MirrorLake_stretch",
    "mean_score_prediction": 4.641201600432396
  },
  {
    "image_id": "DelicateFlowers_stretch",
    "mean_score_prediction": 4.866788648068905
  },
  {
    "image_id": "MasonLake(1)_stretch",
    "mean_score_prediction": 4.719015792012215
  },
  {
    "image_id": "Flamingo_stretch",
    "mean_score_prediction": 4.875060260295868
  },
  {
    "image_id": "DevilsBathtub_stretch",
    "mean_score_prediction": 5.2287106439471245
  },
  {
    "image_id": "NorthBubble_stretch",
    "mean_score_prediction": 5.1060978174209595
  },
  {
    "image_id": "GeneralGrant_stretch",
    "mean_score_prediction": 4.987100437283516
  },
  {
    "image_id": "WestBranchAusable(2)_stretch",
    "mean_score_prediction": 5.008791744709015
  },
  {
    "image_id": "HancockKitchenOutside_stretch",
    "mean_score_prediction": 5.682315617799759
  },
  {
    "image_id": "WaffleHouse_stretch",
    "mean_score_prediction": 4.762106288224459
  },
  {
    "image_id": "SouthBranchKingsRiver_stretch",
    "mean_score_prediction": 4.937887400388718
  },
  {
    "image_id": "TunnelView(1)_stretch",
    "mean_score_prediction": 4.768234632909298
  },
  {
    "image_id": "BandonSunset(1)_stretch",
    "mean_score_prediction": 5.4902793020009995
  },
  {
    "image_id": "NiagaraFalls_stretch",
    "mean_score_prediction": 4.442140851169825
  },
  {
    "image_id": "HDRMark_stretch",
    "mean_score_prediction": 5.84960824996233
  },
  {
    "image_id": "M3MiddlePond_stretch",
    "mean_score_prediction": 5.305880360305309
  },
  {
    "image_id": "AirBellowsGap_stretch",
    "mean_score_prediction": 4.909702703356743
  },
  {
    "image_id": "SunsetPoint(2)_stretch",
    "mean_score_prediction": 5.149501070380211
  },
  {
    "image_id": "LetchworthTeaTable(1)_stretch",
    "mean_score_prediction": 5.175177402794361
  },
  {
    "image_id": "MtRushmore(2)_stretch",
    "mean_score_prediction": 5.072313718497753
  },
  {
    "image_id": "BenJerrys_stretch",
    "mean_score_prediction": 5.4621873795986176
  },
  {
    "image_id": "Exploratorium(1)_stretch",
    "mean_score_prediction": 5.0280596986413
  },
  {
    "image_id": "TupperLake(1)_stretch",
    "mean_score_prediction": 5.0898493602871895
  },
  {
    "image_id": "CadesCove_stretch",
    "mean_score_prediction": 5.258308954536915
  },
  {
    "image_id": "BloomingGorse(2)_stretch",
    "mean_score_prediction": 5.377756170928478
  },
  {
    "image_id": "ArtistPalette_stretch",
    "mean_score_prediction": 4.961717389523983
  },
  {
    "image_id": "RoundBarnInside_stretch",
    "mean_score_prediction": 4.809913873672485
  },
  {
    "image_id": "RoundStoneBarn_stretch",
    "mean_score_prediction": 5.126234024763107
  }
]
    total_our_07 = 0
    for m in res_our_07:
        total_our_07 += m["mean_score_prediction"]
    print("total_our_07", total_our_07 / len(res_fattal))


if __name__ == '__main__':
    im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/BarHarborSunrise.exr", format="EXR-FI")
    im = im - im.min()
    f = 2048
    im = im / im.max()
    im = hdr_image_util.reshape_image(im, train_reshape=False)
    im = im * ((10)) * 255
    im = im.clip(0, 255).astype('uint8')
    imageio.imwrite("/Users/yaelvinker/Desktop/BarHarborSunrise_e.png", im)
    # plt.imshow(im)
    # plt.show()
    print(im.min())
    print(im.max())
    # im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/data/merged_6G7M_20150328_183029_897.dng", format="RAW-FI").astype('float32')
    # np.save("/Users/yaelvinker/PycharmProjects/lab/utils/dng_file.npy", im)
    # im = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/utils/data/bigFogMap.hdr", format="HDR-FI").astype(
    #     'float32')
    # np.save("/Users/yaelvinker/PycharmProjects/lab/utils/bigFogMap.npy", im)
    # nima_parse()
    # parse_text_file("/Users/yaelvinker/PycharmProjects/lab/tests/f_factors.txt")
    # f_factors = {"synagogue.hdr":5147.48457 / 255,
    #     "2.dng": 77091.89682000001/ 255,
    #     "WillyDesk.exr": 773750342.73066/ 255,
    #     "cathedral.hdr": 6392911.482795/ 255,
    #     "OtterPoint.exr": 160986.84582/ 255,
    #     "bigFogMap.hdr": 773750342.73066/ 255,
    #     "BigfootPass.exr": 287.340375/ 255,
    #     "belgium.hdr": 52180538.8149/ 255,
    #     "1.dng": 458.67105/ 255,
    #     "OCanadaNoLights.exr": 47817.05123999999/ 255,
    #     "507.exr": 2559365.4513/ 255}
    # print(f_factors)
    # output_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/test_factors2.npy")
    # np.save(output_path, f_factors)
    # f_factors = {"synagogue.hdr": 5147.484680616225/ 255,
    # "2.dng": 32116.472509918534/ 255,
    # "WillyDesk.exr": 14312976.807574302/ 255,
    # "cathedral.hdr": 105996.78497167058/ 255,
    # "OtterPoint.exr": 81024.35837634196/ 255,
    # "bigFogMap.hdr": 8280458.96901711/ 255,
    # "BigfootPass.exr": 287.3403826836523/ 255,
    #              "belgium.hdr": 745215.6282670127/ 255,
    #
    #              "1.dng": 458.6709486473698/ 255,
    # "OCanadaNoLights.exr": 47817.05117080077/ 255,
    # "507.exr": 99853.76623482697/ 255}
    # output_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/test_factors1.npy")
    # np.save(output_path, f_factors)
    # print(f_factors)
    # f_test()
    # gather_all_architectures3(arch_dir, output_path, epoch)
    # f_factors = {
    #     "synagogue.hdr":20.186214,
    #     "2.dng":302.321164,
    #     "WillyDesk.exr": 3034315.069532,
    #     "cathedral.hdr": 25070.241109,
    #     "OtterPoint.exr": 631.320964,
    #     "bigFogMap.hdr": 3034315.069532,
    #     "BigfootPass.exr": 1.126825,
    #     "belgium.hdr": 204629.563980,
    #     "1.dng": 1.798710,
    #     "OCanadaNoLights.exr": 187.517848,
    #     "507.exr": 10036.727260
    # }
    # output_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy")
    # np.save(output_path, f_factors)
    # f_factors = {"belgium": 955688.364876,
    #                 "bigFogMap": 259547.191089,
    #                 "cathedral": 70488.191421,
    #                 "synagogue": 4478.121059,
    #                 "507": 98865.115084,
    #                 "BigfootPass": 361.233703,
    #                 "OCanadaNoLights": 48295.221683,
    #                 "WillyDesk": 2584959.105935}
    #
    # output_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/exr_factors.npy")
    # np.save(output_path, f_factors)
    # struct_loss_formula_test2()
    # mu_std_test()
    # new_ssim_test()
    # normalization_test()
    # sub_test()
    # struct_loss_res("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr")
    # epochs = ["600", "430"]
    # im_numbers = ["8", "1", "2"]
    # for epoch in epochs:
    #     for im_number in im_numbers:
    #         gather_all_architectures("/cs/labs/raananf/yael_vinker/03_29/results/pyramid",
    #                 "/cs/labs/raananf/yael_vinker/03_29/summary/pyramid", epoch, "03_17", im_number)
    # gather_all_architectures_accuracy("/cs/labs/raananf/yael_vinker/03_29/results/pyramid",
    #                                   "/cs/labs/raananf/yael_vinker/03_29/summary/pyramid",
    #                                   "590", "03_29")
    # f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/2a.hdr")
    # f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/belgium.hdr")
    # # f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/cathedral.hdr")
    # f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr")
    # f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/WillyDesk.exr")

    # our_custom_ssim_test()
    # patchD()
