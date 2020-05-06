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

if __name__ == '__main__':
    # f_test()
    # gather_all_architectures3(arch_dir, output_path, epoch)
    f_factors = {
        "synagogue.hdr":20.186214,
        "2.dng":302.321164,
        "WillyDesk.exr": 3034315.069532,
        "cathedral.hdr": 25070.241109,
        "OtterPoint.exr": 631.320964,
        "bigFogMap.hdr": 3034315.069532,
        "BigfootPass.exr": 1.126825,
        "belgium.hdr": 204629.563980,
        "1.dng": 1.798710,
        "OCanadaNoLights.exr": 187.517848,
        "507.exr": 10036.727260
    }
    output_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy")
    np.save(output_path, f_factors)
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
