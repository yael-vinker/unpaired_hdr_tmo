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
    print(our_ssim_loss(b1, b1))
    print(our_ssim_loss(b2, b1))
    print("\nour_sigma_loss")
    print(our_custom_sigma_loss(b1, b1))
    b2 = b2 / b2.max()
    print(our_custom_sigma_loss(b2, b1))
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
    t_m_hdr = t_m_hdr / t_m_hdr.max()
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

def f_gamma_test(im_path):
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    if np.min(rgb_img) < 0:
        rgb_img = rgb_img + np.min(rgb_img)
    gray_im = hdr_image_util.to_gray(rgb_img)
    gray_im_temp = hdr_image_util.reshape_im(gray_im, 256, 256)
    brightness_factor = hdr_image_util.get_brightness_factor(gray_im_temp) * 255
    print(brightness_factor)
    gray_im_gamma = (gray_im / np.max(gray_im)) ** (1 / (1 + 2*np.log10(brightness_factor)))
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

if __name__ == '__main__':
    # f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/belgium.hdr")
    # f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/cathedral.hdr")
    f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/synagogue.hdr")
    f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data/WillyDesk.exr")
    f_gamma_test("/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/2a.hdr")
    # our_custom_ssim_test()
    # patchD()
