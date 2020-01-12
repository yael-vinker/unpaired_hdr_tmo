import numpy as np
import pathlib
import imageio
import os
import torch
import params
import ssim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import tranforms as transforms_
import torchvision.transforms as transforms
import torchvision.transforms.functional as F1
import torch.nn.functional as F
import hdr_image_utils
import gan_trainer_utils
from old_files import HdrImageFolder

import torchvision.datasets as dset
import gan_trainer
from torch.autograd import Variable
import numpy as np
from math import exp
import ssim
import TMQI


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
    result_dir_pref, input_dim, apply_g_ssim = gan_trainer.parse_arguments()
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

        im_log_norm = im_log / np.max(im_log) # 0-1
        hdr_image_utils.print_image_details(im_log_norm, "im_log_norm")

        std_im = (im_log_norm - 0.5) / 0.5 # -1 1
        hdr_image_utils.print_image_details(std_im, "std_im")

        im_n = (std_im - np.min(std_im)) / (np.max(std_im) - np.min(std_im)) # 0-1 (im_log_norm)
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
    plt.subplot(2,1,1)
    plt.imshow(im1, cmap='gray')
    plt.subplot(2,1,2)
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
    import TMQI
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
    import unet_multi_filters.Unet as squre_unet
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
                              con_operator=params.original_unet, filters=32, bilinear=False, network=params.unet_network, dilation=0)

    new_torus = squre_unet.UNet(1, 1, 0, depth=3, layer_factor=layer_factor,
                              con_operator=params.original_unet, filters=32, bilinear=False, network=params.torus_network, dilation=2)
    # print(unet_conv)
    # summary(unet_conv, (1, 256, 256), device="cpu")

    print(new_torus)
    summary(new_torus, (1, 256, 256), device="cpu")

def to_gray(im):
    return np.dot(im[...,:3], [0.299, 0.587, 0.114]).astype('float32')

def to_0_1_range(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def to_0_1_range_tensor(im):
    return (im - im.min()) / (im.max() - im.min())

def ssim_test():
    im_tone_mapped = imageio.imread("/Users/yaelvinker/PycharmProjects/lab/local_log_1000_unet_original_unet_depth_2/model_results/1/1_epoch_1_rgb.png")
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

if __name__ == '__main__':
    our_custom_ssim_test()
    ssim_test()
    a = torch.tensor([1,0,-1,-3])
    b = a<=0
    print(b.sum())
    print(b.count(True))
    print(len([a<=0]))
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


