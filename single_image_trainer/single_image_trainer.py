import sys
import inspect
import os
import torch.optim as optim
import tranforms as my_transforms
from torch.utils.tensorboard import SummaryWriter
import logging
import imageio
from models import ssim
import numpy as np
import argparse
import torch
from utils import hdr_image_util, data_loader_util
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from data_generator import create_dng_npy_data
import tranforms
import time


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_path", type=str, default="/Users/yaelvinker/PycharmProjects/lab/data/"
                                                       "hdr_data_with_f/train/OtterPoint.npy")
    parser.add_argument("--full_size_im", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=1001)
    parser.add_argument("--steps_to_save", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="/Users/yaelvinker/PycharmProjects/lab/single_image_trainer/single_image_output")

    parser.add_argument("--wind_size", type=int, default=5)
    parser.add_argument('--pyramid_weight_list', help='delimited list input', type=str, default="2,4,6")
    parser.add_argument('--apply_sig_mu_ssim', type=int, default=0)
    parser.add_argument('--struct_method', type=str, default="gamma_ssim")
    parser.add_argument('--std_norm_factor', type=float, default=0.8)

    parser.add_argument('--std_pyramid_weight_list', help='delimited list input', type=str, default="4,1,1")
    parser.add_argument('--intensity_epsilon', type=float, default=0.001)
    # parser.add_argument('--intensity_epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--std_method', type=str, default="gamma_factor_loss_bilateral")
    parser.add_argument('--blf_alpha', type=float, default=0.8,
                        help="if blf_input is log than specify alpha")

    parser.add_argument('--mu_pyramid_weight_list', help='delimited list input', type=str, default="2,2,2")

    opt = parser.parse_args()


    train_size = "_256_"
    if opt.full_size_im:
        train_size = "_full_size_"
    output_dir_name = os.path.splitext(os.path.basename(opt.im_path))[0] + train_size + "struct_" + opt.pyramid_weight_list + \
                      "_cntrst_" + opt.std_method + "_eps" + str(opt.intensity_epsilon) + "_" + opt.std_pyramid_weight_list + "_mu_" + opt.mu_pyramid_weight_list
    opt.run_name = output_dir_name

    output_dir = os.path.join(opt.output_path, output_dir_name)
    mkdir(output_dir)
    mkdir(os.path.join(output_dir, "images"))
    mkdir(os.path.join(output_dir, "loss"))
    mkdir(os.path.join(output_dir, "logs"))

    opt.output_path = output_dir
    opt.images_output_path = os.path.join(output_dir, "images")
    opt.loss_output_path = os.path.join(output_dir, "loss")
    opt.log_dir = os.path.join(output_dir, "logs")

    opt.logger = logging.getLogger(__name__)
    opt.writer = SummaryWriter()

    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    opt.device = device
    opt.pyramid_weight_list = torch.FloatTensor([float(item) for item in opt.pyramid_weight_list.split(',')]).to(device)
    opt.std_pyramid_weight_list = torch.FloatTensor([float(item) for item in opt.std_pyramid_weight_list.split(',')]).to(device)
    opt.mu_pyramid_weight_list = torch.FloatTensor([float(item) for item in opt.mu_pyramid_weight_list.split(',')]).to(
        device)
    return opt


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def imshow(tensor, title=None):
    plt.figure()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    plt.imshow(image[0].numpy(), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(1) # pause a bit so that plots are updated
    plt.show()


def image_loader(opt):
    im_path = opt.im_path
    rgb_img, gray_im_log, f_factor = \
        create_dng_npy_data.hdr_preprocess(im_path,
                                           1, train_reshape=False,
                                           gamma_log=10,
                                           f_factor_path="/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy",
                                           use_new_f=False)
    print("f_factor", f_factor)
    color_im, input_im = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
    gray_original_im = hdr_image_util.to_gray_tensor(color_im)
    gray_original_im_norm = gray_original_im / gray_original_im.max()
    return input_im.unsqueeze(dim=0), color_im.unsqueeze(dim=0), \
           gray_original_im_norm.unsqueeze(dim=0), \
           gray_original_im.unsqueeze(dim=0), f_factor


def npy_loader(path, addFrame, hdrMode, normalization, std_norm_factor, min_stretch,
               max_stretch):
    """
    load npy files that contain the loaded HDR file, and binary image of windows centers.

    """
    data = np.load(path, allow_pickle=True)
    input_im = data[()]["input_image"]
    color_im = data[()]["display_image"]
    if not hdrMode:
        if normalization == "max_normalization":
            input_im = input_im / input_im.max()
        elif normalization == "bugy_max_normalization":
            input_im = input_im / 255
        elif normalization == "stretch":
            input_im = ((input_im - input_im.min()) / input_im.max()) * max_stretch - min_stretch
            input_im = np.clip(input_im, 0, 1)
    if hdrMode:
        gray_original_im = hdr_image_util.to_gray_tensor(color_im)
        gray_original_im_norm = gray_original_im / gray_original_im.max()
        if addFrame:
            input_im = data_loader_util.add_frame_to_im(input_im)
        if "gamma_factor" in data[()].keys():
            return input_im.unsqueeze(dim=0), color_im.unsqueeze(dim=0), \
                   gray_original_im_norm.unsqueeze(dim=0), \
                   gray_original_im.unsqueeze(dim=0), data[()]["gamma_factor"]
        return input_im, color_im, gray_original_im_norm, gray_original_im, 0
    return input_im, color_im, input_im, input_im, 0


def get_input(opt):
    if opt.full_size_im:
        input_im, color_im, gray_original_norm, gray_original, gamma_factor = image_loader(opt)
    else:
        input_im, color_im, gray_original_norm, gray_original, gamma_factor = \
            npy_loader(opt.im_path, addFrame=False, hdrMode=True, normalization="",
                       std_norm_factor=0, min_stretch=0, max_stretch=0)
    # imshow(input_im, title='input_im')
    # imshow(gray_original_norm, title='gray_original_norm')
    # imshow(gray_original, title='gray_original')

    gamma_factor = torch.tensor([gamma_factor])
    blf_input = ssim.get_blf_log_input(gray_original_norm, gamma_factor.to(opt.device, torch.float), alpha=opt.blf_alpha)
    r_weights = ssim.get_radiometric_weights(blf_input, opt.wind_size, 0.05,
                                                     1, "log")
    return input_im.to(opt.device, torch.float), color_im.to(opt.device, torch.float), \
           gray_original_norm.to(opt.device, torch.float), gray_original.to(opt.device, torch.float), \
           gamma_factor.to(opt.device, torch.float), r_weights


def get_losses(opt):
    struct_loss = ssim.OUR_CUSTOM_SSIM_PYRAMID(window_size=opt.wind_size,
                                               pyramid_weight_list=opt.pyramid_weight_list,
                                               pyramid_pow=False, use_c3=False,
                                               apply_sig_mu_ssim=opt.apply_sig_mu_ssim,
                                               struct_method=opt.struct_method,
                                               std_norm_factor=opt.std_norm_factor,
                                               crop_input=False)
    contrast_loss = ssim.IntensityLoss(opt.intensity_epsilon, opt.std_pyramid_weight_list, opt.alpha,
                                             opt.std_method, opt.wind_size, crop_input=False)
    mu_loss = ssim.MuLoss(opt.mu_pyramid_weight_list, opt.wind_size, crop_input=False)
    return struct_loss, contrast_loss, mu_loss


def get_input_optimizer(input_img):
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.01)
    return optimizer


def save_im(opt, run, loss, input_im, color_im):
    im = input_im.clone().detach()[0].permute(1, 2, 0).numpy()
    im = np.clip(im, 0, 1)
    im_color = color_im[0].clone().permute(1, 2, 0).detach().cpu().numpy()
    im_color = hdr_image_util.back_to_color(im_color, im)
    im_name = "iter[%d]loss[%.4f]color_norm.png" % (run[0], loss)
    opt.writer.add_image("gen_image", im_color)
    opt.writer.close()

    imageio.imwrite(os.path.join(opt.images_output_path, im_name), im_color)
    im_name = "iter[%d]loss[%.4f]color_clip.png" % (run[0], loss)
    im_color2 = (np.clip(im_color,0,1) * 255).astype('uint8')
    imageio.imwrite(os.path.join(opt.images_output_path, im_name), im_color2)
    im = (im * 255).astype('uint8')
    im_name = "iter[%d]loss[%.4f].png" % (run[0], loss)
    imageio.imwrite(os.path.join(opt.images_output_path, im_name), im)


def save_loss(opt, run, struct_err_lst, contrast_err_lst, cmprs_err_lst):
    iters_n = len(struct_err_lst)
    plt.figure()
    plt.plot(range(iters_n), struct_err_lst, '-r', label='struct')
    plt.plot(range(iters_n), contrast_err_lst, '-b', label='contrast')
    plt.plot(range(iters_n), cmprs_err_lst, '-g', label='cmprs')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    title = "iter[%d]\n%s" % (run[0], opt.run_name)
    plt.title(title, fontSize=8)
    plt.savefig(os.path.join(opt.loss_output_path, str(run[0]) + ".png"))  # should before show method
    plt.close()


def train_on_single_image():
    print(torch.__version__)
    opt = get_opt()
    input_im, color_im, gray_original_norm, \
        gray_original, gamma_factor, r_weights = get_input(opt)
    input_im_source = input_im.clone()
    optimizer = get_input_optimizer(input_im)
    struct_loss, contrast_loss, mu_loss = get_losses(opt)

    print('Optimizing..')
    run = [0]
    struct_err_lst = []
    contrast_err_lst = []
    mu_err_lst = []
    while run[0] <= opt.num_steps:
        def closure():
            start = time.time()
            # correct the values of updated input image
            input_im.data.clamp_(0, 1)

            optimizer.zero_grad()

            struct_err = struct_loss(input_im, gray_original_norm.detach(),
                                     input_im_source.detach(), r_weights)
            contrast_err = contrast_loss(input_im, input_im_source.detach(),
                                         gray_original_norm.detach(), r_weights,
                                         gamma_factor, gray_original.detach())
            mu_err = mu_loss(input_im, gray_original.detach(), input_im_source.detach(), r_weights)

            struct_err_lst.append(struct_err)
            contrast_err_lst.append(contrast_err)
            mu_err_lst.append(mu_err)

            loss = struct_err + contrast_err + mu_err
            loss.backward()

            if run[0] % opt.steps_to_save == 0:
                cur_time = time.time() - start
                print("run {}:".format(run))
                print("struct_err[%.4f] contrast_err[%.4f] mu_err[%.4f] sec[%.4f]" %
                      (struct_err.item(), contrast_err.item(), mu_err.item(), cur_time))
                print()
                save_im(opt, run, struct_err + contrast_err + mu_err, input_im, color_im)
                save_loss(opt, run, struct_err_lst, contrast_err_lst, mu_err_lst)
            run[0] += 1
            return struct_err + contrast_err + mu_err

        optimizer.step(closure)

        # a last correction...
    input_im.data.clamp_(0, 1)
    # imshow(input_im.detach(), title='Output Image')
    return input_im
    # img2 = torch.FloatTensor(input_im.size()).uniform_(0, 1)
    # img1 = Variable(hdr_im, requires_grad=False)
    # img2 = Variable(img2, requires_grad=True)
    #
    # optimizer = optim.Adam([img2], lr=0.01)
    #
    # i = 0
    # img_list = []
    # img_list2 = []
    # ssim_value = ssim_loss(img1, img2).item()
    # print("Initial ssim:", ssim_value)
    # while ssim_value > 0.0002:
    #     if i % 1000 == 0:
    #         im = img2[0].permute(1, 2, 0).detach().cpu().numpy()
    #         sub = im - hdr_im_numpy
    #         print("sub", sub.shape, sub.max(), sub.min())
    #         data = {'sub': sub}
    #         np.save("/Users/yaelvinker/PycharmProjects/lab/utils/sub.npy", data)
    #         imageio.imwrite("/Users/yaelvinker/PycharmProjects/lab/utils/sub.png", im)
    #         print(ssim_value)
    #         print("im", im.shape, im.max(), im.min())
    #         im = (im - np.min(im)) / (np.max(im) - np.min(im))
    #         im = (im * 255).astype('uint8')
    #         img_list2.append(im)
    #         imageio.imwrite("/Users/yaelvinker/PycharmProjects/lab/utils/res.png", im)
    #
    #     optimizer.zero_grad()
    #     ssim_out = ssim_loss(img1, img2)
    #     ssim_value = ssim_out.item()
    #     ssim_out.backward()
    #     optimizer.step()
    #     i += 1
    # imageio.mimsave("/Users/yaelvinker/PycharmProjects/lab/utils/res.gif", img_list2, fps=5)
train_on_single_image()