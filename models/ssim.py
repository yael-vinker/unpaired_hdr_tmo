from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import params
import torch


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # print(window.shape)
    # print(window)
    window = torch.ones((1, channel, window_size, window_size))
    return window


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class TMQI_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(TMQI_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_from_tmqi(img1, img2, window, self.window_size, channel, self.size_average)


class OUR_CUSTOM_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, use_c3=False, apply_sig_mu_ssim=False,
                 struct_method="our_custom_ssim", std_norm_factor=1):
        super(OUR_CUSTOM_SSIM, self).__init__()
        self.struct_methods = {
            "reg_ssim": our_custom_ssim,
            "struct_loss": structural_loss,
            "struct_loss_a": structural_loss_a,
            "struct_loss_b": structural_loss_b
        }
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.mse_loss = torch.nn.MSELoss()
        self.use_c3 = use_c3
        self.apply_sig_mu_ssim = apply_sig_mu_ssim
        self.struct_method = struct_method
        self.struct_loss = self.struct_methods[struct_method]
        self.std_norm_factor = std_norm_factor

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        if self.struct_method == "reg_ssim":
            return self.our_custom_ssim(img1, img2, window, self.window_size, channel, self.mse_loss, self.use_c3, self.apply_sig_mu_ssim)
        return self.struct_loss(self.window_size, img1, img2, self.std_norm_factor)


class OUR_SIGMA_SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(OUR_SIGMA_SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return our_custom_sigma_loss(img1, img2, window, self.window_size, channel, self.mse_loss)


class OUR_CUSTOM_SSIM_PYRAMID(torch.nn.Module):
    def __init__(self, pyramid_weight_list, window_size=11, pyramid_pow=False, use_c3=False,
                 apply_sig_mu_ssim=False, struct_method="our_custom_ssim", std_norm_factor=1):
        super(OUR_CUSTOM_SSIM_PYRAMID, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.mse_loss = torch.nn.MSELoss()
        self.pyramid_weight_list = pyramid_weight_list
        self.pyramid_pow = pyramid_pow
        self.use_c3 = use_c3
        self.apply_sig_mu_ssim = apply_sig_mu_ssim
        self.struct_methods = {
            "reg_ssim": our_custom_ssim,
            "struct_loss": structural_loss,
            "struct_loss_a": structural_loss_a,
            "struct_loss_b": structural_loss_b
        }
        self.struct_method = struct_method
        self.struct_loss = self.struct_methods[struct_method]
        self.std_norm_factor = std_norm_factor

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        # if self.pyramid_pow:
        #     return our_custom_ssim_pyramid_pow(img1, img2, window, self.window_size, channel,
        #                                        self.pyramid_weight_list, self.mse_loss, self.use_c3)
        if self.struct_method == "reg_ssim":
            return our_custom_ssim_pyramid(img1, img2, window, self.window_size, channel, self.pyramid_weight_list,
                                           self.mse_loss, self.use_c3, self.apply_sig_mu_ssim)
        return our_custom_ssim_pyramid_new(self.window_size, img1, img2, self.std_norm_factor, self.pyramid_weight_list,
                                        self.struct_loss)




def our_custom_ssim(img1, img2, window, window_size, channel, mse_loss, use_c3, apply_sig_mu_ssim):
    window = window / window.sum()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq

    mu1_mu2 = mu1 * mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon, 0.5)
    std2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)) + params.epsilon, 0.5)
    if use_c3:
        C3 = 0.03 ** 2 / 2
        s_map = (sigma12 + C3) / (std1 * std2 + C3)
    else:
        s_map = sigma12 / (std1 * std2 + 0.0001)
    ones = torch.ones(s_map.shape, requires_grad=True)
    if torch.cuda.is_available():
        ones = ones.cuda()
    if apply_sig_mu_ssim:
        s_map = (std2 / mu2) * s_map
        return (ones - s_map).mean()
    return (ones - s_map).mean()


def structural_loss(wind_size, a, b, std_norm_factor):
    """
    compute L2[(a - mu_a) / std_a, (b - mu_b) / std_b)] where a is G(gamma(hdr)

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
    print(windows_b.shape)
    print("\na_original : mu", torch.mean(windows[0, 0, 5, 5]), " res std", torch.std(windows[0, 0, 5, 5]))

    mu1 = get_mu(a, window, wind_size)
    mu2 = get_mu(b, window, wind_size)

    std1 = get_std(windows, mu1, wind_size)
    std2 = get_std(windows_b, mu2, wind_size)
    windows = windows - mu1
    windows = windows / (std1 + params.epsilon)
    print("\na : res mu", torch.mean(windows[0, 0, 5, 5]), " res std", torch.std(windows[0, 0, 5, 5]))

    windows_b = windows_b - mu2
    windows_b = windows_b / (std2 + params.epsilon)
    print("\nb : res mu", torch.mean(windows_b[0, 0, 5, 5])," res std", torch.std(windows_b[0, 0, 5, 5]))
    windows, windows_b = fix_shape(windows), fix_shape(windows_b)
    print(windows.shape, windows_b.shape)
    return mse_loss(windows, windows_b)


def structural_loss_a(wind_size, a, b, std_norm_factor):
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
    print("\na_original : mu", torch.mean(windows[0, 0, 5, 5]), " res std", torch.std(windows[0, 0, 5, 5]))
    mu1 = get_mu(a, window, wind_size)
    mu2 = get_mu(b, window, wind_size)
    std2 = get_std(windows_b, mu2, wind_size)
    windows = windows - mu1
    print("\na : res mu", torch.mean(windows[0, 0, 5, 5]), " res std", torch.std(windows[0, 0, 5, 5]))
    windows_b = windows_b - mu2
    windows_b = std_norm_factor * windows_b * (mu1 / (std2 + params.epsilon))
    print("\nb : res mu", torch.mean(windows_b[0, 0, 5, 5])," res std", torch.std(windows_b[0, 0, 5, 5]))
    windows, windows_b = fix_shape(windows), fix_shape(windows_b)
    return mse_loss(windows, windows_b)


def structural_loss_b(wind_size, a, b, std_norm_factor):
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
    print("\na_original : mu", torch.mean(windows[0, 0, 5, 5]), " res std", torch.std(windows[0, 0, 5, 5]))
    mu1 = get_mu(a, window, wind_size)
    mu2 = get_mu(b, window, wind_size)
    windows = windows - mu1
    print("\na : res mu", torch.mean(windows[0, 0, 5, 5]), " res std", torch.std(windows[0, 0, 5, 5]))
    windows_b = windows_b - mu2
    windows_b = std_norm_factor * windows_b * (mu1 / (mu2 + params.epsilon))
    print("\nb : res mu", torch.mean(windows_b[0, 0, 5, 5])," res std", torch.std(windows_b[0, 0, 5, 5]))
    windows, windows_b = fix_shape(windows), fix_shape(windows_b)
    return mse_loss(windows, windows_b)


def our_custom_sigma_loss(img1, img2, window, window_size, channel, mse_loss):
    window = window / window.sum()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq

    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon, 0.5)
    std2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)) + params.epsilon, 0.5)
    std2_normalise = std2 / (torch.max(mu2, torch.zeros_like(sigma2_sq) + params.epsilon))
    std2_normalise = torch.pow(std2_normalise, 0.8)
    # std2 = torch.pow(std2, 0.85)
    return mse_loss(std1, std2_normalise)


def our_custom_ssim_pyramid(img1, img2, window, window_size, channel, pyramid_weight_list,
                            mse_loss, use_c3, apply_sig_mu_ssim):
    ssim_loss_list = []
    for i in range(len(pyramid_weight_list)):
        ssim_loss_list.append(pyramid_weight_list[i] * our_custom_ssim(img1, img2, window, window_size,
                                                                       channel, mse_loss, use_c3, apply_sig_mu_ssim))
        img1 = F.interpolate(img1, scale_factor=0.5, mode='bicubic', align_corners=False)
        img2 = F.interpolate(img2, scale_factor=0.5, mode='bicubic', align_corners=False)
    return torch.sum(torch.stack(ssim_loss_list))


def our_custom_ssim_pyramid_new(window_size, img1, img2, std_norm_factor, pyramid_weight_list,
                            struct_loss):
    ssim_loss_list = []
    for i in range(len(pyramid_weight_list)):
        ssim_loss_list.append(pyramid_weight_list[i] * struct_loss(window_size, img1, img2, std_norm_factor))
        img1 = F.interpolate(img1, scale_factor=0.5, mode='bicubic', align_corners=False)
        img2 = F.interpolate(img2, scale_factor=0.5, mode='bicubic', align_corners=False)
    return torch.sum(torch.stack(ssim_loss_list))


def our_custom_ssim_pyramid_pow(img1, img2, window, window_size, channel, pyramid_weight_list, mse_loss, use_c3):
    ssim_loss_list = []
    for i in range(len(pyramid_weight_list)):
        ssim_loss_list.append(our_custom_ssim(img1, img2, window, window_size, channel, mse_loss, use_c3))
        img1 = F.interpolate(img1, scale_factor=0.5, mode='bicubic', align_corners=False)
        img2 = F.interpolate(img2, scale_factor=0.5, mode='bicubic', align_corners=False)
    ssim_loss_list = torch.stack(ssim_loss_list)
    pow2 = ssim_loss_list ** pyramid_weight_list
    output = torch.prod(pow2)
    return output


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_from_tmqi(img1, img2, window, window_size, channel, size_average=True):
    factor = float(2 ** 8 - 1.)
    window = window / window.sum()
    # if self.original:
    img1 = factor * (img1 - img1.min()) / (img1.max() - img1.min() + params.epsilon)
    img2 = factor * (img2 - img2.min()) / (img2.max() - img2.min() + params.epsilon)
    C1 = 0.01
    C2 = 10.
    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq

    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2

    sigma1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon, 0.5)
    sigma2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)) + params.epsilon, 0.5)

    CSF = 100.0 * 2.6 * (0.0192 + 0.114 * 16) * np.exp(- (0.114 * 16) ** 1.1)
    u_hdr = 128 / (1.4 * CSF)
    sig_hdr = u_hdr / 3.
    sigma1p = torch.distributions.normal.Normal(loc=u_hdr, scale=sig_hdr).cdf(sigma1)
    u_ldr = u_hdr
    sig_ldr = u_ldr / 3.

    sigma2p = torch.distributions.normal.Normal(loc=u_ldr, scale=sig_ldr).cdf(sigma2)
    s_map = ((2 * sigma1p * sigma2p + C1) / (sigma1p ** 2 + sigma2p ** 2 + C1)
             * ((sigma12 + C2) / (sigma1 * sigma2 + C2)))
    if size_average:
        return s_map.mean()
    else:
        return s_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    print("_ssim")
    print(_ssim(img1, img2, window, window_size, channel, size_average))
    return _ssim_from_tmqi(img1, img2, window, window_size, channel, size_average)


def fix_shape(windows):
    windows = windows.permute(0, 4, 2, 3, 1)
    windows = windows.squeeze(dim=4)
    return windows


def get_window_and_set_device(wind_size, a, b):
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


def get_im_as_windows(a, wind_size):
    windows = a.unfold(dimension=2, size=5, step=1)
    windows = windows.unfold(dimension=3, size=5, step=1)
    windows = windows.reshape(windows.shape[0], windows.shape[1],
                              windows.shape[2], windows.shape[3],
                              wind_size * wind_size)
    return windows


def get_mu(x, window, wind_size):
    mu = F.conv2d(x, window, padding=0, groups=1)
    mu = mu.unsqueeze(dim=4)
    mu = mu.expand(-1, -1, -1, -1, wind_size * wind_size)
    return mu


def get_std(windows, mu1, wind_size):
    x_minus_mu = windows - mu1
    x_minus_mu_sq = x_minus_mu.pow(2)
    x_minus_mu_sq = x_minus_mu_sq.squeeze(dim=1)
    x_minus_mu_sq = x_minus_mu_sq.permute(0, 3, 1, 2)
    wind_a = torch.ones((1, wind_size * wind_size, 1, 1))
    wind_a = wind_a / wind_a.sum()
    mu_x_minus_mu_sq = F.conv2d(x_minus_mu_sq, wind_a, padding=0, groups=1)
    std1 = torch.pow(torch.max(mu_x_minus_mu_sq, torch.zeros_like(mu_x_minus_mu_sq)) + params.epsilon, 0.5)
    std1 = std1.expand(-1, wind_size * wind_size, -1, -1)
    std1 = std1.permute(0, 2, 3, 1)
    std1 = std1.unsqueeze(dim=1)
    return std1