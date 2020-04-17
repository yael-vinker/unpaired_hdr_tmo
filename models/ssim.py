from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import params, data_loader_util
import torch
from torch import nn

# =======================================
# ============= Classes ===============
# =======================================
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
            "reg_ssim": our_custom_ssim
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
        return our_custom_ssim_pyramid(img1, img2, window, self.window_size, channel, self.pyramid_weight_list,
                                           self.mse_loss, self.use_c3, self.apply_sig_mu_ssim)


class IntensityLoss(torch.nn.Module):
    def __init__(self, epsilon, pyramid_weight_list):
        super(IntensityLoss, self).__init__()
        self.epsilon = epsilon
        self.window = create_window(5, 1)
        self.pyramid_weight_list = pyramid_weight_list
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, img1, img2, hdr_input):
        ssim_loss_list = []
        for i in range(len(self.pyramid_weight_list)):
            ssim_loss_list.append(self.pyramid_weight_list[i] * std_loss(self.window, img1, self.epsilon))
            img1 = F.interpolate(img1, scale_factor=0.5, mode='bicubic', align_corners=False)
        return torch.sum(torch.stack(ssim_loss_list))


class MuLoss(torch.nn.Module):
    def __init__(self, pyramid_weight_list):
        super(MuLoss, self).__init__()
        self.window = create_window(5, 1)
        self.pyramid_weight_list = pyramid_weight_list
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, img1, img2, hdr_input):
        hdr_input = data_loader_util.crop_input_hdr_batch(hdr_input)
        mu_loss_list = []
        for i in range(len(self.pyramid_weight_list)):
            mu_loss_list.append(self.pyramid_weight_list[i] * mu_loss(self.window, img1, hdr_input, self.mse_loss))
            img1 = F.interpolate(img1, scale_factor=0.5, mode='bicubic', align_corners=False)
            hdr_input = F.interpolate(hdr_input, scale_factor=0.5, mode='bicubic', align_corners=False)
        return torch.sum(torch.stack(mu_loss_list))


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


# =======================================
# ========== Loss Functions =============
# =======================================
def our_custom_ssim(img1, img2, window, window_size, channel, mse_loss, use_c3, apply_sig_mu_ssim):
    window = window / window.sum()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq

    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon ** 2, 0.5)
    std2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)) + params.epsilon ** 2, 0.5)

    mu1_ = mu1.unsqueeze(dim=4).expand(-1, -1, -1, -1, window_size * window_size)
    mu2_ = mu2.unsqueeze(dim=4).expand(-1, -1, -1, -1, window_size * window_size)

    std1_ = std1.unsqueeze(dim=4).expand(-1, -1, -1, -1, window_size * window_size)
    std2_ = std2.unsqueeze(dim=4).expand(-1, -1, -1, -1, window_size * window_size)
    img1_ = get_im_as_windows(img1, window_size)
    img2_ = get_im_as_windows(img2, window_size)
    img1_ = (img1_ - mu1_)
    img1_ = img1_ / (std1_ + params.epsilon)
    img2_ = (img2_ - mu2_)
    img2_ = img2_ / (std2_ + params.epsilon)
    return mse_loss(img1_, img2_)


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


def std_loss(window, img1, epsilon):
    ones = torch.ones(img1.shape)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        ones = ones.cuda(img1.get_device())
    window = window.type_as(img1)
    ones = ones.type_as(img1)
    window = window / window.sum()
    mu1 = F.conv2d(img1, window, padding=5 // 2, groups=1)
    mu1_sq = mu1.pow(2)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=5 // 2, groups=1) - mu1_sq
    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + 1e-10, 0.5)
    res = ones - (std1[0, 0] / (std1[0, 0] + epsilon))
    return res.mean()


def mu_loss(window, fake, hdr_input, mse_loss):
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
    window = window.type_as(fake)
    window = window / window.sum()
    mu1 = F.conv2d(fake, window, padding=5 // 2, groups=1)
    mu2 = F.conv2d(hdr_input, window, padding=5 // 2, groups=1)
    return mse_loss(mu1, mu2.detach())


def sig_loss(window, img1, epsilon, img2):
    ones = torch.ones(img1.shape)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        ones = ones.cuda(img1.get_device())
    window = window.type_as(img1)
    ones = ones.type_as(img1)
    window = window / window.sum()
    mu1 = F.conv2d(img1, window, padding=5 // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=5 // 2, groups=1)
    mu1_mu2 = mu1 * mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=5 // 2, groups=1) - mu1_mu2
    res = ones - (sigma12[0, 0] / (sigma12[0, 0] + epsilon))
    return res.mean()


# =======================================
# ========= Helper Functions ============
# =======================================
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
    m = nn.ZeroPad2d(wind_size // 2)
    a = m(a)
    windows = a.unfold(dimension=2, size=wind_size, step=1)
    windows = windows.unfold(dimension=3, size=wind_size, step=1)
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