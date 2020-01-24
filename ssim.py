from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import params


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


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
    img1 = factor * (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = factor * (img2 - img2.min()) / (img2.max() - img2.min())
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


def print_tensor_details(im, title):
    print(title)
    print("shape : ", im.shape)
    print("max : ", im.max(), "  min : ", im.min(), "mean : ", im.mean())
    print("type : ", im.dtype)
    # print("unique values : ", np.unique(im.numpy()).shape[0])
    print()


def our_custom_ssim(img1, img2, window, window_size, channel, mse_loss):
    window = window / window.sum()
    factor = float(2 ** 8 - 1.)
    img1 = factor * (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = factor * (img2 - img2.min()) / (img2.max() - img2.min())
    # print_tensor_details(img1, "im1")
    # print_tensor_details(img2, "im2")

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # print_tensor_details(mu1, "mu1")
    # print_tensor_details(mu2, "mu2")

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    # print_tensor_details(mu1_sq, "mu1_sq")
    # print_tensor_details(mu2_sq, "mu2_sq")

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq

    mu1_mu2 = mu1 * mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    a = sigma1_sq <= 0
    b = sigma2_sq <= 0
    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon, 0.5)
    std2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)) + params.epsilon, 0.5)
    # std1 = (sigma1_sq + sigma1_sq.min().abs() + params.epsilon).pow(0.5)
    # std2 = (sigma2_sq + sigma1_sq.min().abs() + params.epsilon).pow(0.5)

    # norm_im1 = torch.div(torch.add(img1, -mu1), std1)
    # norm_im2 = torch.div(torch.add(img2, -mu2), std2)
    C2 = 10.
    s_map = (sigma12 + C2) / (std1 * std2 + C2)
    # s_map1 = torch.pow((((img1 - mu1 + C2) / std1+ C2) - ((img2 - mu2+ C2) / std2+ C2)), 2)
    # print("mse_loss", mse_loss(((img1 - mu1) / std1) , ((img2 - mu2) / std2)))
    # s_map1 = 2 * F.conv2d(((img1 - mu1) * (img2 - mu2)), window, padding=window_size // 2, groups=channel) / (std1 * std2)

    # print((norm_im1 - norm_im2).pow(2).mean())
    return s_map.mean()


class OUR_CUSTOM_SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(OUR_CUSTOM_SSIM, self).__init__()
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

        return our_custom_ssim(img1, img2, window, self.window_size, channel, self.mse_loss)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    print("_ssim")
    print(_ssim(img1, img2, window, window_size, channel, size_average))
    return _ssim_from_tmqi(img1, img2, window, window_size, channel, size_average)

#
# if __name__ == '__main__':
#     img1 = Variable(torch.ones(4, 3, 256, 256))
#     img2 = Variable(torch.ones(4, 3, 256, 256))
#     print(ssim(img1, img2))
#     ssim_loss = SSIM(window_size=11)
#
#     print(ssim_loss(img1, img2))
