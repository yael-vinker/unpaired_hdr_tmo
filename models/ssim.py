from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import params, data_loader_util
import torch
from torch import nn
from utils import printer


# =======================================
# ============= Classes ===============
# =======================================
class OUR_CUSTOM_SSIM_PYRAMID(torch.nn.Module):
    def __init__(self, pyramid_weight_list, window_size=5, pyramid_pow=False, use_c3=False,
                 apply_sig_mu_ssim=False, struct_method="our_custom_ssim", std_norm_factor=1, crop_input=True):
        super(OUR_CUSTOM_SSIM_PYRAMID, self).__init__()
        self.window_size = window_size
        self.crop_input = crop_input
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.gaussian_kernel = get_gaussian_kernel(window_size=window_size, channel=1)
        self.laplacian_kernel = get_laplacian_kernel(kernel_size=window_size)
        self.mse_loss = torch.nn.MSELoss()
        self.pyramid_weight_list = pyramid_weight_list
        self.pyramid_pow = pyramid_pow
        self.use_c3 = use_c3
        self.apply_sig_mu_ssim = apply_sig_mu_ssim
        self.struct_methods = {
            "hdr_ssim": our_custom_ssim,
            "gamma_ssim": our_custom_ssim,
            "laplace_ssim": struct_loss_laplac,
            "gamma_ssim_bilateral": our_custom_ssim_bilateral
        }
        self.struct_method = struct_method
        if struct_method not in ["hdr_ssim", "gamma_ssim", "laplace_ssim", "gamma_ssim_bilateral"]:
            assert 0, "Unsupported struct_method"
        self.struct_loss = self.struct_methods[struct_method]
        self.std_norm_factor = std_norm_factor

    def forward(self, fake, hdr_input_original_gray_norm, hdr_input, r_weights=None):
        (_, channel, _, _) = fake.size()
        if channel == self.channel and self.window.data.type() == fake.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if fake.is_cuda:
                window = window.cuda(fake.get_device())
            window = window.type_as(fake)

            self.window = window
            self.channel = channel
        if self.struct_method == "gamma_ssim":
            if self.crop_input:
                hdr_input = data_loader_util.crop_input_hdr_batch(hdr_input)
            return our_custom_ssim_pyramid(fake, hdr_input, window, self.window_size, channel,
                                           self.pyramid_weight_list,
                                           self.mse_loss, self.use_c3,
                                           self.apply_sig_mu_ssim, r_weights)
        elif self.struct_method == "hdr_ssim":
            return our_custom_ssim_pyramid(fake, hdr_input_original_gray_norm, window, self.window_size,
                                           channel, self.pyramid_weight_list,
                                           self.mse_loss, self.use_c3, self.apply_sig_mu_ssim, r_weights)
        elif self.struct_method == "gamma_ssim_bilateral":
            return our_custom_ssim_pyramid_bilateral(fake, hdr_input_original_gray_norm, window, self.window_size,
                                       channel, self.pyramid_weight_list,
                                       self.mse_loss, self.use_c3, self.apply_sig_mu_ssim, r_weights)
        elif self.struct_method == "laplace_ssim":
            hdr_input = data_loader_util.crop_input_hdr_batch(hdr_input)
            return our_custom_ssim_pyramid_laplace(fake, hdr_input, self.gaussian_kernel, self.laplacian_kernel,
                                           self.pyramid_weight_list, self.mse_loss, self.window_size)


class IntensityLoss(torch.nn.Module):
    def __init__(self, epsilon, pyramid_weight_list, alpha=1, std_method="std", wind_size=5, crop_input=True):
        super(IntensityLoss, self).__init__()
        self.epsilon = epsilon
        self.crop_input = crop_input
        self.wind_size = wind_size
        self.window = create_window(wind_size, 1)
        # self.window = get_gaussian_kernel2(wind_size, 1)
        self.pyramid_weight_list = pyramid_weight_list
        self.mse_loss = torch.nn.MSELoss()
        self.std_methods = {
            "std": std_loss,
            "std_bilateral": std_loss_bilateral,
            "gamma_factor_loss": gamma_factor_loss,
            "gamma_factor_loss_bilateral": gamma_factor_loss_bilateral,
            "std_gamma_loss": std_gamma_loss,
            "bilateral_origin": loss_bilateral_original_im,
            "bilateral_origin_blf": loss_bilateral_original_im_blf,
            "blf_wind_off": std_loss_blf_off,
            "blf_wind_off_log": std_loss_blf_off,
            "stdloss_blfmu_res": gamma_factor_blfmu_res,
            "stdloss_blfmu_std": gamma_factor_blfmu_std,
            "stdloss_stdmu_std": gamma_factor_stdmu_std,
            "stdloss_stdmu_res": gamma_factor_stdmu_res,
            "no_blf_stdmu_std": gamma_factor_stdmu_std_no_blf,
            "no_blf_stdmu2_std": gamma_factor_stdmu_std_no_blf2,
            "no_blf_stdmu_res": gamma_factor_stdmu_res_no_blf
        }
        self.std_loss = self.std_methods[std_method]
        self.alpha = alpha
        self.mse_loss = torch.nn.MSELoss()
        self.std_method = std_method
        self.apply_on_log = False
        if std_method == "blf_wind_off_log":
            self.apply_on_log = True

    def forward(self, fake, hdr_input, hdr_original_im, r_weights=None, f_factors=None, hdr_original_gray=None):
        if f_factors is not None and f_factors.sum() == 0:
            f_factors = None
        if self.crop_input:
            hdr_input = data_loader_util.crop_input_hdr_batch(hdr_input)
        cur_weights = None
        ssim_loss_list = []
        for i in range(len(self.pyramid_weight_list)):
            if r_weights:
                cur_weights = r_weights[i]
            ssim_loss_list.append(self.pyramid_weight_list[i] *
                                  self.std_loss(self.window, fake, hdr_input, hdr_original_im,
                                                self.epsilon, self.mse_loss,
                                                self.alpha, cur_weights, f_factors, self.wind_size,
                                                hdr_original_gray, self.apply_on_log))
            fake = F.interpolate(fake, scale_factor=0.5, mode='bicubic', align_corners=False)
            hdr_input = F.interpolate(hdr_input, scale_factor=0.5, mode='bicubic', align_corners=False)
            hdr_original_im = F.interpolate(hdr_original_im, scale_factor=0.5, mode='bicubic', align_corners=False)
            # if apply_clamp:
            fake = fake.clamp(0, 1)
            hdr_input = hdr_input.clamp(0, 1)
            hdr_original_im = hdr_original_im.clamp(0, 1)
        return torch.sum(torch.stack(ssim_loss_list))


class MuLoss(torch.nn.Module):
    def __init__(self, pyramid_weight_list, wind_size, crop_input=True):
        super(MuLoss, self).__init__()
        self.wind_size = wind_size
        self.window = create_window(wind_size, 1)
        # self.window = get_gaussian_kernel2(5, 1)
        self.pyramid_weight_list = pyramid_weight_list
        self.mse_loss = torch.nn.MSELoss()
        self.mu_loss_method = mu_loss
        self.crop_input = crop_input

    def forward(self, fake, img2, hdr_input, r_weights):
        if self.crop_input:
            hdr_input = data_loader_util.crop_input_hdr_batch(hdr_input)
        if r_weights:
            self.mu_loss_method = mu_loss_bilateral
        cur_weights = None
        mu_loss_list = []
        for i in range(len(self.pyramid_weight_list)):
            if r_weights:
                cur_weights = r_weights[i]
            mu_loss_list.append(self.pyramid_weight_list[i] * self.mu_loss_method(self.window, fake, hdr_input,
                                                                      self.mse_loss, cur_weights, self.wind_size))
            fake = F.interpolate(fake, scale_factor=0.5, mode='bicubic', align_corners=False)
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
def our_custom_ssim(img1, img2, window, window_size, channel, mse_loss, use_c3, apply_sig_mu_ssim, cur_weights):
    window = window / window.sum()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq

    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon2, 0.5)
    std2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)) + params.epsilon2, 0.5)

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


def our_custom_ssim_bilateral(img1, img2, window, wind_size, channel, mse_loss, use_c3, apply_sig_mu_ssim, r_weights):
    window = window / window.sum()

    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    img1_windows = get_im_as_windows(img1, wind_size)
    img1_windows = img1_windows.squeeze(dim=1)
    img1_windows = img1_windows.permute(0, 3, 1, 2)
    mu1 = torch.sum(img1_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    img1_bf_mu_w = mu1.expand(-1, wind_size * wind_size, -1, -1)
    img1_minus_mean_sq = (img1_windows - img1_bf_mu_w) * (img1_windows - img1_bf_mu_w)
    sigma_img1_sq = torch.sum(img1_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std1 = torch.pow(torch.max(sigma_img1_sq, torch.zeros_like(sigma_img1_sq)) + params.epsilon, 0.5)

    img2_windows = get_im_as_windows(img2, wind_size)
    img2_windows = img2_windows.squeeze(dim=1)
    img2_windows = img2_windows.permute(0, 3, 1, 2)
    mu2 = torch.sum(img2_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    img2_bf_mu_w = mu2.expand(-1, wind_size * wind_size, -1, -1)
    img2_minus_mean_sq = (img2_windows - img2_bf_mu_w) * (img2_windows - img2_bf_mu_w)
    sigma_img2_sq = torch.sum(img2_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std2 = torch.pow(torch.max(sigma_img2_sq, torch.zeros_like(sigma_img2_sq)) + params.epsilon, 0.5)

    mu1 = mu1.unsqueeze(dim=4).expand(-1, -1, -1, -1, wind_size * wind_size)
    mu2 = mu2.unsqueeze(dim=4).expand(-1, -1, -1, -1, wind_size * wind_size)

    std1 = std1.unsqueeze(dim=4).expand(-1, -1, -1, -1, wind_size * wind_size)
    std2 = std2.unsqueeze(dim=4).expand(-1, -1, -1, -1, wind_size * wind_size)

    img1 = get_im_as_windows(img1, wind_size)
    img2 = get_im_as_windows(img2, wind_size)
    img1 = (img1 - mu1)
    img1 = img1 / (std1 + params.epsilon2)
    img2 = (img2 - mu2)
    img2 = (img2) / (std2 + params.epsilon2)
    return mse_loss(img1, img2)


def struct_loss_laplac(gaussian_kernel, laplacian_kernel, fake, gamma_input, mse_loss, wind_size):
    b, c, h, w = fake.shape

    if fake.is_cuda:
        gaussian_kernel = gaussian_kernel.cuda(fake.get_device())
        laplacian_kernel = laplacian_kernel.cuda(fake.get_device())

    gaussian_kernel = gaussian_kernel.type_as(fake)
    laplacian_kernel = laplacian_kernel.type_as(fake)
    laplacian_kernel = laplacian_kernel.repeat(c, 1, 1, 1)

    gamma_input_gaussian = F.conv2d(gamma_input, gaussian_kernel, padding=wind_size // 2, groups=1)
    laplacian_res_gamma = F.conv2d(gamma_input_gaussian, laplacian_kernel, padding=wind_size // 2, stride=1, groups=c)
    laplacian_res_gamma[laplacian_res_gamma < 0] = 0

    laplacian_res_fake = F.conv2d(fake, gaussian_kernel, padding=wind_size // 2, groups=1)
    laplacian_res_fake = F.conv2d(laplacian_res_fake, laplacian_kernel, padding=wind_size // 2, stride=1, groups=c)
    laplacian_res_fake[laplacian_res_fake < 0] = 0
    return mse_loss(laplacian_res_fake, laplacian_res_gamma)


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
                            mse_loss, use_c3, apply_sig_mu_ssim, r_weights):
    cur_weights = None
    ssim_loss_list = []
    for i in range(len(pyramid_weight_list)):
        if r_weights is not None:
            cur_weights = r_weights[i]
        ssim_loss_list.append(pyramid_weight_list[i] * our_custom_ssim(img1, img2, window, window_size,
                                                                       channel, mse_loss, use_c3,
                                                                       apply_sig_mu_ssim, cur_weights))

        img1 = F.interpolate(img1, scale_factor=0.5, mode='bicubic', align_corners=False)
        img2 = F.interpolate(img2, scale_factor=0.5, mode='bicubic', align_corners=False)
    return torch.sum(torch.stack(ssim_loss_list))


def our_custom_ssim_pyramid_bilateral(img1, img2, window, window_size, channel, pyramid_weight_list,
                            mse_loss, use_c3, apply_sig_mu_ssim, r_weights):
    cur_weights = None
    ssim_loss_list = []
    for i in range(len(pyramid_weight_list)):
        if r_weights is not None:
            cur_weights = r_weights[i]
        ssim_loss_list.append(pyramid_weight_list[i] * our_custom_ssim_bilateral(img1, img2, window, window_size,
                                                                       channel, mse_loss, use_c3,
                                                                       apply_sig_mu_ssim, cur_weights))

        img1 = F.interpolate(img1, scale_factor=0.5, mode='bicubic', align_corners=False)
        img2 = F.interpolate(img2, scale_factor=0.5, mode='bicubic', align_corners=False)
    return torch.sum(torch.stack(ssim_loss_list))


def our_custom_ssim_pyramid_laplace(fake, gamma_input, gaussian_kernel, laplacian_kernel,
                                    pyramid_weight_list, mse_loss, wind_size):
    ssim_loss_list = []
    for i in range(len(pyramid_weight_list)):
        ssim_loss_list.append(pyramid_weight_list[i] * struct_loss_laplac(gaussian_kernel, laplacian_kernel,
                                                                          fake, gamma_input, mse_loss, wind_size))

        fake = F.interpolate(fake, scale_factor=0.5, mode='bicubic', align_corners=False)
        gamma_input = F.interpolate(gamma_input, scale_factor=0.5, mode='bicubic', align_corners=False)
    return torch.sum(torch.stack(ssim_loss_list))


def std_loss(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None, f_factors=None):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    ones = ones.type_as(fake)
    window = window / window.sum()
    mu1 = F.conv2d(fake, window, padding=5 // 2, groups=1)
    mu1_sq = mu1.pow(2)
    sigma1_sq = F.conv2d(fake * fake, window, padding=5 // 2, groups=1) - mu1_sq
    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + 1e-10, 0.5)
    res = ones - (std1 / (std1 + epsilon))
    return res.mean()


def gamma_factor_loss(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5, hdr_original_gray=None, apply_on_log=False):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window = window / window.sum()
    mu_fake = F.conv2d(fake, window, padding=wind_size // 2, groups=1)
    mu_fake_sq = mu_fake.pow(2)
    sigma_fake_sq = F.conv2d(fake * fake, window, padding=wind_size // 2, groups=1) - mu_fake_sq
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    mu_gamma = F.conv2d(gamma_hdr, window, padding=wind_size // 2, groups=1)
    mu_gamma_sq = mu_gamma.pow(2)
    sigma_gamma_sq = F.conv2d(gamma_hdr * gamma_hdr, window, padding=wind_size // 2, groups=1) - mu_gamma_sq
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    mu_hdr_im_pow_gamma = F.conv2d(hdr_im_pow_gamma, window, padding=wind_size // 2, groups=1)
    std_objective = (alpha / (f_factors + epsilon)) * std_gamma * (mu_hdr_im_pow_gamma)
    if hdr_original_gray is not None:
        hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
        hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
        std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective.detach()))
    return res.mean()


def std_gamma_loss(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None):
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
    window = window.type_as(fake)
    window = window / window.sum()
    mu_fake = F.conv2d(fake, window, padding=5 // 2, groups=1)
    mu_fake_sq = mu_fake.pow(2)
    sigma_fake_sq = F.conv2d(fake * fake, window, padding=5 // 2, groups=1) - mu_fake_sq
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + 1e-10, 0.5)

    mu_gamma = F.conv2d(gamma_hdr, window, padding=5 // 2, groups=1)
    mu_gamma_sq = mu_gamma.pow(2)
    sigma_gamma_sq = F.conv2d(gamma_hdr * gamma_hdr, window, padding=5 // 2, groups=1) - mu_gamma_sq
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + 1e-10, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    mu_hdr_im_pow_gamma = F.conv2d(hdr_im_pow_gamma, window, padding=5 // 2, groups=1)
    std_objective = alpha * std_gamma
    return mse_loss(std_fake, std_objective.detach())


def gamma_factor_loss_bilateral(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5, hdr_original_gray=None, log=False):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    # hdr_original_windows = get_im_as_windows(hdr_original_im, wind_size)
    # hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    # hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    # hdr_original_bf_mu = torch.sum(hdr_original_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    # hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    # hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (
    #         hdr_original_windows - hdr_original_bf_mu)
    # sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * weights_map, axis=1).unsqueeze(
    #     dim=1) / weights_map_sum
    # std_hdr_original = torch.pow(
    #     torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)

    # hdr_original_bf_mu = torch.sum(hdr_original_windows * distance_gause, axis=1).unsqueeze(dim=1) / torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    # hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    # hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (
    #         hdr_original_windows - hdr_original_bf_mu)
    # sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * distance_gause, axis=1).unsqueeze(
    #     dim=1) / torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    # std_hdr_original2 = torch.pow(
    #     torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)
    # w1 = (r_weights.sum(dim=1) - 1)

    std_objective = (alpha / (f_factors + epsilon)) * std_gamma * (hdr_pow_mu)

    # if hdr_original_gray is not None:
    #     hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
    #     hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
    #     std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective.detach()))
    return res.mean()


def gamma_factor_stdmu_std(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5, hdr_original_gray=None):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    hdr_original_windows = get_im_as_windows(hdr_original_im, wind_size)
    hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    hdr_original_bf_mu = torch.sum(hdr_original_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (
            hdr_original_windows - hdr_original_bf_mu)
    sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * weights_map, axis=1).unsqueeze(
        dim=1) / weights_map_sum
    std_hdr_original = torch.pow(
        torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)

    alpha = alpha / (256 / hdr_original_im.shape[2])
    std_objective = (alpha / f_factors) * std_gamma * (hdr_pow_mu + epsilon)
    std_objective2 = std_objective * (std_hdr_original.mean() / std_hdr_original)
    std_objective2 = std_objective / (256 / hdr_original_im.shape[2])
    # if hdr_original_gray is not None:
    #     hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
    #     hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
    #     std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective2.detach()))
    return res.mean()


def gamma_factor_stdmu_res(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                           f_factors=None, wind_size=5, hdr_original_gray=None):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    # hdr_original_windows = get_im_as_windows(hdr_original_im, wind_size)
    # hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    # hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    # hdr_original_bf_mu = torch.sum(hdr_original_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    # hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    # hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (
    #         hdr_original_windows - hdr_original_bf_mu)
    # sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * weights_map, axis=1).unsqueeze(
    #     dim=1) / weights_map_sum
    # std_hdr_original = torch.pow(
    #     torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)

    std_objective = (alpha / f_factors) * std_gamma * (hdr_pow_mu + epsilon)

    if hdr_original_gray is not None:
        hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
        hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
        std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective.detach()))
    # res *= (std_hdr_original.mean() / std_hdr_original)
    return res.mean()


def gamma_factor_stdmu_std_no_blf(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5, hdr_original_gray=None):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    hdr_original_windows = get_im_as_windows(hdr_original_im, wind_size)
    hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    hdr_original_bf_mu = torch.sum(hdr_original_windows * distance_gause, axis=1).unsqueeze(dim=1) / \
                         torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (
            hdr_original_windows - hdr_original_bf_mu)
    sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * distance_gause, axis=1).unsqueeze(dim=1) / \
                            torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    std_hdr_original = torch.pow(
        torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)

    std_objective = (alpha / f_factors) * std_gamma * (hdr_pow_mu + epsilon)
    std_objective2 = std_objective * (std_hdr_original.mean() / std_hdr_original)

    # if hdr_original_gray is not None:
    #     hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
    #     hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
    #     std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective2.detach()))
    return res.mean()


def gamma_factor_stdmu_std_no_blf2(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5, hdr_original_gray=None):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    hdr_original_windows = get_im_as_windows(hdr_original_im, wind_size)
    hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    hdr_original_bf_mu = torch.sum(hdr_original_windows * distance_gause, axis=1).unsqueeze(dim=1) / \
                         torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (
            hdr_original_windows - hdr_original_bf_mu)
    sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * distance_gause, axis=1).unsqueeze(dim=1) / \
                            torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    std_hdr_original = torch.pow(
        torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)

    std_objective = (alpha / f_factors) * std_gamma * (hdr_pow_mu + epsilon)
    std_objective2 = std_objective * (std_hdr_original.mean() / (2 * std_hdr_original))

    # if hdr_original_gray is not None:
    #     hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
    #     hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
    #     std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective2.detach()))
    return res.mean()




def gamma_factor_stdmu_res_no_blf(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                           f_factors=None, wind_size=5, hdr_original_gray=None):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    hdr_original_windows = get_im_as_windows(hdr_original_im, wind_size)
    hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    hdr_original_bf_mu = torch.sum(hdr_original_windows * distance_gause, axis=1).unsqueeze(dim=1) / \
                         torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (
            hdr_original_windows - hdr_original_bf_mu)
    sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * distance_gause, axis=1).unsqueeze(dim=1) / \
                            torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    std_hdr_original = torch.pow(
        torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)

    std_objective = (alpha / f_factors) * std_gamma * (hdr_pow_mu + epsilon)

    # if hdr_original_gray is not None:
    #     hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
    #     hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
    #     std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective.detach()))
    res *= (std_hdr_original.mean() / std_hdr_original)
    return res.mean()


def gamma_factor_blfmu_std(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5, hdr_original_gray=None):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    w_blf = (r_weights.sum(dim=1) - 1).unsqueeze(dim=1)
    print("---- w_blf mean ", w_blf.mean())
    std_objective = (w_blf / w_blf.mean()) * ((alpha / f_factors) * std_gamma * (hdr_pow_mu + epsilon))

    if hdr_original_gray is not None:
        hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
        hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
        std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective.detach()))
    return res.mean()


def gamma_factor_blfmu_res(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                           f_factors=None, wind_size=5, hdr_original_gray=None):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    w_blf = (r_weights.sum(dim=1) - 1).unsqueeze(dim=1)
    std_objective = (alpha / f_factors) * std_gamma * (hdr_pow_mu + epsilon)

    if hdr_original_gray is not None:
        hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
        hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
        std_objective = std_objective * hdr_original_gray_max

    res = ones - (std_fake / (std_fake + std_objective.detach()))
    res *= (w_blf / w_blf.mean())
    return res.mean()


def loss_bilateral_original_im(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    hdr_original_windows = get_im_as_windows(hdr_original_im, wind_size)
    hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    hdr_original_bf_mu = torch.sum(hdr_original_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (hdr_original_windows - hdr_original_bf_mu)
    sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_hdr_original = torch.pow(torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)

    res = ones - (std_fake / (std_fake + std_hdr_original.detach() + epsilon))
    return res.mean()


def loss_bilateral_original_im_blf(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    hdr_original_windows = get_im_as_windows(hdr_original_im, wind_size)
    hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    hdr_original_bf_mu = torch.sum(hdr_original_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (hdr_original_windows - hdr_original_bf_mu)
    sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_hdr_original = torch.pow(torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)

    res = ones - (std_fake / (std_fake + std_hdr_original.detach() + epsilon))
    res *= r_weights.mean(dim=1).unsqueeze(dim=1)
    return res.mean()


def get_std_no_blf(input_im, wind_size, distance_gause):
    hdr_original_windows = get_im_as_windows(input_im, wind_size)
    hdr_original_windows = hdr_original_windows.squeeze(dim=1)
    hdr_original_windows = hdr_original_windows.permute(0, 3, 1, 2)
    hdr_original_bf_mu = torch.sum(hdr_original_windows * distance_gause, axis=1).unsqueeze(dim=1) / \
                         torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    hdr_original_bf_mu = hdr_original_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    hdr_original_minus_mean_sq = (hdr_original_windows - hdr_original_bf_mu) * (
            hdr_original_windows - hdr_original_bf_mu)
    sigma_hdr_original_sq = torch.sum(hdr_original_minus_mean_sq * distance_gause, axis=1).unsqueeze(dim=1) / \
                            torch.sum(distance_gause, axis=1).unsqueeze(dim=1)
    std_hdr_original = torch.pow(
        torch.max(sigma_hdr_original_sq, torch.zeros_like(sigma_hdr_original_sq)) + params.epsilon, 0.5)
    return std_hdr_original


def std_loss_blf_off(window, fake, gamma_hdr, hdr_original_im, threshold=0.1, mse_loss=None, alpha=1, r_weights=None,
                      f_factors=None, wind_size=5, hdr_original_gray=None, log=False):
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window_expand = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window_expand.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf_mu = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf_mu = fake_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    fake_minus_mean_sq = (fake_windows - fake_bf_mu) * (fake_windows - fake_bf_mu)
    sigma_fake_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_fake = torch.pow(torch.max(sigma_fake_sq, torch.zeros_like(sigma_fake_sq)) + params.epsilon, 0.5)

    gamma_windows = get_im_as_windows(gamma_hdr, wind_size)
    gamma_windows = gamma_windows.squeeze(dim=1)
    gamma_windows = gamma_windows.permute(0, 3, 1, 2)
    gamma_bf_mu = torch.sum(gamma_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    gamma_bf_mu = gamma_bf_mu.expand(-1, wind_size * wind_size, -1, -1)
    gamma_minus_mean_sq = (gamma_windows - gamma_bf_mu) * (gamma_windows - gamma_bf_mu)
    sigma_gamma_sq = torch.sum(gamma_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std_gamma = torch.pow(torch.max(sigma_gamma_sq, torch.zeros_like(sigma_gamma_sq)) + params.epsilon, 0.5)

    f_factors = f_factors.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    hdr_im_pow_gamma = torch.pow(hdr_original_im, 1 - f_factors)
    hdr_pow_windows = get_im_as_windows(hdr_im_pow_gamma, wind_size)
    hdr_pow_windows = hdr_pow_windows.squeeze(dim=1)
    hdr_pow_windows = hdr_pow_windows.permute(0, 3, 1, 2)
    hdr_pow_mu = torch.sum(hdr_pow_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    std_objective = (alpha / f_factors) * std_gamma * (hdr_pow_mu + params.epsilon2)
    if hdr_original_gray is not None:
        hdr_original_gray = hdr_original_gray.view(hdr_original_gray.size(0), -1)
        hdr_original_gray_max = hdr_original_gray.max(1, keepdim=True)[0].reshape(f_factors.shape)
        std_objective = std_objective * hdr_original_gray_max

    if log:
        hdr_original_im = torch.log(hdr_original_im * 1000 + 1)
        hdr_original_im = hdr_original_im / hdr_original_im.max()
    std_hdr_original = get_std_no_blf(hdr_original_im, wind_size, distance_gause)
    import matplotlib.pyplot as plt
    plt.imshow(std_hdr_original[0,0].numpy(), cmap='gray')
    plt.show()
    res = ones - (std_fake / (std_fake + std_objective.detach() + params.epsilon2))
    res[std_hdr_original >= threshold] = 0
    return res.mean()


def std_loss_bilateral(window, fake, gamma_hdr, hdr_original_im, epsilon, mse_loss=None, alpha=1,
                       r_weights=None, f_factors=None):
    wind_size = 5
    ones = torch.ones(fake.shape)
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
        ones = ones.cuda(fake.get_device())
    window = window.type_as(fake)
    window = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    fake_bf = fake_bf.expand(-1, wind_size * wind_size, -1, -1)

    fake_minus_mean_sq = (fake_windows - fake_bf) * (fake_windows - fake_bf)
    sigma1_sq = torch.sum(fake_minus_mean_sq * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon, 0.5)

    res = ones - (std1 / (std1 + epsilon))
    return res.mean()


def mu_loss(window, fake, hdr_input, mse_loss, r_weights, wind_size):
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
    window = window.type_as(fake)
    window = window / window.sum()
    mu1 = F.conv2d(fake, window, padding=5 // 2, groups=1)
    mu2 = F.conv2d(hdr_input, window, padding=5 // 2, groups=1)
    return mse_loss(mu1, mu2.detach())


def mu_loss_bilateral(window, fake, hdr_input, mse_loss, r_weights, wind_size=5):
    if fake.is_cuda:
        window = window.cuda(fake.get_device())
    window = window.type_as(fake)
    window = window.reshape(1, wind_size * wind_size, 1, 1).contiguous()
    distance_gause = window.expand(1, -1, r_weights.shape[2], r_weights.shape[3])
    weights_map = distance_gause * r_weights
    weights_map_sum = torch.sum(weights_map, axis=1).unsqueeze(dim=1)

    fake_windows = get_im_as_windows(fake, wind_size)
    fake_windows = fake_windows.squeeze(dim=1)
    fake_windows = fake_windows.permute(0, 3, 1, 2)
    fake_bf = torch.sum(fake_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum

    hdr_input_windows = get_im_as_windows(hdr_input, wind_size)
    hdr_input_windows = hdr_input_windows.squeeze(dim=1)
    hdr_input_windows = hdr_input_windows.permute(0, 3, 1, 2)
    hdr_input_bf = torch.sum(hdr_input_windows * weights_map, axis=1).unsqueeze(dim=1) / weights_map_sum
    return mse_loss(fake_bf, hdr_input_bf.detach())


# =======================================
# ========= Helper Functions ============
# =======================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2 / float(2 * sigma ** 2))) for x in range(window_size)])
    # return gauss / gauss.sum()
    return gauss

# def get_bilateral_filter(window_size)


def create_window(window_size, channel):
    window = torch.ones((1, channel, window_size, window_size))
    return window / window.sum()


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


def get_gaussian_kernel(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def get_gaussian_kernel2(window_size, channel):
    sigma = 2
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def get_laplacian_kernel(kernel_size):
    """Returns a 2D Laplacian kernel array."""
    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    return kernel.double()


def get_blf_log_input(hdr_original_gray_norm, gamma_factor, alpha=1):
    gamma_factor = gamma_factor.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
    # print(gamma_factor)
    brightness_factor = 10 ** (1 / gamma_factor - 1)
    # print(brightness_factor)
    res_log = torch.log(hdr_original_gray_norm * brightness_factor + 1) ** alpha
    return res_log


def get_radiometric_weights(gamma_input, wind_size, sigma_r, bilateral_mu, blf_input="gamma"):
    if blf_input == "gamma":
        gamma_input = data_loader_util.crop_input_hdr_batch(gamma_input)
    gamma_input = gamma_input ** bilateral_mu
    m = nn.ZeroPad2d(wind_size // 2)
    radiometric_weights_arr = []
    for i in range(4):
        centers_gamma = gamma_input.expand(-1, wind_size * wind_size, -1, -1)
        a = m(gamma_input)
        windows = a.unfold(dimension=2, size=wind_size, step=1)
        windows = windows.unfold(dimension=3, size=wind_size, step=1)
        windows = windows.reshape(windows.shape[0], windows.shape[1],
                                  windows.shape[2], windows.shape[3],
                                  wind_size * wind_size)
        windows = windows.squeeze(dim=1)
        windows = windows.permute(0, 3, 1, 2)
        radiometric_dist = torch.abs(windows - centers_gamma)
        k = radiometric_dist ** 2 / (2 * sigma_r ** 2)
        radiometric_gaus = torch.exp(-k)
        radiometric_gaus = radiometric_gaus.type_as(gamma_input)
        radiometric_weights_arr.append(radiometric_gaus)
        gamma_input = F.interpolate(gamma_input, scale_factor=0.5, mode='bicubic', align_corners=False)
    return radiometric_weights_arr
