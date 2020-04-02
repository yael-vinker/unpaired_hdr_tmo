import numpy as np
import torch
# from __future__ import print_function
from torchvision.datasets import DatasetFolder
import utils.hdr_image_util as hdr_image_util
from utils import params
import utils.data_loader_util as data_loader_util
from torch import nn
import torch.nn.functional as F
IMG_EXTENSIONS_local = ('.npy')


def npy_loader(path, addFrame, hdrMode, normalization, apply_wind_norm, std_norm_factor, get_window_input):
    """
    load npy files that contain the loaded HDR file, and binary image of windows centers.
    :param path: image path
    :return:
    """

    data = np.load(path, allow_pickle=True)
    input_im = data[()]["input_image"]
    color_im = data[()]["display_image"]
    if not hdrMode:
        if normalization == "max_normalization":
            input_im = input_im / input_im.max()
        elif normalization == "bugy_max_normalization":
            input_im = input_im / 255
        elif normalization == "min_max_normalization":
            input_im = (input_im - input_im.min()) / (input_im.max() - input_im.min())
    if hdrMode:
        gray_original_im = hdr_image_util.to_gray_tensor(color_im)
        gray_original_im_norm = gray_original_im / gray_original_im.max()
        if apply_wind_norm:
            input_im = get_window_input(5, gray_original_im_norm, input_im, std_norm_factor)
        if addFrame:
            input_im = data_loader_util.add_frame_to_im(input_im)
        return input_im, color_im, gray_original_im_norm, gray_original_im
    return input_im, color_im, input_im, input_im


class ProcessedDatasetFolder(DatasetFolder):
    """
    A customized data loader, to load .npy file that contains a dict
    of numpy arrays that represents hdr_images and window_binary_images.
    """

    def __init__(self, root, dataset_properties,
                 hdrMode, transform=None, target_transform=None,
                 loader=npy_loader):
        super(ProcessedDatasetFolder, self).__init__(root, loader, IMG_EXTENSIONS_local,
                                                     transform=transform,
                                                     target_transform=target_transform)
        self.loader_map = {"a": hdr_windows_loader_a,
                           "b": hdr_windows_loader_b,
                           "c": hdr_windows_loader_c,
                           "d": hdr_windows_loader_d}
        self.imgs = self.samples
        self.addFrame = dataset_properties["add_frame"]
        self.hdrMode = hdrMode
        self.normalization = dataset_properties["normalization"]
        self.apply_wind_norm = dataset_properties["apply_wind_norm"]
        self.std_norm_factor = dataset_properties["std_norm_factor"]
        self.wind_norm_option = dataset_properties["wind_norm_option"]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: {'hdr_image': im, 'binary_wind_image': binary_im}
        """
        path, target = self.samples[index]
        input_im, color_im, gray_original_norm, gray_original = self.loader(path, self.addFrame, self.hdrMode,
                                                                            self.normalization, self.apply_wind_norm,
                                                                            self.std_norm_factor,
                                                                            self.loader_map[self.wind_norm_option])
        return {"input_im": input_im, "color_im": color_im, "original_gray_norm": gray_original_norm, "original_gray": gray_original}


def hdr_windows_loader_a(wind_size, a, b, std_norm_factor):
    """
    map mu_hdr -> mu_gamma, std_hdr -> (std_gamma / mu_gamma) ^ std_norm_factor
    :param wind_size:
    :param a: original_im
    :param b: target_im
    :param std_norm_factor
    :return:
    """
    a, b, window = get_window_and_set_device(wind_size, a, b)
    windows = get_im_as_windows(a, wind_size)
    mu1, std1 = get_mu_and_std(a, window)
    mu2, std2 = get_mu_and_std(b, window)
    norm_std2 = std2 / (torch.max(mu2, torch.zeros_like(mu2)) + params.epsilon)
    compressed_std = torch.pow(norm_std2, std_norm_factor)
    print("\ntarget mu", mu2[0, 0, 100, 100], " target std", compressed_std[0, 0, 100, 100])
    mu1, std1 = mu1.unsqueeze(dim=4), std1.unsqueeze(dim=4)
    mu1 = mu1.expand(-1, -1, -1, -1, wind_size * wind_size)
    std1 = std1.expand(-1, -1, -1, -1, wind_size * wind_size)
    windows = windows - mu1
    windows = windows / (std1 + params.epsilon)
    mu2, compressed_std = mu2.unsqueeze(dim=4), compressed_std.unsqueeze(dim=4)
    mu2, compressed_std = mu2.expand(-1, -1, -1, -1, 25), compressed_std.expand(-1, -1, -1, -1, 25)
    windows = windows * compressed_std
    windows = windows + mu2
    print("\nres mu", torch.mean(windows[0, 0, 100, 100])," res std", torch.std(windows[0, 0, 100, 100]))
    windows = windows.permute(0, 4, 2, 3, 1)
    windows = windows.squeeze(dim=4)
    windows = windows.squeeze(dim=0)
    return windows


def get_window_and_set_device(wind_size, a, b):
    m = nn.ZeroPad2d(wind_size // 2)
    a, b = m(a), m(b)
    a, b = torch.unsqueeze(a, dim=0), torch.unsqueeze(b, dim=0)
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


def get_im_as_windows(a, wind_size):
    windows = a.unfold(dimension=2, size=5, step=1)
    windows = windows.unfold(dimension=3, size=5, step=1)
    windows = windows.reshape(windows.shape[0], windows.shape[1],
                              windows.shape[2], windows.shape[3],
                              wind_size * wind_size)
    return windows


def hdr_windows_loader_b(wind_size, a, b, std_norm_factor):
    """
    map mu_hdr -> mu_gamma, std_hdr -> (std_hdr / mu_hdr) ^ std_norm_factor
    :param wind_size:
    :param a: original_im
    :param b: target_im
    :param std_norm_factor
    :return:
    """
    a, b, window = get_window_and_set_device(wind_size, a, b)
    windows = get_im_as_windows(a, wind_size)
    mu1, std1 = get_mu_and_std(a, window)
    mu2, std2 = get_mu_and_std(b, window)
    norm_std1 = std1 / (torch.max(mu1, torch.zeros_like(mu1)) + params.epsilon)
    compressed_std = torch.pow(norm_std1, std_norm_factor)
    print("\ntarget mu", mu2[0, 0, 100, 100], " target std", compressed_std[0, 0, 100, 100])
    mu1, std1 = mu1.unsqueeze(dim=4), std1.unsqueeze(dim=4)
    mu1 = mu1.expand(-1, -1, -1, -1, wind_size * wind_size)
    std1 = std1.expand(-1, -1, -1, -1, wind_size * wind_size)
    windows = windows - mu1
    windows = windows / (std1 + params.epsilon)
    mu2, compressed_std = mu2.unsqueeze(dim=4), compressed_std.unsqueeze(dim=4)
    mu2, compressed_std = mu2.expand(-1, -1, -1, -1, 25), compressed_std.expand(-1, -1, -1, -1, 25)
    windows = windows * compressed_std
    windows = windows + mu2
    print("\nres mu", torch.mean(windows[0, 0, 100, 100])," res std", torch.std(windows[0, 0, 100, 100]))
    windows = windows.permute(0, 4, 2, 3, 1)
    windows = windows.squeeze(dim=4)
    windows = windows.squeeze(dim=0)
    return windows


def hdr_windows_loader_c(wind_size, a, b, std_norm_factor):
    """
    map mu_hdr -> mu_gamma, std_hdr ->  std_norm_factor * std_hdr * (mu_gamma / mu_hdr)
    :param wind_size:
    :param a: original_im
    :param b: target_im
    :param std_norm_factor
    :return:
    """
    a, b, window = get_window_and_set_device(wind_size, a, b)
    windows = get_im_as_windows(a, wind_size)
    mu1, std1 = get_mu_and_std(a, window)
    mu2, std2 = get_mu_and_std(b, window)
    norm_std1 = std1 * (mu2 / (torch.max(mu1, torch.zeros_like(mu1)) + params.epsilon))
    compressed_std = norm_std1 * std_norm_factor
    print("\ntarget mu", mu2[0, 0, 100, 100], " target std", compressed_std[0, 0, 100, 100])
    mu1, std1 = mu1.unsqueeze(dim=4), std1.unsqueeze(dim=4)
    mu1 = mu1.expand(-1, -1, -1, -1, wind_size * wind_size)
    std1 = std1.expand(-1, -1, -1, -1, wind_size * wind_size)
    windows = windows - mu1
    windows = windows / (std1 + params.epsilon)
    mu2, compressed_std = mu2.unsqueeze(dim=4), compressed_std.unsqueeze(dim=4)
    mu2, compressed_std = mu2.expand(-1, -1, -1, -1, 25), compressed_std.expand(-1, -1, -1, -1, 25)
    windows = windows * compressed_std
    windows = windows + mu2
    print("\nres mu", torch.mean(windows[0, 0, 100, 100])," res std", torch.std(windows[0, 0, 100, 100]))
    windows = windows.permute(0, 4, 2, 3, 1)
    windows = windows.squeeze(dim=4)
    windows = windows.squeeze(dim=0)
    return windows


def hdr_windows_loader_d(wind_size, a, b, std_norm_factor):
    """
    map mu_hdr -> mu_gamma, std_hdr ->  std_norm_factor * mu_gamma
    :param wind_size:
    :param a: original_im
    :param b: target_im
    :param std_norm_factor
    :return:
    """
    a, b, window = get_window_and_set_device(wind_size, a, b)
    windows = get_im_as_windows(a, wind_size)
    mu1, std1 = get_mu_and_std(a, window)
    mu2, std2 = get_mu_and_std(b, window)
    norm_std1 = mu2
    compressed_std = norm_std1 * std_norm_factor
    print("\ntarget mu", mu2[0, 0, 100, 100], " target std", compressed_std[0, 0, 100, 100])
    mu1, std1 = mu1.unsqueeze(dim=4), std1.unsqueeze(dim=4)
    mu1 = mu1.expand(-1, -1, -1, -1, wind_size * wind_size)
    std1 = std1.expand(-1, -1, -1, -1, wind_size * wind_size)
    windows = windows - mu1
    windows = windows / (std1 + params.epsilon)
    mu2, compressed_std = mu2.unsqueeze(dim=4), compressed_std.unsqueeze(dim=4)
    mu2, compressed_std = mu2.expand(-1, -1, -1, -1, 25), compressed_std.expand(-1, -1, -1, -1, 25)
    windows = windows * compressed_std
    windows = windows + mu2
    print("\nres mu", torch.mean(windows[0, 0, 100, 100])," res std", torch.std(windows[0, 0, 100, 100]))
    windows = windows.permute(0, 4, 2, 3, 1)
    windows = windows.squeeze(dim=4)
    windows = windows.squeeze(dim=0)
    return windows

