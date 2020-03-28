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


def npy_loader(path, addFrame, hdrMode, normalization, use_c3, apply_wind_norm, device):
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
        if addFrame:
            input_im = data_loader_util.add_frame_to_im(input_im)
        gray_original_im = hdr_image_util.to_gray_tensor(color_im)
        gray_original_im_norm = gray_original_im / gray_original_im.max()
        if apply_wind_norm:
            input_im = hdr_windows_loader(5, gray_original_im_norm, input_im, device)
        return input_im, color_im, gray_original_im_norm, gray_original_im
    return input_im, color_im, input_im, input_im


def hdr_windows_loader(wind_size, a, b, device):
    """

    :param wind_size:
    :param a: original_im
    :param b: target_im
    :return:
    """
    m = nn.ZeroPad2d(wind_size // 2)
    a, b = m(a), m(b)
    a, b = torch.unsqueeze(a, dim=0), torch.unsqueeze(b, dim=0)
    windows = a.unfold(dimension=2, size=5, step=1)
    windows = windows.unfold(dimension=3, size=5, step=1)
    windows = windows.reshape(windows.shape[0], windows.shape[1],
                              windows.shape[2], windows.shape[3],
                              wind_size * wind_size)
    window = torch.ones((1, 1, wind_size, wind_size))
    window = window / window.sum()
    if a.is_cuda:
        window = window.cuda(a.get_device())
        if not b.is_cuda:
            b = b.cuda(a.get_device())
    b = b.type_as(a)
    window = window.type_as(a)
    mu1 = F.conv2d(a, window, padding=0, groups=1)
    mu2 = F.conv2d(b, window, padding=0, groups=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    sigma1_sq = F.conv2d(a * a, window, padding=0, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(b * b, window, padding=0, groups=1) - mu2_sq
    std1 = torch.pow(torch.max(sigma1_sq, torch.zeros_like(sigma1_sq)) + params.epsilon, 0.5)
    std2 = torch.pow(torch.max(sigma2_sq, torch.zeros_like(sigma2_sq)) + params.epsilon, 0.5)
    norm_std2 = std2 / (torch.max(mu2, torch.zeros_like(sigma1_sq)) + params.epsilon)
    compressed_std = torch.pow(norm_std2, 0.8)
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
    # hdr_image_util.print_tensor_details(windows,"result")
    return windows





class ProcessedDatasetFolder(DatasetFolder):
    """
    A customized data loader, to load .npy file that contains a dict
    of numpy arrays that represents hdr_images and window_binary_images.
    """

    def __init__(self, root, addFrame, hdrMode, normalization, transform=None, target_transform=None,
                 loader=npy_loader, use_c3=False, apply_wind_norm=False, device=torch.device("cpu")):
        super(ProcessedDatasetFolder, self).__init__(root, loader, IMG_EXTENSIONS_local,
                                                     transform=transform,
                                                     target_transform=target_transform)
        self.imgs = self.samples
        self.addFrame = addFrame
        self.hdrMode = hdrMode
        self.normalization = normalization
        self.use_c3 = use_c3
        self.apply_wind_norm = apply_wind_norm
        self.device = device

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: {'hdr_image': im, 'binary_wind_image': binary_im}
        """
        path, target = self.samples[index]
        # if self.test_mode:
        # input_im, color_im = self.loader(path, self.test_mode)
        input_im, color_im, gray_original_norm, gray_original = self.loader(path, self.addFrame, self.hdrMode,
                                                        self.normalization, self.use_c3, self.apply_wind_norm, self.device)
        return {"input_im": input_im, "color_im": color_im, "original_gray_norm": gray_original_norm, "original_gray": gray_original}
        # else:
        #     input_im = self.loader(path, self.test_mode)
        # image, binary_window = self.loader(path)
        # sample = {params.image_key: image, params.window_image_key: binary_window}
        # if self.transform:
        #     image = self.transform(image)
        # return input_im, target
