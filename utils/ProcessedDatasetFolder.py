import os

import numpy as np
import torch
from torchvision.datasets import DatasetFolder

from utils import data_loader_util, hdr_image_util, params
import torch.nn.functional as F

IMG_EXTENSIONS_local = ('.npy')


def get_ldr_im(normalization, input_im, max_stretch, min_stretch):
    if normalization == "max_normalization":
        input_im = input_im / input_im.max()
    elif normalization == "bugy_max_normalization":
        input_im = input_im / 255
    elif normalization == "stretch":
        input_im = ((input_im - input_im.min()) / input_im.max()) * max_stretch - min_stretch
        input_im = np.clip(input_im, 0, 1)
    return input_im


def get_f(use_hist_fit, f_dict_path, im_name, factor_coeff, use_contrast_ratio_f, gray_original_im, gamma_factor):
    if use_hist_fit:
        data = np.load(f_dict_path, allow_pickle=True)
        if im_name in data[()]:
            f_factor = data[()][im_name]
            # print("[%s] found in dict [%.4f]" % (im_name, f_factor))
            brightness_factor = f_factor * 255 * factor_coeff
    elif use_contrast_ratio_f:
        print("contrast ratio")
        im_max = np.percentile(gray_original_im, 99.0)
        im_min = np.percentile(gray_original_im, 1.0)
        if im_min == 0:
            print("min = 0")
            im_min += 0.0001
        brightness_factor = im_max / im_min * factor_coeff
    else:
        print("================== no factor found for [%s] falls to gamma ==================" % (im_name))
        brightness_factor = (10 ** (1 / gamma_factor - 1)) * factor_coeff
    return brightness_factor


def npy_loader(path, addFrame, hdrMode, normalization, min_stretch,
               max_stretch, factor_coeff, use_contrast_ratio_f, use_hist_fit, f_dict_path,
               final_shape_addition):
    """
    load npy files that contain the loaded HDR file, and binary image of windows centers.

    """
    data = np.load(path, allow_pickle=True)
    input_im = data[()]["input_image"]
    color_im = data[()]["display_image"]
    input_im_max, color_im_max = input_im.max(), color_im.max()
    input_im = F.interpolate(input_im.unsqueeze(dim=0), size=(params.input_size, params.input_size), mode='bicubic',
                             align_corners=False).squeeze(dim=0).clamp(min=0, max=input_im_max)
    color_im = F.interpolate(color_im.unsqueeze(dim=0), size=(params.input_size, params.input_size), mode='bicubic',
                             align_corners=False).squeeze(dim=0).clamp(min=0, max=color_im_max)
    if not hdrMode:
        input_im = get_ldr_im(normalization, input_im, max_stretch, min_stretch)
        return input_im, color_im, input_im, input_im, 0
    if hdrMode:
        gray_original_im = hdr_image_util.to_gray_tensor(color_im)
        gray_original_im_norm = gray_original_im / gray_original_im.max()
        gamma_factor = data[()]["gamma_factor"]
        # TODO(): fix this part
        im_name = os.path.splitext(os.path.basename(path))[0]
        brightness_factor = get_f(use_hist_fit, f_dict_path, im_name, factor_coeff, use_contrast_ratio_f,
                                  gray_original_im, gamma_factor)
        gray_original_im = gray_original_im - gray_original_im.min()
        a = torch.log10((gray_original_im / gray_original_im.max()) * brightness_factor + 1)
        input_im = a / a.max()
        if addFrame:
            input_im = data_loader_util.add_frame_to_im(input_im, final_shape_addition, final_shape_addition)
        return input_im, color_im, gray_original_im_norm, gray_original_im, data[()]["gamma_factor"]


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
        self.imgs = self.samples
        self.addFrame = dataset_properties["add_frame"]
        self.hdrMode = hdrMode
        self.normalization = dataset_properties["normalization"]
        self.max_stretch = dataset_properties["max_stretch"]
        self.min_stretch = dataset_properties["min_stretch"]
        self.factor_coeff = dataset_properties["factor_coeff"]
        self.use_contrast_ratio_f = dataset_properties["use_contrast_ratio_f"]
        self.use_hist_fit = dataset_properties["use_hist_fit"]
        self.f_train_dict_path = dataset_properties["f_train_dict_path"]
        self.final_shape_addition = dataset_properties["final_shape_addition"]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: {'hdr_image': im, 'binary_wind_image': binary_im}
        """
        path, target = self.samples[index]
        input_im, color_im, gray_original_norm, gray_original, gamma_factor = self.loader(path, self.addFrame,
                                                                                          self.hdrMode,
                                                                                          self.normalization,
                                                                                          self.min_stretch,
                                                                                          self.max_stretch,
                                                                                          self.factor_coeff,
                                                                                          self.use_contrast_ratio_f,
                                                                                          self.use_hist_fit,
                                                                                          self.f_train_dict_path,
                                                                                          self.final_shape_addition)
        return {"input_im": input_im, "color_im": color_im, "original_gray_norm": gray_original_norm,
                "original_gray": gray_original, "gamma_factor": gamma_factor}
