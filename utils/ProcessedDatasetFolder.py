import numpy as np
from torchvision.datasets import DatasetFolder
import utils.hdr_image_util as hdr_image_util
import utils.data_loader_util as data_loader_util
import torch
IMG_EXTENSIONS_local = ('.npy')


def npy_loader(path, addFrame, hdrMode, normalization, std_norm_factor, min_stretch,
               max_stretch, factor_coeff):
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
        gamma_factor = data[()]["gamma_factor"]
        brightness_factor = (10 ** (1 / gamma_factor - 1)) * factor_coeff
        gray_original_im = gray_original_im - gray_original_im.min()
        a = torch.log10((gray_original_im / gray_original_im.max()) * brightness_factor + 1)
        input_im = a / a.max()
        if addFrame:
            input_im = data_loader_util.add_frame_to_im(input_im)
        if "gamma_factor" in data[()].keys():
            return input_im, color_im, gray_original_im_norm, gray_original_im, data[()]["gamma_factor"]
        return input_im, color_im, gray_original_im_norm, gray_original_im, 0
    return input_im, color_im, input_im, input_im, 0


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
        self.std_norm_factor = dataset_properties["std_norm_factor"]
        self.max_stretch = dataset_properties["max_stretch"]
        self.min_stretch = dataset_properties["min_stretch"]
        self.factor_coeff = dataset_properties["factor_coeff"]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: {'hdr_image': im, 'binary_wind_image': binary_im}
        """
        path, target = self.samples[index]
        input_im, color_im, gray_original_norm, gray_original, gamma_factor = self.loader(path, self.addFrame, self.hdrMode,
                                                                            self.normalization,
                                                                            self.std_norm_factor,
                                                                              self.min_stretch,
                                                                              self.max_stretch,
                                                                            self.factor_coeff)
        return {"input_im": input_im, "color_im": color_im, "original_gray_norm": gray_original_norm,
                "original_gray": gray_original, "gamma_factor": gamma_factor}
