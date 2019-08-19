from __future__ import print_function

from torchvision.datasets import DatasetFolder
import imageio
import pathlib
import skimage
import params
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import hdr_image_utils



IMG_EXTENSIONS_local = ('.hdr','.bmp','.png', '.dng')
IMAGE_SCALE = 100
MAX_AXIS = 600


def hdr_loader(path, input_dim, trainMode):
    """
    Loader for .hdr image
    :param path: image path
    :param im_size: new size
    :return: the given HDR image, scaled to im_size*im_size, normalised to [0,1]
    """
    path_lib_path = pathlib.Path(path)
    file_extension = os.path.splitext(path)[1]
    if file_extension == ".hdr":
        im_origin = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
    elif file_extension == ".dng":
        im_origin = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
    elif file_extension == ".bmp":
        im_origin = imageio.imread(path_lib_path).astype('float32')
    else:
        raise Exception('invalid hdr file format: {}'.format(file_extension))


    #     path = "hdr05.hdr"
    # path = pathlib.Path(path)
    # im_origin = imageio.imread(path)
    # hdr_image_utils.print_image_details(im_origin, "hdr")
    # # raw = rawpy.imread('im2.dng')
    # # rgb = raw.postprocess()
    # # imageio.imsave("dng_image.hdr", rgb)
    # path = pathlib.Path("dng_image.dng")
    # dng_hdr_im = imageio.imread(path, format="RAW-FI")
    # hdr_image_utils.print_image_details(dng_hdr_im, "dng")
    # HDR - FI
    # im_origin = imageio.imread(path, format='HDR-FI')
    # im_origin = imageio.imread(path)
    # if input_dim == 1:
    #     im_origin = hdr_image_utils.RGB2YUV(im_origin)
    # if trainMode:
    #     height = im_origin.shape[0]
    #     width = im_origin.shape[1]
    #     max_axis = height if height > width else width
    #     value = max_axis / MAX_AXIS
    #     new_height = int(height / value)
    #     new_width = int(width / value)
    #     im = cv2.resize(im_origin, (new_width, new_height))
    #     height = im.shape[0] - 128
    #     width = im.shape[1] - 128
    #     rand_x = random.randint(0, width)
    #     rand_y = random.randint(0, height)
    #     im = im[rand_y: rand_y + 128, rand_x: rand_x + 128]
    # else:
    #     im = cv2.resize(im_origin, (128, 128))
    # print(path)
    # print(np.min(im_origin + 1))
    # im = np.log(im_origin + 1)
    # im = im_origin
    max_origin = np.max(im_origin)
    im = (im_origin / max_origin)
    # im_log = np.log(im + 1)
    # im100 = (im / max_origin) * IMAGE_SCALE
    # im_log = np.log(im100 + 1)
    return im



class HdrImageFolder(DatasetFolder):
    """
    A customized data loader, to load .hdr file
    """

    def __init__(self, root, input_dim=3, trainMode=False, transform=None, target_transform=None,
                 loader=hdr_loader):
        super(HdrImageFolder, self).__init__(root, loader, IMG_EXTENSIONS_local,
                                             transform=transform,
                                             target_transform=target_transform)
        self.input_dim = input_dim
        self.imgs = self.samples
        self.trainMode = trainMode


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: {'hdr_image': im, 'binary_wind_image': binary_im}
        """
        path, target = self.samples[index]
        sample = self.loader(path, self.input_dim, self.trainMode)
        # binary_window = np.zeros(image.shape)
        # sample = {params.image_key: image, params.window_image_key: binary_window}
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
