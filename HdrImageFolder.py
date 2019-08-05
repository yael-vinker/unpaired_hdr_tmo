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
import hdr_image_utils


IMG_EXTENSIONS_local = ('.hdr','.jpg','.png')
IMAGE_SCALE = 100
MAX_AXIS = 600


def hdr_loader(path, input_dim, trainMode):
    """
    Loader for .hdr image
    :param path: image path
    :param im_size: new size
    :return: the given HDR image, scaled to im_size*im_size, normalised to [0,1]
    """
    path = pathlib.Path(path)
    # im_origin = imageio.imread(path, format='HDR-FI')
    im_origin = imageio.imread(path)
    if input_dim == 1:
        im_origin = hdr_image_utils.RGB2YUV(im_origin)
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

    im = im_origin
    max_origin = np.max(im)
    im = im_origin / max_origin
    im_log = np.log(im + 1)
    # im100 = (im / max_origin) * IMAGE_SCALE
    # im_log = np.log(im100 + 1)
    return im_log



class HdrImageFolder(DatasetFolder):
    """
    A customized data loader, to load .hdr file
    """

    def __init__(self, root, input_dim, trainMode, transform=None, target_transform=None,
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
        image = self.loader(path, self.input_dim, self.trainMode)
        binary_window = np.zeros(image.shape)
        sample = {params.image_key: image, params.window_image_key: binary_window}
        if self.transform:
            sample = self.transform(sample)
        return sample
