from __future__ import print_function

from torchvision.datasets import DatasetFolder
import imageio
import pathlib
import skimage
import params
import cv2
from scipy import ndimage
import numpy as np
import random
import time
import torch


IMG_EXTENSIONS_local = ['.hdr']
IMAGE_SCALE = 100
MAX_AXIS = 600


def hdr_loader(path, device):
    """
    Loader for .hdr image
    :param path: image path
    :param im_size: new size
    :return: the given HDR image, scaled to im_size*im_size, normalised to [0,1]
    """
    # print("-----hdr")

    start = time.time()
    path = pathlib.Path(path)
    im_origin = imageio.imread(path, format='HDR-FI')

    # print("load image ",time.time() - start)
    # start = time.time()
    # if trainMode:
    height = im_origin.shape[0]
    width = im_origin.shape[1]
    max_axis = height if height > width else width
    value = max_axis / MAX_AXIS
    new_height = int(height / value)
    new_width = int(width / value)
    im = cv2.resize(im_origin, (new_width, new_height))
    image = im.transpose((2, 0, 1))
    image_tensor = torch.as_tensor(image)
    # print(image_tensor.shape)
    # height = image_tensor.shape[1]
    # width = image_tensor.shape[2]
    # max_axis = height if height > width else width
    # value = max_axis / MAX_AXIS
    # new_height = int(height / value)
    # new_width = int(width / value)
    # im = image_tensor.reshape(new_width, new_height)
    height = image_tensor.shape[1] - 128
    width = image_tensor.shape[2] - 128
    rand_x = random.randint(0, width)
    rand_y = random.randint(0, height)
    # cur = time.time()
    im = image_tensor[:, rand_y: rand_y + 128, rand_x: rand_x + 128]
    max_origin = im.max()
    im100 = (im / max_origin) * IMAGE_SCALE
    im_log = torch.log(im100 + 1)

    # print(im.shape)
    #     height = im.shape[0] - 128
    #     width = im.shape[1] - 128
    #     rand_x = random.randint(0, width)
    #     rand_y = random.randint(0, height)
    #     cur = time.time()
    #     im = im[rand_y: rand_y + 128, rand_x: rand_x + 128]
    #     print("slice actual ", time.time() - cur)
    # else:
    #     im = cv2.resize(im_origin, (128, 128))
    # print("slice image hdr ",time.time() - start)
    # max_origin = np.nanmax(im)
    # im100 = (im / max_origin) * IMAGE_SCALE
    # im_log = np.log(im100 + 1)
    # return im_log
    return im_log



class HdrImageFolder(DatasetFolder):
    """
    A customized data loader, to load .hdr file
    """

    def __init__(self, root, device, transform=None, target_transform=None,
                 loader=hdr_loader):
        super(HdrImageFolder, self).__init__(root, loader, IMG_EXTENSIONS_local,
                                             transform=transform,
                                             target_transform=target_transform)
        self.imgs = self.samples
        self.device = device


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: {'hdr_image': im, 'binary_wind_image': binary_im}
        """
        path, target = self.samples[index]
        image = self.loader(path, self.device)
        binary_window = np.zeros(image.shape)
        sample = {params.image_key: image, params.window_image_key: binary_window}
        if self.transform:
            sample = self.transform(sample)
        return sample