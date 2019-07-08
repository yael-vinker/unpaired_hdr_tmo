from __future__ import print_function
from torchvision.datasets import DatasetFolder
import numpy as np
import pathlib
import imageio
import cv2
import random
import matplotlib.pyplot as plt

IMG_EXTENSIONS_local = ('.png', '.bmp')
IMAGE_MAX_VALUE = 255
IMAGE_SCALE = 100

def ldr_loader(path):
    """
    load npy files that contain the loaded HDR file, and binary image of windows centers.
    :param path: image path
    :return:
    """
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    im = (im_origin / IMAGE_MAX_VALUE) * IMAGE_SCALE
    height = im.shape[0] - 128
    width = im.shape[1] - 128
    rand_x = random.randint(0, width)
    rand_y = random.randint(0, height)
    window = im[rand_y: rand_y + 128, rand_x: rand_x + 128]
    # im = cv2.resize(np.log(im + 1), (128, 128))
    return window


class LdrDatasetFolder(DatasetFolder):
    """
    A customized data loader, to load .npy file that contains a dict
    of numpy arrays that represents hdr_images and window_binary_images.
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=ldr_loader):
        super(LdrDatasetFolder, self).__init__(root, loader, IMG_EXTENSIONS_local,
                                                     transform=transform,
                                                     target_transform=target_transform)
        self.imgs = self.samples
