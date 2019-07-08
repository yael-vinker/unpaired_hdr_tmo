from __future__ import print_function
from torchvision.datasets import DatasetFolder
import imageio
import pathlib
import skimage
import params
import cv2
from scipy import ndimage
import hdr_image_utils as utils
import numpy as np

IMG_EXTENSIONS_local = ['.hdr']
IMAGE_SCALE = 100


def hdr_loader(path):
    """
    Loader for .hdr image
    :param path: image path
    :param im_size: new size
    :return: the given HDR image, scaled to im_size*im_size, normalised to [0,1]
    """
    path = pathlib.Path(path)
    im_origin = imageio.imread(path, format='HDR-FI')
    max_origin = np.nanmax(im_origin)
    im = (im_origin / max_origin) * IMAGE_SCALE
    return cv2.resize(np.log(im + 1), (128, 128))


class HdrImageFolder(DatasetFolder):
    """
    A customized data loader, to load .hdr file
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=hdr_loader):
        super(HdrImageFolder, self).__init__(root, loader, IMG_EXTENSIONS_local,
                                             transform=transform,
                                             target_transform=target_transform)
        self.imgs = self.samples


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: {'hdr_image': im, 'binary_wind_image': binary_im}
        """
        path, target = self.samples[index]
        image = self.loader(path)
        binary_window = np.zeros(image.shape)
        sample = {params.image_key: image, params.window_image_key: binary_window}
        if self.transform:
            sample = self.transform(sample)
        return sample