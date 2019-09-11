from __future__ import print_function
from torchvision.datasets import DatasetFolder
import imageio
import pathlib
import skimage
import cv2
import numpy as np

IMG_EXTENSIONS_local = ['.hdr', '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


def hdr_loader(path, im_size=256):
    """
    Loader for .hdr image
    :param path: image path
    :param im_size: new size
    :return: the given HDR image, scaled to im_size*im_size, normalised to [0,1]
    """
    path1 = pathlib.Path(path)
    image = imageio.imread(path1, format='HDR-FI')
    new_im = skimage.exposure.rescale_intensity(image, out_range=(0, 1000))
    # new_im = (new_im / 1000)
    new_im = cv2.resize(new_im, (im_size, im_size))
    tone_map = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    im_dub = tone_map.process(new_im.copy().astype(np.float32)[:, :, ::-1])
    im1 = np.clip(im_dub, 0, 1).astype('float32')[:, :, ::-1]
    return np.ascontiguousarray(im1)


class TestImagesDatasetFolder(DatasetFolder):
    """
    A customized data loader, to load .hdr files
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=hdr_loader):
        super(TestImagesDatasetFolder, self).__init__(root, loader, IMG_EXTENSIONS_local,
                                             transform=transform,
                                             target_transform=target_transform)
        self.imgs = self.samples

