import collections
import numbers

import numpy as np
import skimage.transform
import torch
import torchvision.transforms as torch_transforms

from utils import params


# class Normalize(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
#     will normalize each channel of the input ``torch.*Tensor`` i.e.
#     ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
#
#     .. note::
#         This transform acts out of place, i.e., it does not mutates the input tensor.
#
#     Args:
#         mean (sequence): Sequence of means for each channel.
#         std (sequence): Sequence of standard deviations for each channel.
#     """
#
#     def __init__(self, mean, std, inplace=False):
#         self.mean = mean
#         self.std = std
#         self.inplace = inplace
#
#     def __call__(self, sample):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#
#         Returns:
#             Tensor: Normalized Tensor image.
#         """
#         tensor, binary_wind = sample[params.image_key], sample[params.window_image_key]
#         return {params.image_key: F.normalize(tensor, self.mean, self.std, self.inplace),
#                 params.window_image_key: binary_wind}
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class NormalizeForDisplay(object):
    def __init__(self, mean, std, device, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = tensor.clone().to(self.device)
        # tensor = tensor.to(self.device)
        mean = torch.tensor(self.mean, dtype=torch.float32, device=self.device)
        std = torch.tensor(self.std, dtype=torch.float32, device=self.device)
        # tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
        # to_pil = torch_transforms.ToPILImage()
        return tensor


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample
        if image.ndim == 2:
            image = image[:, :, None]

        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float()
        return image_tensor


def _is_numpy_image(img):
    return isinstance(img, np.ndarray)


class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(pic, output_size):
        """Get parameters for ``crop`` for center crop.
        Args:
            pic (np array): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to the crop for center crop.
        """

        w, h, c = pic.shape
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return i, j, th, tw

    def __call__(self, pic):
        """
        Args:
            pic (np array): Image to be cropped.
        Returns:
            np array: Cropped image.
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make them 3
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        # get crop params: starting pixels and size of the crop
        i, j, h, w = self.get_params(pic, self.size)
        return pic[i:i + h, j:j + w, :]


class Scale(object):
    """
    Rescale the given numpy image to a specified size.
    """

    def __init__(self, size, dtype=np.float32, interpolation="bilinear"):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.dtype = dtype
        self.interpolation = interpolation

    def __call__(self, pic):
        # if self.dtype == np.uint8:
        #     scaled_im = skimage.transform.resize(pic, (self.size, self.size),  mode='reflect', preserve_range=False)
        #     return util.img_as_ubyte(scaled_im) / 255
        im = skimage.transform.resize(pic, (self.size, self.size), mode='reflect', preserve_range=True)
        return im


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        tensor.sub_(mean).div_(std)
        return tensor
        # return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Exp(object):

    def __init__(self, factor, add_clipping, apply_inverse_to_preprocess, normalised_data, factorised_data):
        self.factor = factor
        self.log_factor = torch.tensor(np.log(1 + factor))
        self.add_clipping = add_clipping
        self.normalised_data = normalised_data
        self.factorised_data = factorised_data

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # if self.normalised_data:
        #     im_0_1 = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        # else:
        #     im_0_1 = tensor
        im_end = torch.exp(tensor)
        import utils.printer
        # utils.printer.print_g_progress(im_end, "in_exp__")
        # if self.normalised_data:
        im_end = im_end / im_end.max()
        # utils.printer.print_g_progress(im_end, "max_div__")
        # if self.factorised_data:
        #     im_end = (im_end - im_end.min()) / (im_end.max() - im_end.min())
        if self.add_clipping:
            im_end = im_end * 1.1
            im_end = torch.clamp(im_end, 0, 1)
            # utils.printer.print_g_progress(im_end, "clip__")
        return im_end


image_transform_no_norm = torch_transforms.Compose([
    Scale(params.input_size),
    CenterCrop(params.input_size),
    ToTensor(),
])

# image in [-1, 1]
gray_image_transform = torch_transforms.Compose([
    Scale(params.input_size),
    CenterCrop(params.input_size),
    ToTensor(),
    Normalize(0.5, 0.5),
])

gray_image_transform_original_range = torch_transforms.Compose([
    Scale(params.input_size),
    CenterCrop(params.input_size),
    ToTensor(),
])

rgb_non_display_image_transform = torch_transforms.Compose([
    Scale(params.input_size),
    CenterCrop(params.input_size),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# image in [0,1..] or rgb image display
rgb_display_image_transform = torch_transforms.Compose([
    Scale(params.input_size),
    CenterCrop(params.input_size),
    ToTensor(),
])

rgb_display_image_transform_numpy = torch_transforms.Compose([
    # Scale(params.input_size),
    # CenterCrop(params.input_size),
])

tmqi_input_transforms = torch_transforms.Compose([
    ToTensor(),
    Normalize(0.5, 0.5),
])

hdr_im_transform = torch_transforms.Compose([
    ToTensor(),
])
