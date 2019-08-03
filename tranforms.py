import random

import params
import torchvision.transforms.functional as F
import torch
import time
import torchvision.transforms as torch_transforms

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

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor, binary_wind = sample[params.image_key], sample[params.window_image_key]
        return {params.image_key: F.normalize(tensor, self.mean, self.std, self.inplace),
                params.window_image_key: binary_wind}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

#
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
#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#
#         Returns:
#             Tensor: Normalized Tensor image.
#         """
#         return F.normalize(tensor, self.mean, self.std, self.inplace)
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
#


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
        IMAGE_SCALE = 100
        image, binary_wind = sample[params.image_key], sample[params.window_image_key]
        if image.ndim == 2:
            print("image.ndim == 2")
            image = image[:, :, None]

        image_tensor = torch.from_numpy(image.transpose((2, 0, 1)))

        # print(image_tensor.shape)
        # # height = image_tensor.shape[1]
        # # width = image_tensor.shape[2]
        # # max_axis = height if height > width else width
        # # value = max_axis / MAX_AXIS
        # # new_height = int(height / value)
        # # new_width = int(width / value)
        # # im = image_tensor.reshape(new_width, new_height)
        # height = image_tensor.shape[1] - 128
        # width = image_tensor.shape[2] - 128
        # rand_x = random.randint(0, width)
        # rand_y = random.randint(0, height)
        # cur = time.time()
        # im = image_tensor[:, rand_y: rand_y + 128, rand_x: rand_x + 128]
        # print("gggg   ", im.shape)
        # max_origin = im.max()
        # im100 = (im / max_origin) * IMAGE_SCALE
        # im_log = torch.log(im100 + 1)

        # There is no need to permute "window_image" channels since it has 1 channel
        return {params.image_key: image_tensor,
                params.window_image_key: torch.from_numpy(binary_wind)}
