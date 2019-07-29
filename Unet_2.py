import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        depth=3,
        wf=4,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        self.padding = 0
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i))
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, int(prev_channels / 2))
            )
            prev_channels = int(prev_channels / 2)
        self.up_path.append(
            UNetUpBlock(prev_channels, in_channels, padding=0)
        )
        self.last_sig = nn.Sigmoid()

    def forward(self, x):
        y = x.float()
        for i, down in enumerate(self.down_path):
            y = down(y)

        for i, up in enumerate(self.up_path):
            y = up(y)
        
        return y
        #return self.last_sig(y)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=5, stride=2, padding=1, bias=True))
        block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):

        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding=1):
        super(UNetUpBlock, self).__init__()
        block = []
        block.append(nn.ConvTranspose2d(in_size, out_size, kernel_size=5, stride=2, padding=padding, bias=True))
        block.append(nn.ReLU())
        self.up = nn.Sequential(*block)

    def forward(self, x):
        out = self.up(x)
        return out
