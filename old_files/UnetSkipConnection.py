import torch
from torch import nn


class UNetSkipConnection(nn.Module):
    def __init__(
            self,
            in_channels=3,
            depth=3,
            wf=4,
            padding=1,
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
        super(UNetSkipConnection, self).__init__()
        self.padding = 0
        self.depth = depth
        self.input_dim = in_channels
        self.down_path = nn.ModuleList()
        self.down_path = nn.Sequential(
            UNetConvBlock(self.input_dim, 16),
            UNetConvBlock(16, 32),
            UNetConvBlock(32, 64),
        )
        self.up_path = nn.Sequential(
            UNetUpBlock(64, 32),
            UNetUpBlock(32, 16),
            UNetUpBlock(16, self.input_dim),
        )

        # self.conv_block = double_conv(32, 16)

    def forward(self, x):
        y = x.float()
        blocks = []
        for i, down in enumerate(self.down_path):
            blocks.append(y)
            y = down(y)

        for i, up in enumerate(self.up_path):
            y_down = blocks[-i - 1]
            y = up(y, y_down)

        # return out
        return y
        # return self.last_sig(y)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding=1, stride=2):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=4, stride=stride, padding=padding, bias=True))
        block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        block = []
        block.append(nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=True))
        # block.append(nn.ReLU())
        self.block = nn.Sequential(*block)
        self.conv_block = double_conv(out_size * 2, out_size)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
               :, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def forward(self, x1, x2):
        x1_up = self.block(x1)
        x2_down_crop = self.center_crop(x2, x1_up.shape[2:])
        out = torch.cat([x1_up, x2_down_crop], 1)
        out = self.conv_block(out)
        return out


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
