# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import params


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_traspose(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv_traspose, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, network, dilation=0):
        super(down, self).__init__()
        if network == "unet":
            self.down_sample = nn.MaxPool2d(2)
        else:  # for torus
            self.down_sample = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=0, dilation=dilation),
        self.mpconv = nn.Sequential(
            self.down_sample,
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, layer_factor, bilinear, network, dilation):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if network == "unet":
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_ch // layer_factor, in_ch // layer_factor, 2, stride=2)

        else:  # for torus
            self.up = nn.ConvTranspose2d(in_ch // layer_factor, in_ch // layer_factor, 3, stride=1, padding=0, dilation=dilation)

        self.conv = double_conv_traspose(in_ch, out_ch)

    def forward(self, x1, x2, con_operator, network):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if con_operator == "original_unet" or con_operator == "torus":
            x = torch.cat([x2, x1], dim=1)
        elif con_operator == "square":
            square_x = torch.pow(x2, 2)
            x = torch.cat([x2, x1, square_x], dim=1)
        elif con_operator == "square_root":
            square_root_x = torch.pow(x2 + params.epsilon, 0.5)
            x = torch.cat([x2, x1, square_root_x], dim=1)
        else:  # square & square root
            square_x = torch.pow(x2, 2)
            square_root_x = torch.pow(x2 + params.epsilon, 0.5)
            x = torch.cat([x2, x1, square_x, square_root_x], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x