# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import params


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, unet_norm):
        super(double_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=0)
        if unet_norm == 'batch_norm':
            self.norm = nn.BatchNorm2d(out_ch)
        elif unet_norm == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out_ch)
        else:
            self.norm = None
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=0)
        if unet_norm == 'batch_norm':
            self.norm1 = nn.BatchNorm2d(out_ch)
        elif unet_norm == 'instance_norm':
            self.norm1 = nn.InstanceNorm2d(out_ch)
        else:
            self.norm1 = None
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=0),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=0),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.relu(x)
        x = self.conv1(x)
        if self.norm1:
            x = self.norm1(x)
        x = self.relu1(x)
        return x


class double_conv_traspose(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, unet_norm):
        super(double_conv_traspose, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0)
        if unet_norm == 'batch_norm':
            self.norm = nn.BatchNorm2d(out_ch)
        elif unet_norm == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out_ch)
        else:
            self.norm = None
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0)
        if unet_norm == 'batch_norm':
            self.norm1 = nn.BatchNorm2d(out_ch)
        elif unet_norm == 'instance_norm':
            self.norm1 = nn.InstanceNorm2d(out_ch)
        else:
            self.norm1 = None
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv = nn.Sequential(
        #     nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.relu(x)
        x = self.conv1(x)
        if self.norm1:
            x = self.norm1(x)
        x = self.relu1(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, unet_norm):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, unet_norm)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, network, dilation=0, unet_norm='none'):
        super(down, self).__init__()
        if network == params.unet_network:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, unet_norm)
            )
        elif network == params.torus_network:  # for torus
            self.mpconv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=0, dilation=dilation),
                double_conv(in_ch, out_ch, unet_norm)
            )
        else:
            assert 0, "Unsupported network request: {}".format(self.network)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear, layer_factor, network, dilation, unet_norm):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if network == params.unet_network:
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_ch // layer_factor, in_ch // layer_factor, 2, stride=2)

        elif network == params.torus_network:  # for torus
            self.up = nn.ConvTranspose2d(in_ch // layer_factor, in_ch // layer_factor, 3, stride=1, padding=0,
                                         dilation=dilation)
        else:
            assert 0, "Unsupported network request: {}".format(network)

        self.conv = double_conv_traspose(in_ch, out_ch, unet_norm)

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
        if con_operator == params.original_unet:
            x = torch.cat([x2, x1], dim=1)
        elif con_operator == params.square:
            square_x = torch.pow(x2, 2)
            x = torch.cat([x2, x1, square_x], dim=1)
        elif con_operator == params.square_root:
            square_root_x = torch.pow(x2 + params.epsilon, 0.5)
            x = torch.cat([x2, x1, square_root_x], dim=1)
        elif con_operator == params.square_and_square_root:
            square_x = torch.pow(x2, 2)
            square_root_x = torch.pow(x2 + params.epsilon, 0.5)
            x = torch.cat([x2, x1, square_x, square_root_x], dim=1)
        elif con_operator == params.gamma:
            square_root_x = torch.pow(x2 + params.epsilon, 0.02)
            x = torch.cat([x2, x1, square_root_x], dim=1)
        else:
            assert 0, "Unsupported con_operator request: {}".format(con_operator)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
