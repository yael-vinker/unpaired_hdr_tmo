# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, input_images_mean, bilinear):
        super(UNet, self).__init__()

        # self.inc = inconv(n_channels, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        # self.up1 = up(1024 * 2, 256, bilinear)
        # self.up2 = up(512 * 2, 128, bilinear)
        # self.up3 = up(256 * 2, 64, bilinear)
        # self.up4 = up(128 * 2, 64, bilinear)
        # self.outc = outconv(64, n_classes)
        down_ch = 32
        self.inc = inconv(n_channels, down_ch)
        ch = down_ch
        self.down1 = down(ch, ch * 2)
        ch = ch * 2
        self.down2 = down(ch, ch * 2)
        ch = ch * 2
        self.down3 = down(ch, ch * 2)
        ch = ch * 2
        self.down4 = down(ch, ch)
        ch = ch * 2
        print(ch)
        self.up1 = up(ch * 2, int(ch / 4), bilinear)
        ch = int(ch / 2)
        self.up2 = up(ch * 2, int(ch / 4), bilinear)
        ch = int(ch / 2)
        self.up3 = up(ch * 2, int(down_ch), bilinear)
        ch = int(ch / 2)
        self.up4 = up(ch * 2, down_ch, bilinear)
        self.outc = outconv(down_ch, n_classes)
        if input_images_mean == 0:
            self.last_sig = nn.Tanh()
        elif input_images_mean == 0.5:
            self.last_sig = nn.Sigmoid()
        else:
            raise Exception('ERROR: Invalid images range')
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.last_sig(x)
        return x