# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, input_images_mean, depth, layer_factor, con_operator, filters, bilinear, network, dilation):
        super(UNet, self).__init__()
        self.con_operator = con_operator
        down_ch = filters
        self.depth = depth
        self.inc = inconv(n_channels, down_ch)
        ch = down_ch
        self.down_path = nn.ModuleList()
        for i in range(self.depth - 1):
            self.down_path.append(
                down(ch, ch * 2, network, dilation=dilation)
            )
            ch = ch * 2
        self.down_path.append(down(ch, ch, network))

        self.up_path = nn.ModuleList()
        for i in range(self.depth):
            if i >= self.depth - 2:
                self.up_path.append(
                    up(ch * layer_factor, down_ch, bilinear, layer_factor, network, dilation=dilation)
                )
            else:
                self.up_path.append(
                    up(ch * layer_factor, ch // 2, bilinear, layer_factor, network, dilation=dilation)
                )
            ch = ch // 2
        self.outc = outconv(down_ch, n_classes)
        if input_images_mean == 0:
            self.last_sig = nn.Tanh()
        elif input_images_mean == 0.5:
            self.last_sig = nn.Sigmoid()
        else:
            raise Exception('ERROR: Invalid images range')
    def forward(self, x):
        next_x = self.inc(x)
        x_results = [next_x]
        for i, down_layer in enumerate(self.down_path):
            next_x = down_layer(next_x)
            x_results.append(next_x)

        up_x = x_results[(self.depth)]
        for i, up_layer in enumerate(self.up_path):
            up_x = up_layer(up_x, x_results[(self.depth - (i + 1))], self.con_operator)
        x = self.outc(up_x)
        x = self.last_sig(x)
        return x