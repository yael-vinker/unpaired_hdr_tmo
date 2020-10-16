# full assembly of the sub-parts to form the complete net

from .unet_parts import *
from models import Blocks
import utils.printer

class UNet(nn.Module):
    def __init__(self, n_channels, output_dim, last_layer, depth, layer_factor, con_operator, filters, bilinear,
                 network, dilation, to_crop, unet_norm, stretch_g, activation, apply_exp, doubleConvTranspose):
        super(UNet, self).__init__()
        self.to_crop = to_crop
        self.con_operator = con_operator
        self.network = network
        down_ch = filters
        self.depth = depth
        padding = 2
        if doubleConvTranspose:
            padding = 0
        self.inc = inconv(n_channels, down_ch, unet_norm, activation, padding)
        ch = down_ch
        self.down_path = nn.ModuleList()
        for i in range(self.depth - 1):
            self.down_path.append(
                down(ch, ch * 2, network, dilation=dilation, unet_norm=unet_norm, activation=activation, padding=padding)
            )
            ch = ch * 2
            if network == params.torus_network:
                dilation = dilation * 2
        if doubleConvTranspose:
            self.down_path.append(last_down(ch, ch, network, dilation=dilation, unet_norm=unet_norm, activation=activation))
        else:
            self.down_path.append(down(ch, ch, network, dilation=dilation, unet_norm=unet_norm, activation=activation, padding=padding))

        self.up_path = nn.ModuleList()
        for i in range(self.depth):
            in_ch = ch * layer_factor
            if con_operator == params.square_and_square_root_manual_d:
                in_ch += 1
            if i >= self.depth - 2:
                self.up_path.append(
                    up(in_ch, down_ch, bilinear, layer_factor, network,
                       dilation=dilation, unet_norm=unet_norm, activation=activation,
                       doubleConvTranspose=doubleConvTranspose, padding=padding)
                )
            else:
                self.up_path.append(
                    up(in_ch, ch // 2, bilinear, layer_factor, network,
                       dilation=dilation, unet_norm=unet_norm, activation=activation,
                       doubleConvTranspose=doubleConvTranspose, padding=padding)
                )
            ch = ch // 2
            if network == params.torus_network:
                dilation = dilation // 2
        self.outc = outconv(down_ch, output_dim)
        if apply_exp:
            self.exp = Blocks.Exp()
        else:
            self.exp = None
        self.last_sig = None
        if last_layer == 'tanh':
            self.last_sig = nn.Tanh()
        if last_layer == "sigmoid":
            self.last_sig = nn.Sigmoid()
        if last_layer == 'msig':
            self.last_sig = Blocks.MySig(3)
        # else:

        if stretch_g != "none":
            stretch_options = {"batchMax": Blocks.BatchMaxNormalization(),
                               "instanceMinMax": Blocks.MinMaxNormalization()}
            # self.clip = Blocks.Clip()
            self.stretch = stretch_options[stretch_g]
        else:
            self.stretch = None

    def forward(self, x, apply_crop=True):
        d_weight_mul = 1.0
        if self.con_operator == params.square_and_square_root_manual_d:
            d_weight_mul = x[0, 1, 0, 0]
        # print("d_weight_mul", d_weight_mul)
        next_x = self.inc(x)
        x_results = [next_x]
        for i, down_layer in enumerate(self.down_path):
            next_x = down_layer(next_x)
            x_results.append(next_x)

        up_x = x_results[(self.depth)]
        for i, up_layer in enumerate(self.up_path):
            up_x = up_layer(up_x, x_results[(self.depth - (i + 1))], self.con_operator, self.network, d_weight_mul)

        x_out = self.outc(up_x)
        # x_out = torch.cat([x, x_out], dim=1)
        if self.to_crop and apply_crop:
            b, c, h, w = x_out.shape
            th, tw = h - 2 * params.shape_addition, w - 2 * params.shape_addition
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            i, j, h, w = i, j, th, tw
            x_out = x_out[:, :, i: i + h, j:j + w]
        if self.last_sig is not None:
            x_out = self.last_sig(x_out)
        # if self.exp:
        #     x_out = self.exp(x_out)
        # # if self.clip:
        # #     x_out = self.clip(x_out)
        # if self.stretch:
        #     x_out = self.stretch(x_out)
        return x_out
