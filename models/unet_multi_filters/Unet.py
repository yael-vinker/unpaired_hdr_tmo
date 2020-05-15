# full assembly of the sub-parts to form the complete net

from .unet_parts import *
from models import Blocks
import utils.printer

class UNet(nn.Module):
    def __init__(self, n_channels, output_dim, last_layer, depth, layer_factor, con_operator, filters, bilinear,
                 network, dilation, to_crop, unet_norm, add_clipping, normalization, apply_exp):
        super(UNet, self).__init__()
        self.to_crop = to_crop
        self.con_operator = con_operator
        self.network = network
        down_ch = filters
        self.depth = depth
        self.inc = inconv(n_channels, down_ch, unet_norm)
        ch = down_ch
        self.down_path = nn.ModuleList()
        for i in range(self.depth - 1):
            self.down_path.append(
                down(ch, ch * 2, network, dilation=dilation, unet_norm=unet_norm)
            )
            ch = ch * 2
            if network == params.torus_network:
                dilation = dilation * 2
        self.down_path.append(down(ch, ch, network, dilation=dilation, unet_norm=unet_norm))

        self.up_path = nn.ModuleList()
        for i in range(self.depth):
            if i >= self.depth - 2:
                self.up_path.append(
                    up(ch * layer_factor, down_ch, bilinear, layer_factor, network, dilation=dilation, unet_norm=unet_norm)
                )
            else:
                self.up_path.append(
                    up(ch * layer_factor, ch // 2, bilinear, layer_factor, network, dilation=dilation, unet_norm=unet_norm)
                )
            ch = ch // 2
            if network == params.torus_network:
                dilation = dilation // 2
        self.outc = outconv(down_ch, output_dim)
        if apply_exp:
            self.exp = Blocks.Exp()
        else:
            self.exp = None
        if last_layer == 'tanh':
            self.last_sig = nn.Tanh()
        if last_layer == 'sigmoid':
            self.last_sig = nn.Sigmoid()
        if last_layer == 'msig':
            self.last_sig = Blocks.MySig(3)
        if add_clipping:
            self.clip = Blocks.Clip()
        else:
            self.clip = None

    def forward(self, x):
        next_x = self.inc(x)
        x_results = [next_x]
        for i, down_layer in enumerate(self.down_path):
            next_x = down_layer(next_x)
            x_results.append(next_x)

        up_x = x_results[(self.depth)]
        for i, up_layer in enumerate(self.up_path):
            up_x = up_layer(up_x, x_results[(self.depth - (i + 1))], self.con_operator, self.network)
        x_out = self.outc(up_x)
        # x_out = torch.cat([x, x_out], dim=1)
        if self.to_crop:
            b, c, h, w = x_out.shape
            th, tw = h - 2 * params.shape_addition, w - 2 * params.shape_addition
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            i, j, h, w = i, j, th, tw
            x_out = x_out[:, :, i: i + h, j:j + w]
        if self.last_sig:
            x_out = self.last_sig(x_out)
        if self.exp:
            x_out = self.exp(x_out)
        if self.clip:
            x_out = self.clip(x_out)
        return x_out
