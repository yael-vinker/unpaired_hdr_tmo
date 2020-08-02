from torch import nn

from models import Blocks


class Discriminator(nn.Module):
    def __init__(self, input_size, input_dim, dim, norm, last_activation):
        super(Discriminator, self).__init__()
        self.model = []
        self.model += [Blocks.Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation="leakyReLU")]
        # downsampling blocks
        n_downsample = 0
        while input_size > 8:
            input_size = int(input_size / 2)
            n_downsample += 1
        for i in range(n_downsample):
            self.model += [Blocks.Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation="leakyReLU")]
            dim *= 2
        # self.model += [nn.Conv2d(dim, 1, 4, 1, 0, bias=False)]
        self.model += [Blocks.Conv2dBlock(dim, 1, 4, 1, 0, norm='none', activation=last_activation)]
        # self.model += [nn.Conv2d(dim, 1, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)

        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer="batch_norm", last_activation="none"):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                Blocks.Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                   norm=norm_layer, activation="leakyReLU")]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            Blocks.Conv2dBlock(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw,
                               norm=norm_layer, activation="leakyReLU")
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        if last_activation == "sigmoid":
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)