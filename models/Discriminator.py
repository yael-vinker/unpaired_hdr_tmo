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
            next_dim = min(dim * 2, 512)
            self.model += [Blocks.Conv2dBlock(dim, next_dim, 4, 2, 1, norm=norm, activation="leakyReLU")]
            dim = next_dim
            # dim *= 2
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


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_size, d_model, input_nc, ndf=64, n_layers=3, norm_layer="batch_norm",
                 last_activation="none", num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            if "dcgan" in d_model:
                netD = Discriminator(input_size, input_nc, ndf, norm_layer, last_activation)
                input_size = input_size // 2
            elif "patchD" in d_model:
                netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, last_activation)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            # model = getattr(self, 'layer' + str(num_D - 1 - i))
            model = getattr(self, 'layer' + str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""
#
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer="batch_norm", last_activation="none"):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         kw = 4
#         # import numpy as np
#         padw = 2
#         # print("padw", padw)
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#
#         nf = ndf
#         for n in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, 512)
#             sequence += [
#                 nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_prev = nf
#         nf = min(nf * 2, 512)
#         sequence += [
#             nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]
#         if last_activation == "sigmoid":
#             sequence += [nn.Sigmoid()]
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)

# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""
#
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, last_activation="none"):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         norm_layer = nn.BatchNorm2d
#         # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#         #     use_bias = norm_layer.func == nn.InstanceNorm2d
#         # else:
#         use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)