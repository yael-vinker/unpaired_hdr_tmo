from torch import nn

import Blocks


class Discriminator(nn.Module):
    def __init__(self, input_size, input_dim, dim, norm):
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
        self.model += [Blocks.Conv2dBlock(dim, 1, 4, 1, 0, norm='none', activation="sigmoid")]
        # self.model += [nn.Conv2d(dim, 1, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)

        self.output_dim = dim

    def forward(self, x):
        return self.model(x)
