from torch import nn
import Blocks


class Discriminator(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, ngpu):
        super(Discriminator, self).__init__()
        self.model = []
        self.ngpu = ngpu
        self.model += [Blocks.Conv2dBlock(input_dim, dim, 4, 2, 1, norm='none', activation="leakyReLU")]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Blocks.Conv2dBlock(dim, 2 * dim, 4, 2, 1, activation="leakyReLU")]
            dim *= 2
        self.model += [Blocks.Conv2dBlock(dim, 1, 4, 1, 0, norm='none', activation="sigmoid")]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, input):
        return self.model(input)
