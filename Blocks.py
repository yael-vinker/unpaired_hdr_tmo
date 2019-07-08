from torch import nn
import torch.tensor as tensor


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='batch2d', activation="none"):
        super(Conv2dBlock, self).__init__()
        self.use_bias = False
        norm_dim = output_dim
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

        if norm == 'none':
            self.norm = None
        else:
            self.norm = nn.BatchNorm2d(norm_dim)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leakyReLU":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        y = x.float()
        out = self.conv(y)
        # if self.norm:
        #     x = self.norm(x)
        out1 = self.activation(out)
        return out1


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, activation="none"):
        super(ConvTranspose2dBlock, self).__init__()
        self.use_bias = False
        # self.pad = nn.ZeroPad2d(padding)
        norm_dim = output_dim
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)
        self.norm = nn.BatchNorm2d(norm_dim)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        norm_dim = output_dim
        self.norm = nn.BatchNorm1d(norm_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


# class Normal(object):
#     def __init__(self, mu, sigma, log_sigma, v=None, r=None):
#         self.mu = mu
#         self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
#         self.logsigma = log_sigma
#         dim = mu.get_shape()
#         if v is None:
#             v = torch.FloatTensor(*dim)
#         if r is None:
#             r = torch.FloatTensor(*dim)
#         self.v = v
#         self.r = r
