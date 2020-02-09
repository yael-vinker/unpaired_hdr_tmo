from torch import nn
import torch

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='none', activation="none"):
        super(Conv2dBlock, self).__init__()
        self.use_bias = False
        norm_dim = output_dim
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

        if norm == 'none':
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.InstanceNorm2d(norm_dim)
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
        if self.norm:
            out = self.norm(out)
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

class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)

class Clip(nn.Module):
    def __init__(self):
        super(Clip, self).__init__()

    def forward(self, x):
        x = x * 1.05
        return torch.clamp(x, min=0.0, max=1.0)



#
# class LambdaLR:
#     def __init__(self, n_epochs, offset, decay_start_epoch):
#         assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
#         self.n_epochs = n_epochs
#         self.offset = offset
#         self.decay_start_epoch = decay_start_epoch
#
#     def step(self, epoch):
#         return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
#
