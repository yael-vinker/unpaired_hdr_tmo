import torch.nn as nn
import torchvision.models as models


def maybe_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor


class _netVGGFeatures(nn.Module):
    def __init__(self):
        super(_netVGGFeatures, self).__init__()
        self.vggnet = maybe_cuda(models.vgg16(pretrained=True))
        self.layer_ids = [2, 7, 12, 21, 30]

    def main(self, z, levels):
        layer_ids = self.layer_ids[:levels]
        id_max = layer_ids[-1] + 1
        output = []
        for i in range(id_max):
            z = self.vggnet.features[i](z)
            if i in layer_ids:
                output.append(z)
        return output

    def forward(self, z, levels):
        output = self.main(z, levels)
        return output


class _VGGDistance(nn.Module):
    def __init__(self, levels):
        super(_VGGDistance, self).__init__()
        self.vgg = _netVGGFeatures()
        self.levels = levels

    def forward(self, I1, I2):
        b_sz = I1.size(0)
        f1 = self.vgg(I1, self.levels)
        f2 = self.vgg(I2, self.levels)
        loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)
        for i in range(self.levels):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss
        return loss


class _VGGFixedDistance(nn.Module):
    def __init__(self):
        super(_VGGFixedDistance, self).__init__()
        self.vgg = _netVGGFeatures()
        self.up = nn.nn.functional.interpolate(size=(224, 224), align_corners=True)

    def forward(self, I1, I2):
        b_sz = I1.size(0)
        f1 = self.vgg(self.up(I1), 5)
        f2 = self.vgg(self.up(I2), 5)
        loss = 0  # torch.abs(I1 - I2).view(b_sz, -1).mean(1)
        for i in range(5):
            if i < 4:
                continue
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss
        return loss


class _VGGMSDistance(nn.Module):
    def __init__(self):
        super(_VGGMSDistance, self).__init__()
        self.vgg = _netVGGFeatures()
        self.subs = nn.AvgPool2d(4)

    def forward(self, I1, I2):
        f1 = self.vgg(I1, 5)
        f2 = self.vgg(I2, 5)
        loss = torch.abs(I1 - I2).mean()
        for i in range(5):
            layer_loss = torch.abs(f1[i] - f2[i]).mean()
            # .mean(3).mean(2).mean(0).sum()
            loss = loss + layer_loss

        f1 = self.vgg(self.subs(I1), 4)
        f2 = self.vgg(self.subs(I2), 4)
        loss = loss + torch.abs(self.subs(I1) - self.subs(I2)).mean()
        for i in range(4):
            layer_loss = torch.abs(f1[i] - f2[i]).mean()
            # .mean(3).mean(2).mean(0).sum()
            loss = loss + layer_loss

        return loss


def distance_metric(image_size, force_l2=False):
    # return vgg_metric._VGGFixedDistance()
    if force_l2:
        return maybe_cuda(nn.L1Loss())
    if image_size == 16:
        return _VGGDistance(2)
    elif image_size == 32:
        return _VGGDistance(3)
    elif image_size == 64:
        return _VGGDistance(4)
    elif image_size > 64:
        return _VGGMSDistance()


if __name__ == '__main__':
    # Toy example

    import torch
    import torch.nn as nn
    import numpy as np

    batch_size = 100
    channels = 1
    image_size = 256
    data = np.load("data/hdr_npy/hdr_npy/hdr05.npy", allow_pickle=True)
    if data.ndim == 2:
        data = data[:, :, None]
    image_tensor = torch.from_numpy(data).float()

    # 	transform_custom = transforms.Compose([
    # 		transforms_.Scale(256),
    # 		transforms_.CenterCrop(256),
    # 		transforms_.ToTensor(),
    # 		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # 	])
    # 	im2 = transform_custom(im1)
    # 	im3 = im2 / torch.max(im2)
    batch = torch.stack((image_tensor, image_tensor))
    batch2 = torch.stack((image_tensor, image_tensor))
    # 	print(batch.shape)
    # 	X1 = torch.randn(batch_size, channels, image_size, image_size)
    # 	X2 = torch.randn(batch_size, channels, image_size, image_size)
    loss_metric = distance_metric(image_size)
    loss = loss_metric(batch, batch2)
    print(loss)
# now optimize the loss with your favourite optimizer...
