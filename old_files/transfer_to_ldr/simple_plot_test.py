import argparse
import os
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.utils as vutils
import torchvision.transforms as transforms
from old_files import HdrImageFolder
import torchvision.datasets as dset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--in_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--data_root_hdr", type=str, default="")
    parser.add_argument("--data_root_ldr", type=str, default="")
    args = parser.parse_args()
    print("in root = ", args.in_dir)
    print("out root = ", args.out_dir)
    print("hdr root = ", args.data_root_hdr)
    print("ldr root = ", args.data_root_ldr)
    return args.in_dir, args.out_dir, args.data_root_hdr, args.data_root_ldr


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_data(isHdr, dir_root, b_size):
    if isHdr:
        dataset = HdrImageFolder.HdrImageFolder(root=dir_root,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                ]))
    else:
        dataset = dset.ImageFolder(root=dir_root,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.ToTensor(),
                                    ]))

    print("number of images in ",dir_root, " = ", len(dataset))
    return torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                       shuffle=True, num_workers=2)

def save_results(epoch, device, dataroot_hdr, dataroot_ldr, out_dir):
    dataloader_hdr = get_data(True, dataroot_hdr, 6)
    dataloader_ldr = get_data(False, dataroot_ldr, 6)
    print("IMAGES WERE LOADED")
    # Grab a batch of real images from the dataloader_hdr
    real_batch = next(iter(dataloader_hdr))
    first_b = real_batch[0].to(device)
    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(first_b[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    # with torch.no_grad():
        # fake = self.netG(first_b).detach().cpu()
    fake = next(iter(dataloader_ldr))[0].to(device)
    img_list1 = []
    img_list1.append(vutils.make_grid(fake[:64], padding=2, normalize=True))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list1[-1].cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(out_dir, "epoch = " + str(epoch)))
    print("IMAGES WERE SAVED AT : ", os.path.join(out_dir, "epoch = " + str(epoch)))
    plt.close()


if __name__ == '__main__':
    in_root, out_root, root_hdr, root_ldr = parse_arguments()
    device1 = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    save_results(5, device1, root_hdr, root_ldr, out_root)

