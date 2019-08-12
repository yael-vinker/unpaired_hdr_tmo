import numpy as np
import torch
import params

import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import tranforms as transforms_
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import hdr_image_utils
import HdrImageFolder

import torchvision.datasets as dset
import gan_trainer


def print_im_details(im, title, disply=False):
    print(title)
    print("max = ", np.max(im), "min = ", np.min(im), "mean = ", np.mean(im), "dtype = ", im.dtype)
    if disply:
        plt.imshow(im)
        plt.show()


def print_batch_details(batch, title, disply=False):
    print(title)
    b_size = batch.shape[0]
    for i in range(b_size):
        im = batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        print_im_details(im, i)

def compare_tensors(b1, b2):
    b_size = b1.shape[0]
    for i in range(b_size):
        im1 = b1[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        im2 = b2[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        print(np.array_equal(im1, im2))


def test_normalize_transform(batch, device):
    print_batch_details(batch, "before")
    before_batch = batch.clone()
    normalize = transforms_.NormalizeForDisplay((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), device)
    new_batch = normalize(batch)
    print_batch_details(new_batch, "after")
    to_pil = transforms.ToPILImage()
    b_size = batch.shape[0]
    for i in range(b_size):
        new_batch[i] = to_pil(new_batch[i])
    batch_origin = F.normalize(new_batch, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
    # normalize_back = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # batch_origin = normalize_back(new_batch)
    compare_tensors(before_batch, batch_origin)



def verify_G_output_range(fake_results, min, max):
    b_size = fake_results.shape[0]
    for i in range(b_size):
        fake_copy = fake_results[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        if np.min(fake_copy) < min:
            print("Error in G output, min value is smaller than ", min)
        if np.max(fake_copy) > max:
            print("Error in G output, max value is larger than ", max)
        # print(fake_copy)
        # gray_num = (fake_copy == 0.5).sum()
        # print("Percentage of 0.5 pixels = ",gray_num / (fake_copy.shape[0] * fake_copy.shape[1]), "%")


def verify_model_load(model, model_name, optimizer):
    # Print model's state_dict
    print("State_dict for " + model_name)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name)
        # print(var_name, "\t", optimizer.state_dict()[var_name])


def model_parameters_update_test(params_befor_opt_step, model, model_name):
    params_after = list(model.parameters())[0].clone()
    is_equal = torch.equal(params_befor_opt_step.data, params_after.data)
    if is_equal:
        print("Error: parameters of model " + model_name + " remain the same")
    else:
        print(model_name + " parameters were updated successfully")


def plot_data(dataloader, device, title):
    """Plot some training images"""
    real_batch = next(iter(dataloader))
    first_b = real_batch[0].to(device)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(first_b[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def plot_npy_data(dataloader, device, title):
    """Plot some training images"""
    real_batch = next(iter(dataloader))
    first_b = real_batch[params.image_key].to(device)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(first_b[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def get_data_root():
    batch_size, num_epochs, model, G_lr, D_lr, train_data_root_npy, train_data_root_ldr, isCheckpoint_str, \
    test_data_root_npy, test_data_root_ldr, g_opt_for_single_d, \
    result_dir_pref, input_dim = gan_trainer.parse_arguments()
    # Decide which device we want to run on
    return train_data_root_npy

def transforms_test():
    transform_original = transforms.Compose([
        transforms.Resize(params.input_size),
        transforms.CenterCrop(params.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_custom = transforms.Compose([
        transforms_.Scale(params.input_size),
        transforms_.CenterCrop(params.input_size),
        transforms_.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_root = get_data_root()
    dataset_origin = dset.ImageFolder(root=data_root,
                                   transform=transform_original)

    dataset_custom = HdrImageFolder.HdrImageFolder(root=data_root,
                                      transform=transform_custom)

    original_transform_im = np.asarray(dataset_origin[0][0].permute(1, 2, 0).detach().cpu().numpy())
    custom_transform_im = np.asarray(dataset_custom[0][0].permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(original_transform_im)
    plt.show()
    plt.imshow(custom_transform_im)
    plt.show()
    print(np.array_equal(original_transform_im, custom_transform_im))
    hdr_image_utils.print_image_details(original_transform_im, "original_transform_im")
    hdr_image_utils.print_image_details(custom_transform_im, "custom_transform_im")


if __name__ == '__main__':
    transforms_test()

