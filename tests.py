import numpy as np
import torch
import params
import gan_trainer_unet
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def verify_G_output_range(fake_results):
    b_size = fake_results.shape[0]
    for i in range(b_size):
        fake_copy = fake_results[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        # print("shape = ", fake_copy.shape, "  min = ", np.min(fake_copy), "  max = ", np.max(fake_copy), "  mean = ", np.mean(fake_copy),)
        if np.min(fake_copy) < 0:
            print("Error in G output, min value is smaller than zero")
        if np.max(fake_copy) > 1:
            print("Error in G output, min value is smaller than zero")
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


def load_data_test():
    batch_size, num_epochs, dataroot_npy, dataroot_ldr, \
    isCheckpoint, test_dataroot_npy, test_dataroot_ldr = \
        gan_trainer_unet.parse_arguments_2()
    torch.manual_seed(params.manualSeed)
    # Decide which device we want to run on
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", num_epochs)
    print("CHECK POINT:", isCheckpoint)
    print("DEVICE:", device)
    print("=====================\n")

    net_G = gan_trainer_unet.create_net("G", device, isCheckpoint)
    net_D = gan_trainer_unet.create_net("D", device, isCheckpoint)

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=params.lr, betas=(params.beta1, 0.999))

    gan_trainer = gan_trainer_unet.GanTrainer(device, batch_size, num_epochs, dataroot_npy, dataroot_ldr,
                                              test_dataroot_npy, test_dataroot_ldr, isCheckpoint,
                             net_G, net_D, optimizer_G, optimizer_D)

    train_npy_dataloader, train_ldr_dataloader, test_npy_dataloader, test_ldr_dataloader = \
        gan_trainer.load_data(dataroot_npy, dataroot_ldr, test_dataroot_npy, test_dataroot_ldr)

    print("[%d] images in train_npy_dset" % len(train_npy_dataloader.dataset))
    print("[%d] images in train_ldr_dset" % len(train_ldr_dataloader.dataset))
    plot_npy_data(train_npy_dataloader, device, "Train npy")
    plot_data(train_ldr_dataloader, device, "Train ldr")

    print("[%d] images in test_npy_dset" % len(test_npy_dataloader.dataset))
    print("[%d] images in test_ldr_dset" % len(test_ldr_dataloader.dataset))
    plot_npy_data(test_npy_dataloader, device, "Test npy")
    plot_data(test_ldr_dataloader, device, "test ldr")


if __name__ == '__main__':
    load_data_test()
