from __future__ import print_function

# import Unet_2
from torch import autograd
import ProcessedDatasetFolder
import os
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Discriminator
import params
import windows_loss_calc
import time
import hdr_image_utils
from torchsummary import summary
import Unet


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, binary_wind = sample[params.image_key], sample[params.window_image_key]
        image = image.transpose((2, 0, 1))
        # There is no need to permute "window_image" channels since it has 1 channel
        return {params.image_key: torch.from_numpy(image),
                params.window_image_key: torch.from_numpy(binary_wind)}


# TODO ask about init BatchNorm weights
def weights_init(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=params.num_epochs)
    parser.add_argument("--G_lr", type=float, default=params.lr)
    parser.add_argument("--D_lr", type=float, default=params.lr)
    parser.add_argument("--data_root_npy", type=str, default=params.dataroot_npy)
    parser.add_argument("--data_root_ldr", type=str, default=params.dataroot_ldr)
    parser.add_argument("--checkpoint", type=str, default="no")
    parser.add_argument("--test_data_root_npy", type=str, default=params.test_dataroot_npy)
    parser.add_argument("--test_data_root_ldr", type=str, default=params.test_dataroot_ldr)
    parser.add_argument("--test_red_wind", type=str, default=params.test_dataroot_red_wind_ldr)
    parser.add_argument("--apply_windows_loss", type=str, default="no")
    args = parser.parse_args()
    return args.batch, args.epochs, args.G_lr, args.D_lr, os.path.join(args.data_root_npy), os.path.join(args.data_root_ldr), \
            args.checkpoint, os.path.join(args.test_data_root_npy), os.path.join(args.test_data_root_ldr),\
                os.path.join(args.test_red_wind), args.apply_windows_loss


def create_net(net, device_, is_checkpoint):
    if net == "G":
        # Create the Generator (UNet architecture)
    # new_net = Unet_2.UNet_2().to(device_)
        new_net = Unet.UNet().to(device_)
    elif net == "D":
        # Create the Discriminator
        new_net = Discriminator.Discriminator(params.n_downsamples_d, params.input_dim, params.dim,
                                              torch.cuda.device_count()).to(device_)
    else:
        assert 0, "Unsupported network request: {}  (creates only G or D)".format(net)

    # Handle multi-gpu if desired
    if (device_.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        new_net = nn.DataParallel(new_net, list(range(torch.cuda.device_count())))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    if not is_checkpoint:
        new_net.apply(weights_init)
        print("Weights for " + net + " were initialized successfully")
    return new_net


def plot_specific_losses(loss_G_win, loss_G_d, loss_D_fake, loss_D_real, title, iters_n, path):
    plt.figure()
    plt.plot(range(iters_n), loss_G_win, '-b', label='loss G window')
    plt.plot(range(iters_n), loss_G_d, '-r', label='loss G d')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(os.path.join(path, title + "_G_loss.png"))  # should before show method
    plt.close()

    plt.figure()
    plt.plot(range(iters_n), loss_D_fake, '-g', label='loss D fake')
    plt.plot(range(iters_n), loss_D_real, '-y', label='loss D real')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(os.path.join(path, title + "_D_loss.png"))  # should before show method
    plt.close()


def plot_general_losses(loss_G, loss_D_fake, loss_D_real, title, iters_n, path):
    plt.figure()
    plt.plot(range(iters_n), loss_D_fake, '-r', label='loss D fake')
    plt.plot(range(iters_n), loss_D_real, '-b', label='loss D real')
    plt.plot(range(iters_n), loss_G, '-g', label='loss G')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(os.path.join(path, title + ".png"))  # should before show method
    plt.close()


def create_dir(model_path, loss_graph_path, result_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print("Directory ", model_path, " created")
    if not os.path.exists(loss_graph_path):
        os.mkdir(loss_graph_path)
        print("Directory ", loss_graph_path, " created")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print("Directory ", result_path, " created")


class GanTrainer:
    def __init__(self, t_device, t_batch_size, t_num_epochs, train_dataroot_npy,
                 train_dataroot_ldr, test_dataroot_npy, test_dataroot_ldr,
                 test_red_wind_data, t_isCheckpoint, t_netG, t_netD,
                 t_optimizerG, t_optimizerD, apply_G_windows_loss):
        self.batch_size = t_batch_size
        self.num_epochs = t_num_epochs
        self.device = t_device
        self.netG = t_netG
        self.netD = t_netD
        self.optimizerD = t_optimizerD
        self.optimizerG = t_optimizerG
        self.isCheckpoint = t_isCheckpoint
        self.apply_windows_loss = apply_G_windows_loss
        self.real_label, self.fake_label = 1, 0
        self.epoch, self.num_iter, self.test_num_iter = 0, 0, 0
        self.checkpoint = None
        self.errD_real, self.errD_fake, self.errG = None, None, None
        self.errGwin, self.errGd, self.errD = None, None, None
        self.G_losses, self.G_loss_window, self.G_loss_d = [], [], []
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []
        self.test_G_losses, self.test_G_loss_window, self.test_G_loss_d = [], [], []
        self.test_D_losses, self.test_D_loss_fake, self.test_D_loss_real = [], [], []
        self.train_data_loader_npy, self.train_data_loader_ldr, self.test_data_loader_npy, self.test_data_loader_ldr, \
            self.test_data_loader_red_wind = self.load_data(train_dataroot_npy, train_dataroot_ldr, test_dataroot_npy,
                                                       test_dataroot_ldr, test_red_wind_data)
        self.window_height, self.window_width = hdr_image_utils.get_window_size(params.image_size, params.image_size)
        self.half_window_height, self.half_window_width, self.quarter_height, self.quarter_width = \
            hdr_image_utils.get_half_windw_size(self.window_height, self.window_width)
        self.criterion = nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss(reduction='sum')

    def windows_loss(self, fake_images, real_hdr_images, windows_im):
        """
        Calculates the loss of the generator by comparing windows from G's output with windows
        extracted by our algorithm and were tone mapped by scalar division.
        The loss is calculated by: sum(L2 for each window) / number of windows.
        :param fake_images: batch of Variables which are the outputs of netG on the given "real_images".
        Each image is RGB, in range [0,1].
        :param real_hdr_images: batch of Tensors which are HDR images (RGB) in range [0,1].
        :param windows_im: batch of Tensors binary images that contains the centers of the ldr windows.
        :return: Tensor(int), the loss of the entire batch
        """
        b_size = fake_images.shape[0]
        loss = windows_loss_calc.run_all(real_hdr_images, fake_images, self.mse_loss,
                                         self.window_height, self.window_width,
                                         self.half_window_height, self.half_window_width,
                                         windows_im)
        return loss / b_size

    def save_model(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'modelD_state_dict': self.netD.state_dict(),
            'modelG_state_dict': self.netG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
        }, path)

    def load_model(self):
        if self.isCheckpoint:
            self.checkpoint = torch.load(params.models_save_path)
            self.epoch = self.checkpoint['epoch']
            self.netD.load_state_dict(self.checkpoint['modelD_state_dict'])
            self.netG.load_state_dict(self.checkpoint['modelG_state_dict'])
            self.optimizerD.load_state_dict(self.checkpoint['optimizerD_state_dict'])
            self.optimizerG.load_state_dict(self.checkpoint['optimizerG_state_dict'])
            self.netD.train()
            self.netG.train()

    def load_npy_data(self, npy_data_root, shuffle, batch_size):
        npy_dataset = ProcessedDatasetFolder.ProcessedDatasetFolder(root=npy_data_root,
                                                                    transform=transforms.Compose([
                                                                        ToTensor(),
                                                                    ]))
        dataloader = torch.utils.data.DataLoader(npy_dataset, batch_size=batch_size,
                                                 shuffle=shuffle, num_workers=params.workers)
        return dataloader

    def load_ldr_data(self, ldr_data_root, shuffle, batch_size):
        ldr_dataset = dset.ImageFolder(root=ldr_data_root,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                       ]))

        ldr_dataloader = torch.utils.data.DataLoader(ldr_dataset, batch_size=batch_size,
                                                     shuffle=shuffle, num_workers=params.workers)
        return ldr_dataloader

    def load_data(self, train_root_npy, train_root_ldr, test_root_npy, test_root_ldr, test_root_red_wind_ldr):
        """
        :param isHdr: True if images in "dir_root" are in .hdr format, False otherwise.
        :param dir_root: path to wanted directory
        :param b_size: batch size
        :return: DataLoader object of images in "dir_root"
        """
        train_npy_dataloader = self.load_npy_data(train_root_npy, True, self.batch_size)
        test_npy_dataloader = self.load_npy_data(test_root_npy, False, 24)
        train_ldr_dataloader = self.load_ldr_data(train_root_ldr, True, self.batch_size)
        test_ldr_dataloader = self.load_ldr_data(test_root_ldr, False, 24)
        test_ldr_red_wind_data = self.load_ldr_data(test_root_red_wind_ldr, False, 24)

        print("[%d] images in train_npy_dataset" % len(train_npy_dataloader.dataset))
        print("[%d] images in test_npy_dataset" % len(test_npy_dataloader.dataset))
        print("[%d] images in train_ldr_dataloader" % len(train_ldr_dataloader.dataset))
        print("[%d] images in test_npy_dset" % len(test_ldr_dataloader.dataset))
        print("[%d] images in test_red_wind" % len(test_ldr_red_wind_data.dataset))

        return train_npy_dataloader, train_ldr_dataloader, test_npy_dataloader, test_ldr_dataloader, test_ldr_red_wind_data

    def train_D(self, real_hdr_cpu, real_ldr_cpu, label):
        """
        Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        :param real_hdr_cpu: HDR images as input to G to generate fake data
        :param real_ldr_cpu: LDR images as real input to D
        :param label: Tensor contains real labels for first loss
        :return: fake (Tensor) result of G on hdr_data, for G train
        """
        # Train with all-real batch
        self.netD.zero_grad()
        # Forward pass real batch through D
        output_on_real = self.netD(real_ldr_cpu).view(-1)
        # Calculate loss on all-real batch
        self.errD_real = self.criterion(output_on_real, label)
        self.errD_real.backward()

        # Train with all-fake batch
        # Generate fake image batch with G
        fake = self.netG(real_hdr_cpu)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output_on_fake = self.netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        self.errD_fake = self.criterion(output_on_fake, label)
        # Calculate the gradients for this batch
        self.errD_fake.backward()
        # Add the gradients from the all-real and all-fake batches
        self.errD = self.errD_real + self.errD_fake
        # Update D
        self.optimizerD.step()
        self.D_losses.append(self.errD.item())
        self.D_loss_fake.append(self.errD_fake.item())
        self.D_loss_real.append(self.errD_real.item())
        return fake

    def train_G(self, label, fake, real_hdr_cpu, windows_im):
        """
        Update G network: maximize log(D(G(z))) and minimize loss_wind
        :param label: Tensor contains real labels for first loss
        :param fake: (Tensor) result of G on hdr_data
        :param real_hdr_cpu: HDR images as input to windows_loss
        """
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_on_fake = self.netD(fake).view(-1)
        self.errG = self.criterion(output_on_fake, label)
        self.errG.backward()
        self.optimizerG.step()
        self.G_losses.append(self.errG.item())


    def train_G_wind_loss(self, label, fake, real_hdr_cpu, windows_im):
        """
        Update G network: maximize log(D(G(z))) and minimize loss_wind
        :param label: Tensor contains real labels for first loss
        :param fake: (Tensor) result of G on hdr_data
        :param real_hdr_cpu: HDR images as input to windows_loss
        """
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_on_fake = self.netD(fake).view(-1)
        self.errGd = self.criterion(output_on_fake, label)
        self.errGwin = self.windows_loss(fake, real_hdr_cpu.detach(), windows_im.detach())
        self.errG = self.errGwin + (params.g_d_loss_factor * self.errGd)
        self.errG.backward()
        self.optimizerG.step()
        self.G_losses.append(self.errG.item())
        self.G_loss_window.append(self.errGwin.item())
        self.G_loss_d.append(self.errGd.item())


    def train_epoch(self):
        for (h, data_hdr), (l, data_ldr) in zip(enumerate(self.train_data_loader_npy, 0),
                                                enumerate(self.train_data_loader_ldr, 0)):
            batch_start_time = time.time()
            self.num_iter += 1
            with autograd.detect_anomaly():
                real_ldr_cpu = data_ldr[0].to(self.device)
                b_size = real_ldr_cpu.size(0)
                label = torch.full((b_size,), self.real_label, device=self.device)
                real_hdr_cpu = data_hdr[params.image_key].to(self.device)
                windows_im = data_hdr[params.window_image_key].to(self.device)

                fake = self.train_D(real_hdr_cpu, real_ldr_cpu, label)
                if self.apply_windows_loss:
                    self.train_G_wind_loss(label, fake, real_hdr_cpu, windows_im)
                else:
                    self.train_G(label, fake, real_hdr_cpu, windows_im)
            batch_end_time = time.time()
            print("Single [batch] iteration took [%.4f] seconds" % (batch_end_time - batch_start_time))

    def print_cuda_details(self):
        if (self.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            print("Using [%d] GPUs" % torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print("GPU [%d] device name = %s" % (i, torch.cuda.get_device_name(i)))
            print(torch.cuda.current_device())

    def verify_checkpoint(self):
        if isCheckpoint:
            print("Loading model...")
            self.load_model()
            print("Model was loaded")
            print()

    def print_epoch_losses_summary(self, epoch):
        if self.apply_windows_loss:
            # print("self.apply_windows_loss:", self.apply_windows_loss)
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_G_window: %.4f\tLoss_G_d_factorized: %.4f \tLoss_G_d: %.4f'
              % (epoch, self.num_epochs, self.errD.item(), self.errG.item(), self.errGwin.item(),
                 self.errGd.item() * params.g_d_loss_factor, self.errGd.item()))
        else:
            print('[%d/%d]\tLoss_D: %.4f \tLoss_D_real: %.4f \tLoss_D_fake: %.4f \tLoss_G: %.4f'
                  % (epoch, self.num_epochs, self.errD.item(), self.errD_real.item(), self.errD_fake.item(), self.errG.item()))

    def save_loss_plot(self, epoch):
        plot_general_losses(self.G_losses, self.D_loss_fake, self.D_loss_real, "summary epoch_=_" + str(epoch),
                            self.num_iter,
                            params.loss_path)
        if self.apply_windows_loss:
            plot_specific_losses(self.G_loss_window, self.G_loss_d, self.D_loss_fake, self.D_loss_real,
                             "detailed loss, epoch = " + str(epoch), self.num_iter, params.loss_path)

    def train(self):
        self.print_cuda_details()
        self.verify_checkpoint()

        print("Starting Training Loop...")
        for epoch in range(self.epoch, self.num_epochs):
            start = time.time()
            self.train_epoch()
            print("Single [[epoch]] iteration took [%.4f] seconds\n" % (time.time() - start))
            self.save_model(params.models_save_path, epoch)
            self.print_epoch_losses_summary(epoch)
            # self.save_results_plot(epoch, params.results_path)
            if epoch % 5 == 0:
                self.save_results_plot(epoch, params.results_path)

            if epoch % 20 == 0:
                self.save_loss_plot(epoch)

    def save_groups_images(self, first_b_tonemap, fake, red_wind, new_out_dir):
        # for i in range(6):
        #     plt.figure(figsize=(15, 15))
        #     plt.subplot(3, 1, 1)
        #     plt.axis("off")
        #     plt.title("Real Images")
        #     plt.imshow(
        #         np.transpose(vutils.make_grid(first_b_tonemap[i * 4: (i + 1) * 4], padding=5).cpu(), (1, 2, 0)))
        #
        #     img_list1 = [vutils.make_grid(red_wind[i * 4: (i + 1) * 4], padding=5)]
        #     plt.subplot(3, 1, 2)
        #     plt.axis("off")
        #     plt.title("Processed Images")
        #     plt.imshow(np.transpose(img_list1[-1].cpu(), (1, 2, 0)))
        #
        #     img_list2 = [vutils.make_grid(fake[i * 4: (i + 1) * 4], padding=5)]
        #     plt.subplot(3, 1, 3)
        #     plt.axis("off")
        #     plt.title("Fake Images")
        #     plt.imshow(np.transpose(img_list2[-1].cpu(), (1, 2, 0)))
        #     plt.savefig(os.path.join(new_out_dir, "set " + str(i)))
        #     plt.close()
        for i in range(6):
            plt.figure(figsize=(15, 15))
            plt.subplot(2, 1, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(
                np.transpose(vutils.make_grid(first_b_tonemap[i * 4: (i + 1) * 4], padding=5).cpu(), (1, 2, 0)))

            # img_list1 = [vutils.make_grid(red_wind[i * 4: (i + 1) * 4], padding=5)]
            # plt.subplot(3, 1, 2)
            # plt.axis("off")
            # plt.title("Processed Images")
            # plt.imshow(np.transpose(img_list1[-1].cpu(), (1, 2, 0)))

            img_list2 = [vutils.make_grid(fake[i * 4: (i + 1) * 4], padding=5)]
            plt.subplot(2, 1, 2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list2[-1].cpu(), (1, 2, 0)))
            plt.savefig(os.path.join(new_out_dir, "set " + str(i)))
            plt.close()

    def update_test_loss(self, b_size, first_b_tonemap, fake, epoch, test_hdr_image, test_windows_im):
        with torch.no_grad():
            real_label = torch.full((b_size,), self.real_label, device=self.device)
            test_D_output_on_real = self.netD(first_b_tonemap.detach()).view(-1)
            test_errD_real = self.criterion(test_D_output_on_real, real_label)
            self.test_D_loss_real.append(test_errD_real.item())

            fake_label = torch.full((b_size,), self.fake_label, device=self.device)
            output_on_fake = self.netD(fake.detach()).view(-1)
            test_errD_fake = self.criterion(output_on_fake, fake_label)
            test_loss_D = test_errD_real + test_errD_fake
            self.test_D_loss_fake.append(test_errD_fake.item())
            self.test_D_losses.append(test_loss_D.item())

            output_on_fake = self.netD(fake.detach()).view(-1)
            test_errGd = self.criterion(output_on_fake, real_label)
            if self.apply_windows_loss:

                test_errGwin = self.windows_loss(fake.detach(), test_hdr_image.detach(), test_windows_im.detach())
                self.test_G_loss_d.append(test_errGd.item())
                self.test_G_loss_window.append(test_errGwin.item())
                self.test_G_losses.append((test_errGd + test_errGwin).item())
                print("===== Test results =====")
                print(
                    '[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_G_window: %.4f\tLoss_G_d_factorized: %.4f \tLoss_G_d: %.4f'
                    % (
                    epoch, self.num_epochs, test_loss_D, test_errGd, test_errGwin, test_errGd * params.g_d_loss_factor,
                    test_errGd))
                plot_specific_losses(self.test_G_loss_window, self.test_G_loss_d, self.test_D_loss_fake,
                                     self.test_D_loss_real,
                                     "TEST detailed loss, epoch = " + str(epoch), self.test_num_iter, params.loss_path)
            else:
                self.test_G_losses.append(test_errGd.item())
                print("===== Test results =====")
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                    % (epoch, self.num_epochs, test_loss_D, test_errGd))
            plot_general_losses(self.test_G_losses, self.test_D_loss_fake, self.test_D_loss_real,
                                "TEST epoch = " + str(epoch),
                                self.test_num_iter,
                                params.loss_path)


    def get_fake_test_images(self, first_b_hdr):
        with torch.no_grad():
            fake = self.netG(first_b_hdr)
            return fake

    def save_results_plot(self, epoch, out_dir):
        new_out_dir = os.path.join(out_dir, "images_epoch=" + str(epoch))
        if not os.path.exists(new_out_dir):
            os.mkdir(new_out_dir)

        self.test_num_iter += 1
        test_real_batch_tonemap = next(iter(self.test_data_loader_ldr))
        test_first_b_tonemap = test_real_batch_tonemap[0].to(device)

        test_red_wind_batch = next(iter(self.test_data_loader_red_wind))
        test_first_b_red_wind = test_red_wind_batch[0].to(device)

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(test_first_b_tonemap[:25], padding=5).cpu(), (1, 2, 0)))

        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(test_first_b_tonemap[:2], padding=5).cpu(), (1, 2, 0)))

        test_real_batch_hdr = next(iter(self.test_data_loader_npy))
        test_hdr_image = test_real_batch_hdr[params.image_key].to(self.device)

        fake = self.get_fake_test_images(test_hdr_image)
        b_size = test_first_b_tonemap.size(0)
        test_windows_im = test_real_batch_hdr[params.window_image_key].to(self.device)
        self.update_test_loss(b_size, test_first_b_tonemap, fake, epoch, test_hdr_image, test_windows_im)

        fake = fake.detach().cpu()
        img_list1 = [vutils.make_grid(fake[:25], padding=5)]
        img_list2 = [vutils.make_grid(fake[:2], padding=5)]
        plt.subplot(2, 2, 3)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list1[-1].cpu(), (1, 2, 0)))

        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list2[-1].cpu(), (1, 2, 0)))
        plt.savefig(os.path.join(new_out_dir, "ALL epoch = " + str(epoch)))
        plt.close()
        self.save_groups_images(test_first_b_tonemap, fake, test_first_b_red_wind, new_out_dir)



if __name__ == '__main__':
    batch_size, num_epochs, G_lr, D_lr, train_data_root_npy, train_data_root_ldr, isCheckpoint_str, \
        test_data_root_npy, test_data_root_ldr, test_red_wind_data, apply_windows_loss_str = parse_arguments()
    torch.manual_seed(params.manualSeed)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

    isCheckpoint = True
    if isCheckpoint_str == 'no':
        isCheckpoint = False

    apply_windows_loss = True
    if apply_windows_loss_str == 'no':
        apply_windows_loss = False

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", num_epochs)
    print("G LR: ", G_lr)
    print("D LR: ", D_lr)
    print("CHECK POINT:", isCheckpoint)
    print("APPLY WINDOWS LOSS:", apply_windows_loss)
    print("DEVICE:", device)
    print("=====================\n")

    net_G = create_net("G", device, isCheckpoint)
    print("=================  NET G  ==================")
    print(net_G)
    summary(net_G, (3, 256, 256))
    print()

    net_D = create_net("D", device, isCheckpoint)
    print("=================  NET D  ==================")
    print(net_D)
    summary(net_D, (3, 256, 256))
    print()

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=D_lr, betas=(params.beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=G_lr, betas=(params.beta1, 0.999))

    create_dir(params.models_save_path, params.loss_path, params.results_path)

    gan_trainer = GanTrainer(device, batch_size, num_epochs, train_data_root_npy, train_data_root_ldr,
                             test_data_root_npy, test_data_root_ldr, test_red_wind_data, isCheckpoint,
                             net_G, net_D, optimizer_G, optimizer_D, apply_windows_loss)

    gan_trainer.train()
