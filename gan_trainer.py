from __future__ import print_function


import unet.Unet as Unet
import VAE
from torch import autograd
import os
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Discriminator
import params
import time
import hdr_image_utils
from torchsummary import summary
import tranforms as transforms_
import LdrDatasetFolder
import HdrImageFolder
import gan_trainer_utils as g_t_utils
import torchvision.datasets as dset

# TODO ask about init BatchNorm weights
def weights_init(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=params.num_epochs)
    parser.add_argument("--model", type=str, default="skip_connection")
    parser.add_argument("--G_lr", type=float, default=params.lr)
    parser.add_argument("--D_lr", type=float, default=params.lr)
    parser.add_argument("--data_root_npy", type=str, default=params.train_dataroot_hdr)
    parser.add_argument("--data_root_ldr", type=str, default=params.train_dataroot_ldr)
    parser.add_argument("--checkpoint", type=str, default="no")
    parser.add_argument("--test_data_root_npy", type=str, default=params.test_dataroot_hdr)
    parser.add_argument("--test_data_root_ldr", type=str, default=params.test_dataroot_ldr)
    parser.add_argument("--G_opt_for_single_D", type=int, default=1)
    parser.add_argument("--result_dir_prefix", type=str, default="local")
    parser.add_argument("--input_dim", type=int, default=3)
    args = parser.parse_args()
    return args.batch, args.epochs, args.model, args.G_lr, args.D_lr, os.path.join(args.data_root_npy), \
           os.path.join(args.data_root_ldr), args.checkpoint, os.path.join(args.test_data_root_npy), \
           os.path.join(args.test_data_root_ldr), \
           args.G_opt_for_single_D, args.result_dir_prefix, args.input_dim


def create_net(net, device_, is_checkpoint, input_dim):
    if net == "G_VAE":
        # Create the Generator (UNet architecture)
        new_net = VAE.VAE(input_dim).to(device_)
    elif net == "G_skip_connection":
        # new_net = UnetSkipConnection.UNetSkipConnection(input_dim).to(device_)
        new_net = Unet.UNet(input_dim, input_dim).to(device_)
    elif net == "D":
        # Create the Discriminator
        new_net = Discriminator.Discriminator(params.input_size, input_dim, params.dim,
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


class GanTrainer:
    def __init__(self, t_device, t_batch_size, t_num_epochs, train_dataroot_npy,
                 train_dataroot_ldr, test_dataroot_npy, test_dataroot_ldr, t_isCheckpoint, t_netG, t_netD,
                 t_optimizerG, t_optimizerD, input_dim):
        self.batch_size = t_batch_size
        self.num_epochs = t_num_epochs
        self.device = t_device
        self.netG = t_netG
        self.netD = t_netD
        self.optimizerD = t_optimizerD
        self.optimizerG = t_optimizerG
        self.criterion = nn.BCELoss()

        self.real_label, self.fake_label = 1, 0
        self.epoch, self.num_iter, self.test_num_iter = 0, 0, 0

        self.errD_real, self.errD_fake, self.errG = None, None, None
        self.errGd, self.errD = None, None
        self.accG, self.accD, self.accDreal, self.accDfake = None, None, None, None
        self.accG_test, self.accD_test, self.accDreal_test, self.accDfake_test = None, None, None, None
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.G_accuracy_test, self.D_accuracy_real_test, self.D_accuracy_fake_test = [], [], []
        self.G_losses, self.G_loss_window, self.G_loss_d = [], [], []
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []
        self.test_G_losses, self.test_G_loss_window, self.test_G_loss_d = [], [], []
        self.test_D_losses, self.test_D_loss_fake, self.test_D_loss_real = [], [], []

        self.train_data_loader_npy, self.train_data_loader_ldr, self.test_data_loader_npy, self.test_data_loader_ldr = \
            self.load_data(train_dataroot_npy, train_dataroot_ldr, test_dataroot_npy, test_dataroot_ldr,
                           input_dim, testMode=False)

        self.isCheckpoint = t_isCheckpoint
        self.checkpoint = None



    def load_npy_data(self, npy_data_root, shuffle, batch_size, input_dim, trainMode):
        npy_dataset = HdrImageFolder.HdrImageFolder(root=npy_data_root, input_dim=input_dim, trainMode=trainMode,
                                                            transform=transforms.Compose([
                                                                transforms_.Scale(params.input_size),
                                                                transforms_.CenterCrop(params.input_size),
                                                                transforms_.ToTensor(),
                                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                    ]))
        # npy_dataset = dset.ImageFolder(root=npy_data_root,
        #                            transform=transforms.Compose([
        #                                transforms.Resize(params.input_size),
        #                                transforms.CenterCrop(params.input_size),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                            ]))
        dataloader = torch.utils.data.DataLoader(npy_dataset, batch_size=batch_size,
                                                 shuffle=shuffle, num_workers=params.workers)
        return dataloader

    def load_ldr_data(self, ldr_data_root, shuffle, batch_size, input_dim, trainMode):
        ldr_dataset = LdrDatasetFolder.LdrDatasetFolder(root=ldr_data_root, input_dim=input_dim, trainMode=trainMode,
                                       transform=transforms.Compose([
                                           transforms_.Scale(params.input_size),
                                           transforms_.CenterCrop(params.input_size),
                                           transforms_.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        # ldr_dataset = dset.ImageFolder(root=ldr_data_root,
        #                                transform=transforms.Compose([
        #                                    transforms.Resize(params.input_size),
        #                                    transforms.CenterCrop(params.input_size),
        #                                    transforms.ToTensor(),
        #                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                                    # transforms.Grayscale(num_output_channels=1),
        #                                ]))

        ldr_dataloader = torch.utils.data.DataLoader(ldr_dataset, batch_size=batch_size,
                                                     shuffle=shuffle, num_workers=params.workers)
        return ldr_dataloader

    def load_data(self, train_root_npy, train_root_ldr, test_root_npy, test_root_ldr,
                  input_dim, testMode, images_number=4):
        """
        :param isHdr: True if images in "dir_root" are in .hdr format, False otherwise.
        :param dir_root: path to wanted directory
        :param b_size: batch size
        :return: DataLoader object of images in "dir_root"
        """
        train_npy_dataloader = self.load_npy_data(train_root_npy, True, self.batch_size, input_dim, trainMode=True)
        hdr_train_sample = g_t_utils.get_single_hdr_im(train_root_npy, images_number)

        # test_npy_dataloader = self.load_npy_data(test_root_npy, False, 24, trainMode=False)
        test_npy_dataloader = self.load_npy_data(test_root_npy, False, 24, input_dim, trainMode=True)
        hdr_test_sample = g_t_utils.get_single_hdr_im(test_root_npy, images_number)

        # half_batch_size = int(self.batch_size / 2)
        half_batch_size = self.batch_size
        train_ldr_dataloader = self.load_ldr_data(train_root_ldr, True, half_batch_size, input_dim, trainMode=True)
        ldr_train_sample = g_t_utils.get_single_ldr_im(train_root_ldr, images_number)

        # test_ldr_dataloader = self.load_ldr_data(test_root_ldr, False, 24, trainMode=False)
        test_ldr_dataloader = self.load_ldr_data(test_root_ldr, False, 24, input_dim, trainMode=True)
        ldr_test_sample = g_t_utils.get_single_ldr_im(test_root_ldr, images_number)

        g_t_utils.print_dataset_details(images_number, train_npy_dataloader, hdr_train_sample)
        g_t_utils.print_dataset_details(images_number, test_npy_dataloader, hdr_test_sample)
        g_t_utils.print_dataset_details(images_number, train_ldr_dataloader, ldr_train_sample)
        g_t_utils.print_dataset_details(images_number, test_ldr_dataloader, ldr_test_sample)

        if testMode:
            g_t_utils.load_data_test_mode(train_npy_dataloader, train_ldr_dataloader, test_npy_dataloader, test_ldr_dataloader)
        return train_npy_dataloader, train_ldr_dataloader, test_npy_dataloader, test_ldr_dataloader

    def update_accuracy(self, isTest=False):
        len_hdr_train_dset = len(self.train_data_loader_npy.dataset)
        len_hdr_test_dset = len(self.test_data_loader_npy.dataset)
        len_ldr_train_dset = len(self.train_data_loader_ldr.dataset)
        len_ldr_test_dset = len(self.test_data_loader_ldr.dataset)

        test_smaller_len = len_hdr_test_dset if len_hdr_test_dset < len_ldr_test_dset else len_ldr_test_dset
        train_smaller_len = len_hdr_train_dset if len_hdr_train_dset < len_ldr_train_dset else len_ldr_train_dset

        if isTest:
            self.accG_test = self.accG_counter / len_hdr_test_dset
            self.accDreal_test = self.accDreal_counter / test_smaller_len
            self.accDfake_test = self.accDfake_counter / test_smaller_len
            print("accG_test ", self.accG_test ,"=", "accG_counter ", self.accG_counter, "/ len_hdr_test_dset",  len_hdr_test_dset)
            self.accDreal_test = self.accDreal_counter / test_smaller_len
            print("accDreal_test ", self.accDreal_test, "=", "accDreal_counter ", self.accDreal_counter, "/ test_smaller_len",
                  test_smaller_len)
            self.accDfake_test = self.accDfake_counter / test_smaller_len
            print("accDfake_test ", self.accDfake_test, "=", "accDfake_counter ", self.accDfake_counter,
                  "/ test_smaller_len",
                  test_smaller_len)
            self.G_accuracy_test.append(self.accG_test)
            self.D_accuracy_real_test.append(self.accDreal_test)
            self.D_accuracy_fake_test.append(self.accDfake_test)
        else:
            self.accG = self.accG_counter / len_hdr_train_dset
            self.accDreal = self.accDreal_counter / len_ldr_train_dset
            self.accDfake = self.accDfake_counter / len_ldr_train_dset
            self.G_accuracy.append(self.accG)
            self.D_accuracy_real.append(self.accDreal)
            self.D_accuracy_fake.append(self.accDfake)

    def train_D(self, hdr_input, real_ldr_cpu, half_batch_size):
        """
        Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        :param real_hdr_cpu: HDR images as input to G to generate fake data
        :param real_ldr_cpu: LDR images as real input to D
        :param label: Tensor contains real labels for first loss
        :return: fake (Tensor) result of G on hdr_data, for G train
        """
        label = torch.full((half_batch_size,), self.real_label, device=self.device)
        # Train with all-real batch
        self.netD.zero_grad()
        # Forward pass real batch through D
        # real_ldr_half_batch = real_ldr_cpu[0: half_batch_size]
        # print("real_ldr ",real_ldr_cpu.shape)
        output_on_real = self.netD(real_ldr_cpu).view(-1)
        # output_on_real = self.netD(real_ldr_half_batch).view(-1)
        # Real label = 1, so we count the samples on which D was right
        self.accDreal_counter += (output_on_real > 0.5).sum().item()
        # self.D_accuracy_real.append(self.accDreal.item())

        # Calculate loss on all-real batch
        self.errD_real = self.criterion(output_on_real, label)
        # self.errD_real = self.custom_loss(output_on_real, label)
        self.errD_real.backward()

        # Train with all-fake batch
        # Generate fake image batch with G
        fake = self.netG(hdr_input)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output_on_fake = self.netD(fake.detach()).view(-1)
        # Fake label = 0, so we count the samples on which D was right
        self.accDfake_counter += (output_on_fake <= 0.5).sum().item()
        # self.D_accuracy_fake.append(self.accDfake.item())
        # Calculate D's loss on the all-fake batch
        # self.errD_fake = self.custom_loss(output_on_fake, label)
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

    def train_G(self, label, hdr_input):
        """
        Update G network: maximize log(D(G(z))) and minimize loss_wind
        :param label: Tensor contains real labels for first loss
        :param fake: (Tensor) result of G on hdr_data
        :param real_hdr_cpu: HDR images as input to windows_loss
        """
        # for step in range(self.g_opt_for_single_d):
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        fake = self.netG(hdr_input)
        fake_single = np.asarray(fake[0].cpu().detach())
        print("fake --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]" %
              (float(np.max(fake_single)), float(np.min(fake_single)),
               fake_single.dtype, str(fake_single.shape)))

        output_on_fake = self.netD(fake).view(-1)
        # Real label = 1, so wo count number of samples on which G tricked D
        self.accG_counter += (output_on_fake > 0.5).sum().item()
        # self.G_accuracy.append(self.accG.item())
        self.errG = self.criterion(output_on_fake, label)
        # self.errG = self.custom_loss(output_on_fake, label)
        self.errG.backward()
        self.optimizerG.step()
        self.G_losses.append(self.errG.item())

    def train_epoch(self):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        for (h, data_hdr), (l, data_ldr) in zip(enumerate(self.train_data_loader_npy, 0),
                                                enumerate(self.train_data_loader_ldr, 0)):
            start = time.time()
            self.num_iter += 1
            with autograd.detect_anomaly():
                real_ldr_cpu = data_ldr[0].to(self.device)
                # hdr_input = data_hdr[params.image_key].to(self.device)
                hdr_input = data_hdr[0].to(self.device)
                b_size = hdr_input.size(0)
                label = torch.full((b_size,), self.real_label, device=self.device)

                # half_batch_size = int(self.batch_size / 2)
                half_batch_size = self.batch_size
                self.train_D(hdr_input, real_ldr_cpu, half_batch_size)
                self.train_G(label, hdr_input)
            # print("Single [batch] iteration took [%.4f] seconds" % (time.time() - start))
        print("num iters = ", self.num_iter)
        self.update_accuracy()

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

    def print_test_epoch_losses_summary(self, epoch, test_loss_D, test_errGd):
        print("===== Test results =====")
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (epoch, self.num_epochs, test_loss_D, test_errGd))
        print('[%d/%d]\taccuracy_D_real: %.4f \taccuracy_D_fake: %.4f \taccuracy_G: %.4f'
              % (epoch, self.num_epochs, self.accDreal_test, self.accDfake_test, self.accG_test))

    def print_epoch_losses_summary(self, epoch):
        print('[%d/%d]\tLoss_D: %.4f \tLoss_D_real: %.4f \tLoss_D_fake: %.4f \tLoss_G: %.4f'
                  % (epoch, self.num_epochs, self.errD.item(), self.errD_real.item(), self.errD_fake.item(), self.errG.item()))
        print('[%d/%d]\taccuracy_D_real: %.4f \taccuracy_D_fake: %.4f \taccuracy_G: %.4f\n'
            % (epoch, self.num_epochs, self.accDreal, self.accDfake, self.accG))

    def save_loss_plot(self, epoch, output_dir):
        loss_path = os.path.join(output_dir, "loss_plot")
        acc_path = os.path.join(output_dir, "accuracy")
        g_t_utils.plot_general_losses(self.G_losses, self.D_loss_fake, self.D_loss_real, "summary epoch_=_" + str(epoch),
                            self.num_iter, loss_path)
        g_t_utils.plot_general_accuracy(self.G_accuracy, self.D_accuracy_fake, self.D_accuracy_real, "accuracy epoch = "+ str(epoch),
                              self.epoch, acc_path)

    def train(self, output_dir):
        self.print_cuda_details()
        self.verify_checkpoint()
        start_epoch = self.epoch
        print("Starting Training Loop...")
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch += 1
            start = time.time()
            self.train_epoch()
            print("Single [[epoch]] iteration took [%.4f] seconds\n" % (time.time() - start))
            self.save_model(params.models_save_path, epoch)
            self.print_epoch_losses_summary(epoch)

            if epoch % 5 == 0:
                self.save_test_images(epoch, output_dir)
                self.save_test_loss(epoch, output_dir)

            if epoch % 5 == 0:
                self.save_loss_plot(epoch, output_dir)


    def save_groups_images(self, test_hdr_image, first_b_tonemap, fake, new_out_dir):
        b_size = first_b_tonemap.shape[0]
        output_len = int(b_size / 4)
        for i in range(output_len):
            plt.figure(figsize=(15, 15))
            plt.subplot(3, 1, 1)
            plt.axis("off")
            plt.title("Real images")
            plt.imshow(
                np.transpose(vutils.make_grid(first_b_tonemap[i * 4: (i + 1) * 4], padding=5, normalize=True).cpu(), (1, 2, 0)))

            test_hdr_display = g_t_utils.display_batch_as_grid(test_hdr_image, ncols_to_display=(i + 1) * 4, normalization="uint_0_1", isHDR=True, batch_start_index=i * 4)

            plt.subplot(3, 1, 2)
            plt.axis("off")
            plt.title("Processed Images")
            if test_hdr_display.shape[2] == 1:
                plt.imshow(test_hdr_display[:, :, 0], cmap='gray')
            else:
                plt.imshow(test_hdr_display)
            # plt.imshow(test_hdr_display)

            img_list2 = [vutils.make_grid(fake[i * 4: (i + 1) * 4], padding=5, normalize=True)]
            plt.subplot(3, 1, 3)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list2[-1].cpu(), (1, 2, 0)))
            plt.savefig(os.path.join(new_out_dir, "set " + str(i)))
            plt.close()


    def update_test_loss(self, b_size, first_b_tonemap, fake, epoch):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        with torch.no_grad():
            real_label = torch.full((b_size,), self.real_label, device=self.device)
            test_D_output_on_real = self.netD(first_b_tonemap.detach()).view(-1)
            self.accDreal_counter += (test_D_output_on_real > 0.5).sum().item()

            test_errD_real = self.criterion(test_D_output_on_real, real_label)
            self.test_D_loss_real.append(test_errD_real.item())

            fake_label = torch.full((b_size,), self.fake_label, device=self.device)
            output_on_fake = self.netD(fake.detach()).view(-1)
            self.accDfake_counter += (output_on_fake <= 0.5).sum().item()

            test_errD_fake = self.criterion(output_on_fake, fake_label)
            test_loss_D = test_errD_real + test_errD_fake
            self.test_D_loss_fake.append(test_errD_fake.item())
            self.test_D_losses.append(test_loss_D.item())

            output_on_fake = self.netD(fake.detach()).view(-1)
            self.accG_counter += (output_on_fake > 0.5).sum().item()
            test_errGd = self.criterion(output_on_fake, real_label)
            self.test_G_losses.append(test_errGd.item())
            self.update_accuracy(isTest=True)
            self.print_test_epoch_losses_summary(epoch, test_loss_D, test_errGd)

    def get_fake_test_images(self, first_b_hdr):
        with torch.no_grad():
            fake = self.netG(first_b_hdr)
            return fake

    def save_test_loss(self, epoch, out_dir):
        acc_path = os.path.join(out_dir, "accuracy")
        loss_path = os.path.join(out_dir, "loss_plot")
        g_t_utils.plot_general_losses(self.test_G_losses, self.test_D_loss_fake, self.test_D_loss_real,
                                      "TEST epoch loss = " + str(epoch), self.test_num_iter, loss_path)

        g_t_utils.plot_general_accuracy(self.G_accuracy_test, self.D_accuracy_fake_test, self.D_accuracy_real_test,
                                        "TEST epoch acc = " + str(epoch), int(self.epoch / 5) + 1, acc_path)

    def save_test_images(self, epoch, out_dir):

        out_dir = os.path.join(out_dir, "result_images")
        new_out_dir = os.path.join(out_dir, "images_epoch=" + str(epoch))

        if not os.path.exists(new_out_dir):
            os.mkdir(new_out_dir)

        self.test_num_iter += 1
        test_real_batch = next(iter(self.test_data_loader_ldr))
        test_real_first_b = test_real_batch[0].to(device)

        test_real_first_b_display = g_t_utils.display_batch_as_grid(test_real_first_b, ncols_to_display=self.batch_size, normalization="uint_0_1")
        test_first_b_display_small = g_t_utils.display_batch_as_grid(test_real_first_b, ncols_to_display=2, normalization="uint_0_1")

        test_hdr_batch = next(iter(self.test_data_loader_npy))
        # test_hdr_batch_image = test_hdr_batch[params.image_key].to(self.device)
        test_hdr_batch_image = test_hdr_batch[0].to(self.device)

        fake = self.get_fake_test_images(test_hdr_batch_image)
        fake_display = g_t_utils.display_batch_as_grid(fake, ncols_to_display=self.batch_size, normalization="uint_0_1", toPrint=True)
        fake_display_small = g_t_utils.display_batch_as_grid(fake, 2, normalization="uint_0_1")

        b_size = test_real_first_b.size(0)
        self.update_test_loss(b_size, test_real_first_b, fake, epoch)

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        if test_real_first_b_display.shape[2] == 1:
            plt.imshow(test_real_first_b_display[:, :, 0], cmap='gray')
        else:
            plt.imshow(test_real_first_b_display)

        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.title("Real Images")
        if test_first_b_display_small.shape[2] == 1:
            plt.imshow(test_first_b_display_small[:, :, 0], cmap='gray')
        else:
            plt.imshow(test_first_b_display_small)

        plt.subplot(2, 2, 3)
        plt.axis("off")
        plt.title("Fake Images")
        if fake_display.shape[2] == 1:
            plt.imshow(fake_display[:, :, 0], cmap='gray')
        else:
            plt.imshow(fake_display)

        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.title("Fake Images")
        if fake_display_small.shape[2] == 1:
            plt.imshow(fake_display_small[:, :, 0], cmap='gray')
        else:
            plt.imshow(fake_display_small)
        plt.savefig(os.path.join(new_out_dir, "ALL epoch = " + str(epoch)))
        plt.close()
        self.save_groups_images(test_hdr_batch_image, test_real_first_b, fake, new_out_dir)

    def save_model(self, path, epoch):
        path = os.path.join(output_dir, path)
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

    def test(self):
        # test_real_batch_tonemap = next(iter(self.test_data_loader_ldr))
        # test_first_b_tonemap = test_real_batch_tonemap[0].to(device)
        # tests.test_normalize_transform(test_first_b_tonemap, self.device)
        #
        # test_real_batch_hdr = next(iter(self.test_data_loader_npy))
        # test_hdr_image = test_real_batch_hdr[params.image_key].to(self.device)
        # tests.test_normalize_transform(test_hdr_image, self.device)
        #
        test_real_batch_tonemap = next(iter(self.test_data_loader_ldr))
        test_first_b_tonemap = test_real_batch_tonemap[0].to(device)

        test_real_batch_hdr = next(iter(self.test_data_loader_npy))
        test_hdr_image = test_real_batch_hdr[params.image_key].to(self.device)
        new_out_dir = os.path.join(params.results_path, "images_epoch=" + str(1))
        # self.save_groups_images(test_first_b_tonemap, test_hdr_image, new_out_dir)


if __name__ == '__main__':
    batch_size, num_epochs, model, G_lr, D_lr, train_data_root_npy, train_data_root_ldr, isCheckpoint_str, \
        test_data_root_npy, test_data_root_ldr, g_opt_for_single_d,\
        result_dir_pref, input_dim = parse_arguments()
    torch.manual_seed(params.manualSeed)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

    isCheckpoint = True
    if isCheckpoint_str == 'no':
        isCheckpoint = False

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", num_epochs)
    print("MODEL:", model)
    print("G LR: ", G_lr)
    print("D LR: ", D_lr)
    print("CHECK POINT:", isCheckpoint)
    print("TRAIN G [%d] TIMES FOR EACH D STEP" % g_opt_for_single_d)
    print("DEVICE:", device)
    print("=====================\n")

    net_G = create_net("G_" + model, device, isCheckpoint, input_dim)
    print("=================  NET G  ==================")
    print(net_G)
    summary(net_G, (input_dim, params.input_size, params.input_size))
    print()

    net_D = create_net("D", device, isCheckpoint, input_dim)
    print("=================  NET D  ==================")
    print(net_D)
    summary(net_D, (input_dim, params.input_size, params.input_size))
    print()

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=D_lr, betas=(params.beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=G_lr, betas=(params.beta1, 0.999))

    output_dir = g_t_utils.create_dir(result_dir_pref, model, params.models_save_path, params.loss_path, params.results_path)

    gan_trainer = GanTrainer(device, batch_size, num_epochs, train_data_root_npy, train_data_root_ldr,
                             test_data_root_npy, test_data_root_ldr, isCheckpoint,
                             net_G, net_D, optimizer_G, optimizer_D, input_dim)

    gan_trainer.train(output_dir)
    # # gan_trainer.test()
