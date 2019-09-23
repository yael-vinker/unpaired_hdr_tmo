from __future__ import print_function

import unet.Unet as Unet
import VAE
from torch import autograd
import os
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import vgg_metric
import Discriminator
import params
import time
import gan_trainer_utils as g_t_utils
import ProcessedDatasetFolder
import ssim
import printer
# import Writer


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
    parser.add_argument("--model", type=str, default="skip_connection_conv") # up sampling is the default
    parser.add_argument("--G_lr", type=float, default=params.lr)
    parser.add_argument("--D_lr", type=float, default=params.lr)
    parser.add_argument("--data_root_npy", type=str, default=params.train_dataroot_hdr)
    parser.add_argument("--data_root_ldr", type=str, default=params.train_dataroot_ldr)
    parser.add_argument("--checkpoint", type=str, default="no")
    parser.add_argument("--test_data_root_npy", type=str, default=params.test_dataroot_hdr)
    parser.add_argument("--test_data_root_ldr", type=str, default=params.test_dataroot_ldr)
    parser.add_argument("--G_opt_for_single_D", type=int, default=1)
    parser.add_argument("--result_dir_prefix", type=str, default="local")
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--loss_g_d_factor", type=float, default=1)
    parser.add_argument("--ssim_loss_g_factor", type=float, default=1)
    parser.add_argument("--rgb_l2_loss_g_factor", type=float, default=0)
    # if 0, images are in [-1, 1] range, if 0.5 then [0,1]
    parser.add_argument("--input_images_mean", type=float, default=0)
    args = parser.parse_args()
    return args.batch, args.epochs, args.model, args.G_lr, args.D_lr, os.path.join(args.data_root_npy), \
           os.path.join(args.data_root_ldr), args.checkpoint, os.path.join(args.test_data_root_npy), \
           os.path.join(args.test_data_root_ldr), args.G_opt_for_single_D, args.result_dir_prefix, \
           args.input_dim, args.loss_g_d_factor, args.ssim_loss_g_factor, args.rgb_l2_loss_g_factor, args.input_images_mean


def create_net(net, device_, is_checkpoint, input_dim_, input_images_mean_):
    # Create the Generator (UNet architecture)
    # norm_layer = UnetGenerator.get_norm_layer(norm_type='batch')
    if net == "G_VAE":
        new_net = VAE.VAE(input_dim_).to(device_)
    elif net == "G_skip_connection":
        new_net = Unet.UNet(input_dim_, input_dim_, input_images_mean_, bilinear=True).to(device_)
    elif net == "G_skip_connection_conv":
        new_net = Unet.UNet(input_dim_, input_dim_, input_images_mean_, bilinear=False).to(device_)
    # Create the Discriminator
    elif net == "D":
        new_net = Discriminator.Discriminator(params.input_size, input_dim_, params.dim,
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
                 t_optimizerG, t_optimizerD, input_dim, loss_g_d_factor_, ssim_loss_g_factor_, rgb_l2_loss_g_factor_,
                 input_images_mean_, writer_):
        self.writer = writer_
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

        self.errG_d, self.errG_ssim, self.errG_rgb_l2 = None, None, None
        self.errD_real, self.errD_fake, self.errD = None, None, None
        self.accG, self.accD, self.accDreal, self.accDfake = None, None, None, None
        self.accG_test, self.accD_test, self.accDreal_test, self.accDfake_test = None, None, None, None
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.G_accuracy_test, self.D_accuracy_real_test, self.D_accuracy_fake_test = [], [], []
        self.G_loss_ssim, self.G_loss_rgb_l2, self.G_loss_d = [], [], []
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []
        self.test_G_losses_d, self.test_G_loss_ssim, self.test_G_loss_rgb_l2, = [], [], []
        self.test_D_losses, self.test_D_loss_fake, self.test_D_loss_real = [], [], []

        self.train_data_loader_npy, self.train_data_loader_ldr, self.test_data_loader_npy, self.test_data_loader_ldr = \
            self.load_data(train_dataroot_npy, train_dataroot_ldr, test_dataroot_npy, test_dataroot_ldr)

        self.input_dim = input_dim
        self.input_images_mean = input_images_mean_
        self.isCheckpoint = t_isCheckpoint
        self.checkpoint = None
        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss = ssim.SSIM(window_size=5)
        self.vgg_loss = vgg_metric.distance_metric(params.input_size)

        self.loss_g_d_factor = loss_g_d_factor_
        self.retain_loss_g_gd_graph = (ssim_loss_g_factor_ != 0) or (rgb_l2_loss_g_factor_ != 0)
        self.ssim_loss_g_factor = ssim_loss_g_factor_
        self.retain_ssim_graph = rgb_l2_loss_g_factor_ != 0
        self.rgb_l2_loss_g_factor = rgb_l2_loss_g_factor_

    def load_npy_data(self, npy_data_root, batch_size, shuffle, testMode):
        npy_dataset = ProcessedDatasetFolder.ProcessedDatasetFolder(root=npy_data_root, testMode=testMode)
        dataloader = torch.utils.data.DataLoader(npy_dataset, batch_size=batch_size,
                                                 shuffle=shuffle, num_workers=params.workers)
        return dataloader

    def load_ldr_data(self, ldr_data_root, batch_size, shuffle, testMode):
        ldr_dataset = ProcessedDatasetFolder.ProcessedDatasetFolder(root=ldr_data_root, testMode=testMode)
        ldr_dataloader = torch.utils.data.DataLoader(ldr_dataset, batch_size=batch_size,
                                                     shuffle=shuffle, num_workers=params.workers)
        return ldr_dataloader

    def load_data(self, train_root_npy, train_root_ldr, test_root_npy, test_root_ldr):
        """
        :param isHdr: True if images in "dir_root" are in .hdr format, False otherwise.
        :param dir_root: path to wanted directory
        :param b_size: batch size
        :return: DataLoader object of images in "dir_root"
        """
        train_hdr_dataloader = self.load_npy_data(train_root_npy, self.batch_size, shuffle=True, testMode=False)
        test_hdr_dataloader = self.load_npy_data(test_root_npy, self.batch_size, shuffle=False, testMode=True)
        train_ldr_dataloader = self.load_ldr_data(train_root_ldr, self.batch_size, shuffle=True, testMode=False)
        test_ldr_dataloader = self.load_ldr_data(test_root_ldr, self.batch_size, shuffle=False, testMode=True)

        printer.print_dataset_details([train_hdr_dataloader, test_hdr_dataloader, train_ldr_dataloader,
                                       test_ldr_dataloader],
                                        [train_root_npy, test_root_npy, train_root_ldr, test_root_ldr],
                                        ["train_hdr_dataloader", "test_hdr_dataloader", "train_ldr_dataloader",
                                         "test_ldr_dataloader"],
                                        [True, True, False, False],
                                        [True, True, True, True])

        printer.load_data_dict_mode(train_hdr_dataloader, train_ldr_dataloader, "train", images_number=2)
        printer.load_data_dict_mode(test_hdr_dataloader, test_ldr_dataloader, "test", images_number=2)
        return train_hdr_dataloader, train_ldr_dataloader, test_hdr_dataloader, test_ldr_dataloader

    def update_accuracy(self, isTest=False):
        len_hdr_train_dset = len(self.train_data_loader_npy.dataset)
        len_hdr_test_dset = len(self.test_data_loader_npy.dataset)
        len_ldr_train_dset = len(self.train_data_loader_ldr.dataset)
        len_ldr_test_dset = len(self.test_data_loader_ldr.dataset)

        if isTest:
            self.accG_test = self.accG_counter / len_hdr_test_dset
            self.accDreal_test = self.accDreal_counter / len_ldr_train_dset
            self.accDfake_test = self.accDfake_counter / len_ldr_train_dset
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

    def train_D(self, hdr_input, real_ldr_cpu):
        """
        Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        :param real_hdr_cpu: HDR images as input to G to generate fake data
        :param real_ldr_cpu: LDR images as real input to D
        :param label: Tensor contains real labels for first loss
        :return: fake (Tensor) result of G on hdr_data, for G train
        """
        label = torch.full((self.batch_size,), self.real_label, device=self.device)
        # Train with all-real batch
        self.netD.zero_grad()
        # Forward pass real batch through D
        output_on_real = self.netD(real_ldr_cpu).view(-1)
        # Real label = 1, so we count the samples on which D was right
        self.accDreal_counter += (output_on_real > 0.5).sum().item()

        # Calculate loss on all-real batch
        # self.errD_real = self.criterion(output_on_real, label)
        self.errD_real = self.mse_loss(output_on_real, label)
        self.errD_real.backward()

        # Train with all-fake batch
        # Generate fake image batch with G
        fake = self.netG(hdr_input)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output_on_fake = self.netD(fake.detach()).view(-1)
        # Fake label = 0, so we count the samples on which D was right
        self.accDfake_counter += (output_on_fake <= 0.5).sum().item()
        # Calculate D's loss on the all-fake batch
        # self.errD_fake = self.criterion(output_on_fake, label)
        self.errD_fake = self.mse_loss(output_on_fake, label)
        # Calculate the gradients for this batch
        self.errD_fake.backward()
        # Add the gradients from the all-real and all-fake batches
        self.errD = self.errD_real + self.errD_fake
        # Update D
        self.optimizerD.step()
        self.D_losses.append(self.errD.item())
        self.D_loss_fake.append(self.errD_fake.item())
        self.D_loss_real.append(self.errD_real.item())

    def update_g_d_loss(self, output_on_fake, label):
        if self.loss_g_d_factor != 0:
            # self.errG_d = self.loss_g_d_factor * (self.criterion(output_on_fake, label))
            self.errG_d = self.loss_g_d_factor * (self.mse_loss(output_on_fake, label))
            self.errG_d.backward(retain_graph=self.retain_loss_g_gd_graph)
            self.G_loss_d.append(self.errG_d.item())
        else:
            self.errG_d = self.criterion(output_on_fake, label)
            self.G_loss_d.append(self.errG_d.item())


    def update_ssim_loss(self, hdr_input, fake):
        if self.ssim_loss_g_factor != 0:
            if self.input_dim == 3:
                fake_rgb_n = torch.sum(fake, dim=1)[:,None,:, :]
                hdr_input_rgb_n = torch.sum(hdr_input, dim=1)[:,None,:, :]
            elif self.input_images_mean == 0.5:
                fake_rgb_n = fake
                hdr_input_rgb_n = hdr_input
            else:
                fake_rgb_n = fake + 1
                hdr_input_rgb_n = hdr_input + 1
            self.errG_ssim = self.ssim_loss_g_factor * (1 - self.ssim_loss(fake_rgb_n, hdr_input_rgb_n))
            self.errG_ssim.backward()
            self.G_loss_ssim.append(self.errG_ssim.item())


    def update_rgb_l2_loss(self, fake, hdr_input_display):
        # if self.rgb_l2_loss_g_factor != 0 and self.input_dim == 3:
        if self.rgb_l2_loss_g_factor != 0:
            colored_fake = g_t_utils.back_to_color_batch_tensor(fake, hdr_input_display)
            # if self.input_dim == 3:
            # new_fake = g_t_utils.get_rgb_normalize_im_batch(fake)
            # new_hdr_unput = g_t_utils.get_rgb_normalize_im_batch(hdr_input)
            #
            # self.errG_rgb_l2 = self.rgb_l2_loss_g_factor * self.mse_loss(new_fake, new_hdr_unput)
            self.errG_rgb_l2 = self.vgg_loss(colored_fake, hdr_input_display)

            self.errG_rgb_l2.backward()
            self.G_loss_rgb_l2.append(self.errG_rgb_l2.item())
            # else:


    def train_G(self, label, hdr_input, hdr_input_display):
        """
        Update G network: maximize log(D(G(z))) and minimize loss_wind
        :param label: Tensor contains real labels for first loss
        :param fake: (Tensor) result of G on hdr_data
        :param real_hdr_cpu: HDR images as input to windows_loss
        """
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        fake = self.netG(hdr_input)
        printer.print_g_progress(fake)

        output_on_fake = self.netD(fake).view(-1)
        # Real label = 1, so wo count number of samples on which G tricked D
        self.accG_counter += (output_on_fake > 0.5).sum().item()
        # updates all G's losses
        self.update_g_d_loss(output_on_fake, label)
        # self.writer.add_scaler('Loss/train', self.errG_d.item(), self.num_iter)
        self.update_ssim_loss(hdr_input, fake)
        self.update_rgb_l2_loss(fake, hdr_input_display)
        self.optimizerG.step()

    def train_epoch(self):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        for (h, data_hdr), (l, data_ldr) in zip(enumerate(self.train_data_loader_npy, 0),
                                                enumerate(self.train_data_loader_ldr, 0)):
            self.num_iter += 1
            with autograd.detect_anomaly():
                real_ldr_cpu = data_ldr["input_im"].to(self.device)
                hdr_input = data_hdr["input_im"].to(self.device)
                hdr_input_display = data_hdr["color_im"].to(self.device)
                b_size = hdr_input.size(0)
                label = torch.full((b_size,), self.real_label, device=self.device)

                self.train_D(hdr_input, real_ldr_cpu)
                self.train_G(label, hdr_input, hdr_input_display)
        self.update_accuracy()

    def verify_checkpoint(self):
        if isCheckpoint:
            print("Loading model...")
            self.load_model()
            print("Model was loaded")
            print()

    def save_loss_plot(self, epoch, output_dir):
        loss_path = os.path.join(output_dir, "loss_plot")
        acc_path = os.path.join(output_dir, "accuracy")
        g_t_utils.plot_general_losses(self.G_loss_d, self.G_loss_ssim, self.G_loss_rgb_l2, self.D_loss_fake,
                                      self.D_loss_real, "summary epoch_=_" + str(epoch), self.num_iter, loss_path,
                                      (self.loss_g_d_factor != 0),(self.ssim_loss_g_factor != 0),
                                      (self.rgb_l2_loss_g_factor != 0))
        g_t_utils.plot_general_accuracy(self.G_accuracy, self.D_accuracy_fake, self.D_accuracy_real, "accuracy epoch = "+ str(epoch),
                              self.epoch, acc_path)

    def train(self, output_dir):
        printer.print_cuda_details(self.device.type)
        self.verify_checkpoint()
        start_epoch = self.epoch
        print("Starting Training Loop...")
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch += 1
            start = time.time()

            self.train_epoch()

            print("Single [[epoch]] iteration took [%.4f] seconds\n" % (time.time() - start))
            self.save_model(params.models_save_path, epoch)
            printer.print_epoch_losses_summary(epoch, self.num_epochs, self.errD.item(), self.errD_real.item(),
                                               self.errD_fake.item(), self.loss_g_d_factor, self.errG_d,
                                               self.ssim_loss_g_factor, self.errG_ssim,
                                               self.rgb_l2_loss_g_factor, self.errG_rgb_l2)
            if epoch % 1 == 0:
                self.save_test_images(epoch, output_dir)
                self.save_test_loss(epoch, output_dir)

            if epoch % 1 == 0:
                self.save_loss_plot(epoch, output_dir)


    def update_test_loss(self, b_size, first_b_tonemap, fake, hdr_input, epoch):
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
            # if self.loss_g_d_factor != 0:
            test_errGd = self.criterion(output_on_fake, real_label)
            self.test_G_losses_d.append(test_errGd.item())
            if self.ssim_loss_g_factor != 0:
                test_errGssim = self.ssim_loss_g_factor * (1 - self.ssim_loss(fake, hdr_input))
                self.test_G_loss_ssim.append(test_errGssim.item())
            if self.rgb_l2_loss_g_factor != 0:
                new_fake = g_t_utils.get_rgb_normalize_im_batch(fake)
                new_hdr_unput = g_t_utils.get_rgb_normalize_im_batch(hdr_input)
                test_errRGBl2 = self.rgb_l2_loss_g_factor * (self.mse_loss(new_fake, new_hdr_unput) / (new_fake.shape[1] * new_fake.shape[2] * new_fake.shape[3]))
                self.test_G_loss_rgb_l2.append(test_errRGBl2.item())
            self.update_accuracy(isTest=True)
            printer.print_test_epoch_losses_summary(self.num_epochs, epoch, test_loss_D, test_errGd, self.accDreal_test,
                                                 self.accDfake_test, self.accG_test)

    def get_fake_test_images(self, first_b_hdr):
        with torch.no_grad():
            fake = self.netG(first_b_hdr)
            return fake

    def save_test_loss(self, epoch, out_dir):
        acc_path = os.path.join(out_dir, "accuracy")
        loss_path = os.path.join(out_dir, "loss_plot")
        g_t_utils.plot_general_losses(self.test_G_losses_d, self.test_G_loss_ssim, self.test_G_loss_rgb_l2,
                                      self.test_D_loss_fake, self.test_D_loss_real,
                                      "TEST epoch loss = " + str(epoch), self.test_num_iter, loss_path,
                                      (self.loss_g_d_factor != 0), (self.ssim_loss_g_factor != 0),
                                      (self.rgb_l2_loss_g_factor != 0))

        g_t_utils.plot_general_accuracy(self.G_accuracy_test, self.D_accuracy_fake_test, self.D_accuracy_real_test,
                                        "TEST epoch acc = " + str(epoch), self.epoch, acc_path)

    def save_test_images(self, epoch, out_dir):
        out_dir = os.path.join(out_dir, "result_images")
        new_out_dir = os.path.join(out_dir, "images_epoch=" + str(epoch))

        if not os.path.exists(new_out_dir):
            os.mkdir(new_out_dir)

        self.test_num_iter += 1
        test_real_batch = next(iter(self.test_data_loader_ldr))
        test_real_first_b = test_real_batch["input_im"].to(device)

        test_hdr_batch = next(iter(self.test_data_loader_npy))
        # test_hdr_batch_image = test_hdr_batch[params.image_key].to(self.device)
        test_hdr_batch_image = test_hdr_batch["input_im"].to(self.device)

        fake = self.get_fake_test_images(test_hdr_batch_image)
        fake_ldr = self.get_fake_test_images(test_real_batch["input_im"].to(self.device))

        g_t_utils.save_groups_images(test_hdr_batch, test_real_batch, fake, fake_ldr,
                                     new_out_dir, len(self.test_data_loader_npy.dataset), epoch,
                                     self.input_images_mean)
        self.update_test_loss(test_real_first_b.size(0), test_real_first_b, fake, test_hdr_batch_image, epoch)

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
        test_data_root_npy, test_data_root_ldr, g_opt_for_single_d, result_dir_pref, input_dim, loss_g_d_factor, \
    ssim_loss_factor, rgb_l2_loss_g_factor, input_images_mean = parse_arguments()
    torch.manual_seed(params.manualSeed)
    # device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    device = torch.device("cpu")
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
    print("INPUT DIM:", input_dim)
    print("INPUT IMAGES MEAN:", input_images_mean)
    print("LOSS G D FACTOR:", loss_g_d_factor)
    print("SSIM LOSS FACTOR:", ssim_loss_factor)
    print("RGB L2 LOSS FACTOR:", rgb_l2_loss_g_factor)
    print("TRAIN G [%d] TIMES FOR EACH D STEP" % g_opt_for_single_d)
    print("DEVICE:", device)
    print("=====================\n")

    net_G = create_net("G_" + model, device, isCheckpoint, input_dim, input_images_mean)
    print("=================  NET G  ==================")
    print(net_G)
    # summary(net_G, (input_dim, params.input_size, params.input_size))
    print()

    net_D = create_net("D", device, isCheckpoint, input_dim, input_images_mean)
    print("=================  NET D  ==================")
    print(net_D)
    # summary(net_D, (input_dim, params.input_size, params.input_size))
    print()

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=D_lr, betas=(params.beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=G_lr, betas=(params.beta1, 0.999))

    # writer = Writer.Writer(g_t_utils.get_loss_path(result_dir_pref, model, params.loss_path))
    writer = 1
    output_dir = g_t_utils.create_dir(result_dir_pref, model, params.models_save_path, params.loss_path, params.results_path)

    gan_trainer = GanTrainer(device, batch_size, num_epochs, train_data_root_npy, train_data_root_ldr,
                             test_data_root_npy, test_data_root_ldr, isCheckpoint,
                             net_G, net_D, optimizer_G, optimizer_D, input_dim, loss_g_d_factor,
                             ssim_loss_factor, rgb_l2_loss_g_factor, input_images_mean, writer)

    gan_trainer.train(output_dir)
