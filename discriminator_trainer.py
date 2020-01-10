from __future__ import print_function

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch import autograd
from torchsummary import summary

matplotlib.use('Agg')
import Discriminator
import params
import time
import gan_trainer_utils as g_t_utils
import utils.plot_util as plot_util
import utils.data_loader_util as data_loader_util
import utils.model_save_util as model_save_util
import ssim
import printer
import tranforms as custom_transform
# import torus.Unet as TorusUnet
import unet_multi_filters.Unet as Generator

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=params.num_epochs)
    parser.add_argument("--model", type=str, default=params.unet_network)  # up sampling is the default
    parser.add_argument("--con_operator", type=str, default=params.original_unet)
    parser.add_argument("--filters", type=int, default=params.filters)
    parser.add_argument("--unet_depth", type=int, default=2)
    parser.add_argument("--G_lr", type=float, default=params.lr)
    parser.add_argument("--D_lr", type=float, default=params.lr)
    parser.add_argument("--data_root_npy", type=str, default=params.train_dataroot_hdr)
    parser.add_argument("--data_root_ldr", type=str, default=params.train_dataroot_ldr)
    parser.add_argument("--checkpoint", type=str, default="no")
    parser.add_argument("--test_data_root_npy", type=str, default=params.test_dataroot_hdr)
    parser.add_argument("--test_dataroot_original_hdr", type=str, default=params.test_dataroot_original_hdr)

    parser.add_argument("--test_data_root_ldr", type=str, default=params.test_dataroot_ldr)
    parser.add_argument("--result_dir_prefix", type=str, default="d_train_")
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--loss_g_d_factor", type=float, default=1)
    parser.add_argument("--ssim_loss_g_factor", type=float, default=1)
    # if 0, images are in [-1, 1] range, if 0.5 then [0,1]
    parser.add_argument("--input_images_mean", type=float, default=0)
    parser.add_argument("--log_factor", type=int, default=1000)
    parser.add_argument("--use_transform_exp", type=int, default=1)  # int(False) = 0
    parser.add_argument("--epoch_to_save", type=int, default=2)
    args = parser.parse_args()
    return args.batch, args.epochs, args.model, args.con_operator, args.filters, args.G_lr, args.D_lr, os.path.join(args.data_root_npy), \
           os.path.join(args.data_root_ldr), args.checkpoint, os.path.join(args.test_data_root_npy), \
           os.path.join(args.test_data_root_ldr), args.result_dir_prefix, args.input_dim, \
           args.loss_g_d_factor, args.ssim_loss_g_factor, args.input_images_mean, args.use_transform_exp, \
           args.log_factor, args.test_dataroot_original_hdr, args.epoch_to_save, args.unet_depth

class GanTrainer:
    def __init__(self, t_device, t_batch_size, t_num_epochs, train_dataroot_npy,
                 train_dataroot_ldr, test_dataroot_npy, test_dataroot_ldr, t_isCheckpoint, t_netD,
                 t_optimizerD,
                 input_images_mean_, log_factor_, test_dataroot_original_hdr,
                 epoch_to_save_):
        self.batch_size = t_batch_size
        self.num_epochs = t_num_epochs
        self.device = t_device
        self.netD = t_netD
        self.optimizerD = t_optimizerD
        self.criterion = nn.BCELoss()
        self.epoch_to_save = epoch_to_save_
        self.use_transform_exp = 1 #true
        self.real_label, self.fake_label = 1, 0
        self.epoch, self.num_iter, self.test_num_iter = 0, 0, 0

        self.errD_real, self.errD_fake, self.errD = None, None, None
        self.accG, self.accD, self.accDreal, self.accDfake = None, None, None, None
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.G_loss_ssim, self.G_loss_d = [], []
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []

        self.train_data_loader_npy, self.train_data_loader_ldr = \
            data_loader_util.load_data(train_dataroot_npy, train_dataroot_ldr,
                                       self.batch_size, testMode=False, title="train")

        self.input_dim = input_dim
        self.input_images_mean = input_images_mean_
        self.isCheckpoint = t_isCheckpoint
        self.checkpoint = None
        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss = ssim.SSIM(window_size=5)

        self.log_factor = log_factor_
        self.transform_exp = custom_transform.Exp(log_factor_)
        self.normalize = custom_transform.Normalize(0.5, 0.5)

    def update_accuracy(self):
        len_ldr_train_dset = len(self.train_data_loader_ldr.dataset)

        self.accDreal = self.accDreal_counter / len_ldr_train_dset
        self.accDfake = self.accDfake_counter / len_ldr_train_dset
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
        if self.use_transform_exp:
            fake = self.transform_exp(hdr_input)
            fake = self.normalize(fake)
        else:
            fake = hdr_input
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

    def train_epoch(self):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        for (h, data_hdr), (l, data_ldr) in zip(enumerate(self.train_data_loader_npy, 0),
                                                enumerate(self.train_data_loader_ldr, 0)):
            self.num_iter += 1
            with autograd.detect_anomaly():
                real_ldr_cpu = data_ldr[params.gray_input_image_key].to(self.device)
                hdr_input = data_hdr[params.gray_input_image_key].to(self.device)
                self.train_D(hdr_input, real_ldr_cpu)
                plot_util.plot_grad_flow(self.netD.named_parameters(), output_dir, 1)
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
        plot_util.plot_discriminator_losses(self.D_loss_fake,
                                      self.D_loss_real, "summary epoch_=_" + str(epoch), self.num_iter, loss_path)
        plot_util.plot_general_accuracy(self.G_accuracy, self.D_accuracy_fake, self.D_accuracy_real,
                                        "accuracy epoch = " + str(epoch),
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
            new_out_dir = os.path.join(output_dir, "gradient_flow")
            plt.savefig(os.path.join(new_out_dir, "gradient_flow_epoch=" + str(epoch)))
            plt.close()

            print("Single [[epoch]] iteration took [%.4f] seconds\n" % (time.time() - start))

            printer.print_epoch_losses_summary(epoch, self.num_epochs, self.errD.item(), self.errD_real.item(),
                                               self.errD_fake.item(), 0, 0,
                                               0, 0)
            printer.print_epoch_acc_summary(epoch, self.num_epochs, self.accDfake, self.accDreal, 0,
                                            0)
            if epoch % self.epoch_to_save == 0:
                model_save_util.save_discriminator_model(params.models_save_path, epoch, output_dir,
                                           self.netD, self.optimizerD)
                self.save_loss_plot(epoch, output_dir)
                # self.tester.update_TMQI(self.netG, output_dir, epoch)
        # self.writer.close()

    def load_model(self):
        if self.isCheckpoint:
            self.checkpoint = torch.load(params.models_save_path)
            self.epoch = self.checkpoint['epoch']
            self.netD.load_state_dict(self.checkpoint['modelD_state_dict'])
            self.optimizerD.load_state_dict(self.checkpoint['optimizerD_state_dict'])
            self.netD.train()


if __name__ == '__main__':
    batch_size, num_epochs, model, con_operator, filters, G_lr, D_lr, train_data_root_npy, train_data_root_ldr, isCheckpoint_str, \
    test_data_root_npy, test_data_root_ldr, result_dir_pref, input_dim, loss_g_d_factor, \
    ssim_loss_factor, input_images_mean, use_transform_exp, log_factor, test_dataroot_original_hdr, \
    epoch_to_save, depth = parse_arguments()
    torch.manual_seed(params.manualSeed)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    # device = torch.device("cpu")
    isCheckpoint = True
    if isCheckpoint_str == 'no':
        isCheckpoint = False

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", num_epochs)
    print("MODEL:", model, con_operator)
    print("UNET DEPTH: ", depth)
    print("G LR: ", G_lr)
    print("D LR: ", D_lr)
    print("CHECK POINT:", isCheckpoint)
    print("INPUT DIM:", input_dim)
    print("INPUT IMAGES MEAN:", input_images_mean)
    print("LOSS G D FACTOR:", loss_g_d_factor)
    print("SSIM LOSS FACTOR:", ssim_loss_factor)
    print("LOG FACTOR:", log_factor)
    print("DEVICE:", device)
    print("=====================\n")

    net_D = model_save_util.create_net("D", "D", device, isCheckpoint, input_dim, input_images_mean)
    print("=================  NET D  ==================")
    print(net_D)
    summary(net_D, (input_dim, 256, 256), device="cpu")
    print()

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=D_lr, betas=(params.beta1, 0.999))

    output_dir = g_t_utils.create_dir(result_dir_pref + "_log_" + str(log_factor), model, con_operator, params.models_save_path,
                                      params.loss_path, params.results_path, depth)

    gan_trainer = GanTrainer(device, batch_size, num_epochs, train_data_root_npy, train_data_root_ldr,
                             test_data_root_npy, test_data_root_ldr, isCheckpoint,
                             net_D, optimizer_D,
                             input_images_mean, log_factor,
                             test_dataroot_original_hdr, epoch_to_save)

    gan_trainer.train(output_dir)
