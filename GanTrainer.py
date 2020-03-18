from __future__ import print_function

import os
import time
import matplotlib.pyplot as plt
import torch.utils.data
from torch import autograd

import Tester
import matplotlib
# matplotlib.use('Agg')
from utils import printer, params
from models import ssim
import tranforms as custom_transform
import utils.data_loader_util as data_loader_util
import utils.model_save_util as model_save_util
import utils.plot_util as plot_util
import torch.nn.functional as F


class GanTrainer:
    def __init__(self, opt, t_netG, t_netD, t_optimizerG, t_optimizerD, lr_scheduler_G, lr_scheduler_D):
        # ====== GENERAL SETTINGS ======
        self.device = opt.device
        self.isCheckpoint = opt.checkpoint
        self.checkpoint = None

        # ====== TRAINING ======
        self.batch_size = opt.batch_size
        self.num_epochs = opt.num_epochs
        self.netG = t_netG
        self.netD = t_netD
        self.optimizerD = t_optimizerD
        self.optimizerG = t_optimizerG
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.real_label, self.fake_label = 1, 0
        self.epoch, self.num_iter, self.test_num_iter = 0, 0, 0
        self.d_model = opt.d_model

        # ====== LOSS ======
        self.pyramid_loss = opt.pyramid_loss
        self.pyramid_weight_list = opt.pyramid_weight_list
        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss_name = opt.ssim_loss
        if opt.ssim_loss == params.ssim_custom:
            self.ssim_loss = ssim.OUR_CUSTOM_SSIM(window_size=opt.ssim_window_size, use_c3=opt.use_c3_in_ssim,
                                                  apply_sig_mu_ssim=opt.apply_sig_mu_ssim)
        if opt.pyramid_loss:
            self.ssim_loss = ssim.OUR_CUSTOM_SSIM_PYRAMID(window_size=opt.ssim_window_size,
                                                          pyramid_weight_list=opt.pyramid_weight_list,
                                                          pyramid_pow=opt.pyramid_pow, use_c3=opt.use_c3_in_ssim,
                                                          apply_sig_mu_ssim=opt.apply_sig_mu_ssim)
        self.use_c3 = opt.use_c3_in_ssim
        self.ssim_compare_to = opt.ssim_compare_to
        if opt.use_sigma_loss:
            self.sigma_loss = ssim.OUR_SIGMA_SSIM(window_size=opt.ssim_window_size)
            self.sig_loss_factor = opt.use_sigma_loss
        else:
            self.sigma_loss = None

        self.loss_g_d_factor = opt.loss_g_d_factor
        self.ssim_loss_g_factor = opt.ssim_loss_factor
        self.errG_d, self.errG_ssim, self.errG_sigma = None, None, None
        self.errD_real, self.errD_fake, self.errD = None, None, None
        self.accG, self.accD, self.accDreal, self.accDfake = None, None, None, None
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.G_loss_ssim, self.G_loss_d, self.G_loss_sigma = [], [], []
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []

        # ====== DATASET ======
        self.train_data_loader_npy, self.train_data_loader_ldr = \
            data_loader_util.load_data(opt.data_root_npy, opt.data_root_ldr,
                                       self.batch_size, addFrame=opt.add_frame, title="train",
                                       normalization=opt.normalization, use_c3=opt.use_c3_in_ssim)
        self.input_dim = opt.input_dim
        self.input_images_mean = opt.input_images_mean
        self.log_factor = opt.log_factor
        self.normalize = custom_transform.Normalize(0.5, 0.5)
        self.use_factorise_gamma_data = opt.use_factorise_gamma_data

        # ====== POST PROCESS ======
        self.to_crop = opt.add_frame
        self.use_normalization = opt.use_normalization
        self.normalization = opt.normalization

        # ====== SAVE RESULTS ======
        self.output_dir = opt.output_dir
        self.epoch_to_save = opt.epoch_to_save
        self.best_accG = 0
        self.tester = Tester.Tester(self.device, self.loss_g_d_factor, self.ssim_loss_g_factor,
                                    self.log_factor, opt)

    def train(self):
        printer.print_cuda_details(self.device.type)
        self.verify_checkpoint()
        start_epoch = self.epoch
        print("Starting Training Loop...")
        for epoch in range(start_epoch, self.num_epochs):
            start = time.time()
            self.epoch += 1
            self.train_epoch()
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            # self.save_gradient_flow(epoch)
            self.print_epoch_summary(epoch, start)

    def train_epoch(self):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        for (h, data_hdr), (l, data_ldr) in zip(enumerate(self.train_data_loader_npy, 0),
                                                enumerate(self.train_data_loader_ldr, 0)):
            self.num_iter += 1
            with autograd.detect_anomaly():
                real_ldr = data_ldr[params.gray_input_image_key].to(self.device)
                hdr_input = data_hdr[params.gray_input_image_key].to(self.device)
                hdr_original_gray_norm = data_hdr[params.original_gray_norm_key].to(self.device)
                hdr_original_gray = data_hdr[params.original_gray_key].to(self.device)
                self.train_D(hdr_input, real_ldr)
                self.train_G(hdr_input, hdr_original_gray_norm, hdr_original_gray)
                # plot_util.plot_grad_flow(self.netG.named_parameters(), self.output_dir, 1)
        self.update_accuracy()
        if self.epoch > 20:
            self.update_best_G_acc()


    def train_D(self, hdr_input, real_ldr):
        """
        Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        :param real_hdr_cpu: HDR images as input to G to generate fake data
        :param real_ldr: LDR images as real input to D
        :param label: Tensor contains real labels for first loss
        :return: fake (Tensor) result of G on hdr_data, for G train
        """
        # Train with all-real batch
        self.netD.zero_grad()
        # Forward pass real batch through D
        output_on_real = self.netD(real_ldr).view(-1)
        # Real label = 1, so we count the samples on which D was right
        self.accDreal_counter += (output_on_real > 0.5).sum().item()

        # Calculate loss on all-real batch
        label = torch.full(output_on_real.shape, self.real_label, device=self.device)
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

    def train_G(self, hdr_input, hdr_original_gray_norm, hdr_original_gray):
        """
        Update G network: maximize log(D(G(z))) and minimize loss_wind
        :param label: Tensor contains real labels for first loss
        :param hdr_input: (Tensor)
        """
        self.netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        printer.print_g_progress(hdr_input, "hdr_inp")
        fake = self.netG(hdr_input)
        printer.print_g_progress(fake, "output")
        output_on_fake = self.netD(fake).view(-1)
        # Real label = 1, so wo count number of samples on which G tricked D
        self.accG_counter += (output_on_fake > 0.5).sum().item()
        # updates all G's losses
        # fake labels are real for generator cost
        label = torch.full(output_on_fake.shape, self.real_label, device=self.device)
        self.update_g_d_loss(output_on_fake, label)
        self.update_ssim_loss(hdr_original_gray_norm, fake)
        self.update_sigma_loss(hdr_original_gray, fake)
        self.optimizerG.step()

    def update_g_d_loss(self, output_on_fake, label):
        self.errG_d = self.loss_g_d_factor * (self.mse_loss(output_on_fake, label))
        retain_graph = False
        if self.ssim_loss_g_factor:
            retain_graph = True
        self.errG_d.backward(retain_graph=retain_graph)
        self.G_loss_d.append(self.errG_d.item())

    def update_ssim_loss(self, hdr_input_original_gray_norm, fake):
        if self.ssim_loss_g_factor:
            self.errG_ssim = self.ssim_loss_g_factor * self.ssim_loss(fake, hdr_input_original_gray_norm)
            retain_graph = False
            if self.sigma_loss:
                retain_graph = True
            self.errG_ssim.backward(retain_graph=retain_graph)
            self.G_loss_ssim.append(self.errG_ssim.item())

    def update_sigma_loss(self, hdr_input_original_gray, fake):
        if self.sigma_loss:
            # if (not self.use_c3) and (not self.use_factorise_gamma_data):
            #     print("normalize input before sigma loss")
            #     x_max = hdr_input_original.view(hdr_input_original.shape[0], -1).max(dim=1)[0].reshape(
            #         hdr_input_original.shape[0], 1, 1, 1)
            #     hdr_input_original = hdr_input_original / x_max
            self.errG_sigma = self.sig_loss_factor * self.sigma_loss(fake, hdr_input_original_gray)
            self.errG_sigma.backward()
            self.G_loss_sigma.append(self.errG_sigma.item())

    def update_best_G_acc(self):
        if self.accG > self.best_accG:
            self.best_accG = self.accG
            printer.print_best_acc_error(self.best_accG, self.epoch)
            model_save_util.save_best_model(self.netG, self.output_dir, self.optimizerG)

    def verify_checkpoint(self):
        if self.isCheckpoint:
            print("Loading model...")
            self.load_model()
            print("Model was loaded")
            print()

    def save_loss_plot(self, epoch, output_dir):
        loss_path = os.path.join(output_dir, "loss_plot")
        acc_path = os.path.join(output_dir, "accuracy")
        plot_util.plot_general_losses(self.G_loss_d, self.G_loss_ssim, self.G_loss_sigma, self.D_loss_fake,
                                      self.D_loss_real, "summary epoch_=_" + str(epoch), self.num_iter, loss_path,
                                      (self.loss_g_d_factor != 0), (self.ssim_loss_g_factor != 0))
        plot_util.plot_general_accuracy(self.G_accuracy, self.D_accuracy_fake, self.D_accuracy_real,
                                        "accuracy epoch = " + str(epoch),
                                        self.epoch, acc_path)

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

    def update_accuracy(self):
        len_hdr_train_dset = len(self.train_data_loader_npy.dataset)
        len_ldr_train_dset = len(self.train_data_loader_ldr.dataset)

        self.accG = self.accG_counter / len_hdr_train_dset
        self.accDreal = self.accDreal_counter / len_ldr_train_dset
        self.accDfake = self.accDfake_counter / len_ldr_train_dset

        if self.d_model == "patchD":
            self.accG = self.accG / 900
            self.accDreal = self.accDreal / 900
            self.accDfake = self.accDfake / 900

        self.G_accuracy.append(self.accG)
        self.D_accuracy_real.append(self.accDreal)
        self.D_accuracy_fake.append(self.accDfake)

    def print_epoch_summary(self, epoch, start):
        print("Single [[epoch]] iteration took [%.4f] seconds\n" % (time.time() - start))
        printer.print_epoch_losses_summary(epoch, self.num_epochs, self.errD.item(), self.errD_real.item(),
                                           self.errD_fake.item(), self.loss_g_d_factor, self.errG_d,
                                           self.ssim_loss_g_factor, self.errG_ssim, self.errG_sigma)
        printer.print_epoch_acc_summary(epoch, self.num_epochs, self.accDfake, self.accDreal, self.accG,
                                        self.best_accG)
        printer.print_best_acc_error(self.best_accG, self.epoch)
        if epoch % self.epoch_to_save == 0:
            model_save_util.save_model(params.models_save_path, epoch, self.output_dir, self.netG, self.optimizerG,
                                       self.netD, self.optimizerD)
            self.tester.save_test_images(epoch, self.output_dir, self.input_images_mean, self.netD, self.netG,
                                         self.mse_loss, self.ssim_loss, self.num_epochs)
            self.save_loss_plot(epoch, self.output_dir)
            self.tester.save_images_for_model(self.netG, self.output_dir, epoch)

    def save_gradient_flow(self, epoch):
        new_out_dir = os.path.join(self.output_dir, "gradient_flow")
        plt.savefig(os.path.join(new_out_dir, "gradient_flow_epoch=" + str(epoch)))
        plt.close()
