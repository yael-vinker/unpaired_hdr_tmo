from __future__ import print_function

import os
import time
import matplotlib.pyplot as plt
import torch.utils.data
from torch import autograd
import numpy as np

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
from fid import fid_score


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
        self.num_D = opt.num_D
        self.d_pretrain_epochs = opt.d_pretrain_epochs
        self.pre_train_mode = False
        self.enhance_detail = opt.enhance_detail
        self.manual_d_training = opt.manual_d_training

        # ====== LOSS ======
        self.train_with_D = opt.train_with_D
        self.multi_scale_D = opt.multi_scale_D
        self.d_nlayers = opt.d_nlayers
        self.pyramid_weight_list = opt.pyramid_weight_list
        self.mse_loss = torch.nn.MSELoss()
        self.wind_size = opt.ssim_window_size
        self.struct_method = opt.struct_method
        if opt.ssim_loss_factor:
            self.struct_loss = ssim.StructLoss(window_size=opt.ssim_window_size,
                                                  pyramid_weight_list=opt.pyramid_weight_list,
                                                  pyramid_pow=False, use_c3=False,
                                                  apply_sig_mu_ssim=opt.apply_sig_mu_ssim,
                                                  struct_method=opt.struct_method,
                                                  std_norm_factor=opt.std_norm_factor,
                                                  crop_input=opt.add_frame)

        self.use_bilateral_weight = opt.apply_intensity_loss_laplacian_weights
        self.bilateral_sigma_r = opt.bilateral_sigma_r
        self.bilateral_mu = opt.bilateral_mu
        self.blf_input = opt.blf_input
        self.blf_alpha = opt.blf_alpha
        if opt.apply_intensity_loss:
            self.intensity_loss = ssim.IntensityLoss(opt.intensity_epsilon, opt.std_pyramid_weight_list, opt.alpha,
                                                         opt.std_method, opt.ssim_window_size, opt.add_frame)
            self.intensity_loss_factor = opt.apply_intensity_loss
        self.std_mul_max = opt.std_mul_max
        if opt.mu_loss_factor:
            self.mu_loss = ssim.MuLoss(opt.mu_pyramid_weight_list, opt.ssim_window_size, opt.add_frame)
        self.mu_loss_factor = opt.mu_loss_factor

        self.loss_g_d_factor = opt.loss_g_d_factor
        self.struct_loss_factor = opt.ssim_loss_factor
        self.errG_d, self.errG_struct, self.errG_intensity, self.errG_mu = None, None, None, None
        self.errD_real, self.errD_fake, self.errD = None, None, None
        self.accG, self.accD, self.accDreal, self.accDfake = None, None, None, None
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.G_loss_struct, self.G_loss_d, self.G_loss_intensity = [], [], []
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []
        self.apply_intensity_loss = opt.apply_intensity_loss
        self.adv_weight_list = opt.adv_weight_list
        self.strong_details_D_weights = opt.strong_details_D_weights
        self.basic_details_D_weights = opt.basic_details_D_weights
        self.d_weight_mul = 1.0
        self.d_weight_mul_mode = opt.d_weight_mul_mode

        # ====== DATASET ======
        self.train_data_loader_npy, self.train_data_loader_ldr = \
            data_loader_util.load_train_data(opt.dataset_properties, title="train")
        self.input_dim = opt.input_dim
        self.input_images_mean = opt.input_images_mean
        self.log_factor = opt.log_factor
        self.use_factorise_gamma_data = opt.use_factorise_gamma_data
        self.gamma_log = opt.gamma_log
        self.use_new_f = opt.use_new_f
        self.use_hist_fit = opt.use_hist_fit

        # ====== POST PROCESS ======
        self.to_crop = opt.add_frame
        self.use_normalization = opt.use_normalization
        self.normalization = opt.normalization

        # ====== SAVE RESULTS ======
        self.output_dir = opt.output_dir
        self.epoch_to_save = opt.epoch_to_save
        self.best_accG = 0
        self.tester = Tester.Tester(self.device, self.loss_g_d_factor, self.struct_loss_factor,
                                    self.log_factor, opt)
        self.final_epoch = opt.final_epoch
        self.fid_real_path = opt.fid_real_path
        self.fid_res_path = opt.fid_res_path

    def train(self):
        printer.print_cuda_details(self.device.type)
        self.verify_checkpoint()
        start_epoch = self.epoch
        if self.d_pretrain_epochs:
            self.pre_train_mode = True
            print("Starting Discriminator Pre-training Loop...")
            for epoch in range(self.d_pretrain_epochs):
                self.train_epoch()
                self.lr_scheduler_D.step()
                printer.print_epoch_acc_summary(epoch, self.d_pretrain_epochs, self.accDfake, self.accDreal, self.accG)
        self.save_loss_plot(self.d_pretrain_epochs, self.output_dir)
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.pre_train_mode = False
        self.num_iter = 0
        print("\nStarting Training Loop...")
        for epoch in range(start_epoch, self.num_epochs):
            start = time.time()
            self.epoch += 1
            self.train_epoch()
            self.lr_scheduler_G.step()
            if self.train_with_D:
                self.lr_scheduler_D.step()
            self.print_epoch_summary(epoch, start)

    def train_epoch(self):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        for (h, data_hdr), (l, data_ldr) in zip(enumerate(self.train_data_loader_npy, 0),
                                                enumerate(self.train_data_loader_ldr, 0)):
            self.num_iter += 1
            if not self.d_weight_mul_mode == "single":
                self.d_weight_mul = self.num_iter % 2
            with autograd.detect_anomaly():
                real_ldr = data_ldr[params.gray_input_image_key].to(self.device)
                hdr_input = self.get_hdr_input(data_hdr)
                hdr_original_gray_norm = data_hdr[params.original_gray_norm_key].to(self.device)
                hdr_original_gray = data_hdr[params.original_gray_key].to(self.device)
                gamma_factor = data_hdr[params.gamma_factor].to(self.device)
                if self.manual_d_training and not self.d_weight_mul_mode == "single":
                    # self.adv_weight_list = self.d_weight_mul * self.strong_details_D_weights + \
                    #                        (1 - self.d_weight_mul) * self.basic_details_D_weights
                    self.pyramid_weight_list = self.d_weight_mul * self.strong_details_D_weights + \
                                           (1 - self.d_weight_mul) * self.basic_details_D_weights
                if self.train_with_D:
                    self.train_D(hdr_input, real_ldr)
                if not self.pre_train_mode:
                    self.train_G(hdr_input, hdr_original_gray_norm, hdr_original_gray, gamma_factor)
        self.update_accuracy()

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
        self.D_real_pass(real_ldr)
        self.D_fake_pass(hdr_input)
        # Add the gradients from the all-real and all-fake batches
        self.errD = self.errD_real + self.errD_fake
        # Update D
        self.optimizerD.step()
        self.D_losses.append(self.errD.item())
        self.D_loss_fake.append(self.errD_fake.item())
        self.D_loss_real.append(self.errD_real.item())

    def D_real_pass(self, real_ldr):
        # Forward pass real batch through D
        if self.enhance_detail:
            output_on_real = self.netD(ssim.enhance_details(real_ldr, self.wind_size))
        else:
            output_on_real = self.netD(real_ldr)
        if "multiLayerD" in self.d_model:
            loss = []
            for i, input_i in zip(range(len(output_on_real)), output_on_real):
                pred = input_i[-1]
                target_tensor = torch.full(pred.shape, self.real_label, device=self.device)
                loss.append(self.adv_weight_list[i] * self.mse_loss(pred, target_tensor))
                self.accDreal_counter += (pred > 0.5).sum().item()
            self.errD_real = torch.sum(torch.stack(loss))
        else:
            output_on_real = output_on_real.view(-1)
            # Real label = 1, so we count the samples on which D was right
            self.accDreal_counter += (output_on_real > 0.5).sum().item()

            # Calculate loss on all-real batch
            label = torch.full(output_on_real.shape, self.real_label, device=self.device)
            full_scale_loss = self.mse_loss(output_on_real, label)
            if self.multi_scale_D:
                real_ldr = F.interpolate(real_ldr, scale_factor=0.5, mode='bicubic', align_corners=False)
                real_ldr = real_ldr.clamp(0, 1)
                output_on_real = self.netD(real_ldr).view(-1)
                label = torch.full(output_on_real.shape, self.real_label, device=self.device)
                half_scale_loss = self.mse_loss(output_on_real, label)
                self.errD_real = full_scale_loss + half_scale_loss
            else:
                self.errD_real = full_scale_loss
        self.errD_real.backward()

    def D_fake_pass(self, hdr_input):
        # Train with all-fake batch
        # Generate fake image batch with G
        if not self.pre_train_mode:
            fake = self.netG(hdr_input)
        else:
            fake = hdr_input
            if self.to_crop:
                fake = data_loader_util.crop_input_hdr_batch(hdr_input)

        # Classify all fake batch with D
        if self.enhance_detail:
            output_on_fake = self.netD(ssim.enhance_details(fake.detach(), self.wind_size).detach())
        else:
            output_on_fake = self.netD(fake.detach())

        if "multiLayerD" in self.d_model:
            loss = []
            for i, input_i in zip(range(len(output_on_fake)), output_on_fake):
                pred = input_i[-1]
                target_tensor = torch.full(pred.shape, self.fake_label, device=self.device)
                loss.append(self.adv_weight_list[i] * self.mse_loss(pred, target_tensor))
                self.accDfake_counter += (pred <= 0.5).sum().item()
            self.errD_fake = torch.sum(torch.stack(loss))

        else:
            output_on_fake = output_on_fake.view(-1)
            label = torch.full(output_on_fake.shape, self.fake_label, device=self.device)
            # Fake label = 0, so we count the samples on which D was right
            self.accDfake_counter += (output_on_fake <= 0.5).sum().item()
            # Calculate D's loss on the all-fake batch
            full_scale_loss = self.mse_loss(output_on_fake, label)
            if self.multi_scale_D:
                fake = F.interpolate(fake, scale_factor=0.5, mode='bicubic', align_corners=False)
                fake = fake.clamp(0, 1)
                output_on_fake = self.netD(fake.detach()).view(-1)
                label = torch.full(output_on_fake.shape, self.fake_label, device=self.device)
                half_scale_loss = self.mse_loss(output_on_fake, label)
                self.errD_fake = full_scale_loss + half_scale_loss
            else:
                self.errD_fake = full_scale_loss
            # Calculate the gradients for this batch
        self.errD_fake.backward()

    def train_G(self, hdr_input, hdr_original_gray_norm, hdr_original_gray, gamma_factor):
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
        if self.train_with_D:
            if self.enhance_detail:
                output_on_fake = self.netD(ssim.enhance_details(fake, self.wind_size))
            else:
                output_on_fake = self.netD(fake)
            self.update_g_d_loss(output_on_fake)

        if self.use_bilateral_weight:
            blf_input = hdr_input
            if self.blf_input == "log":
                blf_input = ssim.get_blf_log_input(hdr_original_gray_norm, gamma_factor, alpha=self.blf_alpha)
            r_weights = ssim.get_radiometric_weights(blf_input, self.wind_size, self.bilateral_sigma_r,
                                                     self.bilateral_mu, self.blf_input)
        else:
            r_weights = None
        if self.manual_d_training:
            hdr_input = hdr_input[:, :1, :, :]
        self.update_struct_loss(hdr_input, hdr_original_gray_norm, fake, r_weights)
        self.update_intensity_loss(fake, hdr_input, hdr_original_gray_norm, r_weights, gamma_factor, hdr_original_gray)
        self.update_mu_loss(hdr_original_gray_norm, fake, hdr_input, r_weights)
        self.optimizerG.step()

    def get_hdr_input(self, data_hdr):
        hdr_input = data_hdr[params.gray_input_image_key]
        if self.manual_d_training and not self.pre_train_mode:
            weight_channel = torch.full(hdr_input.shape, self.d_weight_mul).type_as(hdr_input)
            hdr_input = torch.cat([hdr_input, weight_channel], dim=1)
        return hdr_input.to(self.device)


    def update_g_d_loss(self, output_on_fake):
        if "multiLayerD" in self.d_model:
            loss = []
            for i, input_i in zip(range(len(output_on_fake)), output_on_fake):
                pred = input_i[-1]
                target_tensor = torch.full(pred.shape, self.real_label, device=self.device)
                loss.append(self.adv_weight_list[i] * self.mse_loss(pred, target_tensor))
                self.accG_counter += (pred > 0.5).sum().item()
            self.errG_d = self.loss_g_d_factor * torch.sum(torch.stack(loss))
        else:
            output_on_fake = output_on_fake.view(-1)
            # Real label = 1, so wo count number of samples on which G tricked D
            self.accG_counter += (output_on_fake > 0.5).sum().item()
            # fake labels are real for generator cost
            label = torch.full(output_on_fake.shape, self.real_label, device=self.device)
            self.errG_d = self.loss_g_d_factor * (self.mse_loss(output_on_fake, label))
        retain_graph = False
        if self.struct_loss_factor:
            retain_graph = True
        self.errG_d.backward(retain_graph=retain_graph)
        self.G_loss_d.append(self.errG_d.item())

    def update_struct_loss(self, hdr_input, hdr_input_original_gray_norm, fake, r_weights):
        if self.struct_loss_factor:
            self.errG_struct = self.struct_loss_factor * self.struct_loss(fake, hdr_input_original_gray_norm,
                                                                            hdr_input, r_weights, self.pyramid_weight_list)
            retain_graph = False
            if self.apply_intensity_loss or self.mu_loss_factor:
                retain_graph = True
            self.errG_struct.backward(retain_graph=retain_graph)
            self.G_loss_struct.append(self.errG_struct.item())

    def update_intensity_loss(self, fake, hdr_input, hdr_original_gray_norm, r_weights, gamma_factor, hdr_original_gray):
        if not self.std_mul_max:
            hdr_original_gray = None
        if self.apply_intensity_loss:
            self.errG_intensity = self.intensity_loss_factor * self.intensity_loss(fake, hdr_input,
                                                                                   hdr_original_gray_norm,
                                                                                   r_weights, gamma_factor,
                                                                                   hdr_original_gray)
            retain_graph = False
            if self.mu_loss_factor:
                retain_graph = True
            self.errG_intensity.backward(retain_graph=retain_graph)
            self.G_loss_intensity.append(self.errG_intensity.item())

    def update_mu_loss(self, hdr_input_original_gray, fake, hdr_input, r_weights):
        if self.mu_loss_factor:
            self.errG_mu = self.mu_loss_factor * self.mu_loss(fake, hdr_input_original_gray, hdr_input, r_weights)
            self.errG_mu.backward()

    def verify_checkpoint(self):
        if self.isCheckpoint:
            print("Loading model...")
            self.load_model()
            print("Model was loaded")
            print()

    def save_loss_plot(self, epoch, output_dir):
        loss_path = os.path.join(output_dir, "loss_plot")
        acc_path = os.path.join(output_dir, "accuracy")
        acc_file_name = "acc" + str(epoch)
        if self.pre_train_mode:
            acc_file_name = "pretrain_" + acc_file_name
        plot_util.plot_general_accuracy(self.G_accuracy, self.D_accuracy_fake, self.D_accuracy_real,
                                        acc_file_name,
                                        self.epoch, acc_path)
        if not self.pre_train_mode:
            plot_util.plot_general_losses(self.G_loss_d, self.G_loss_struct, self.G_loss_intensity, self.D_loss_fake,
                                          self.D_loss_real, "summary epoch_=_" + str(epoch), self.num_iter, loss_path,
                                          (self.loss_g_d_factor != 0), (self.struct_loss_factor != 0))

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
            self.accG = self.accG / params.patchD_map_dim[self.d_nlayers]
            self.accDreal = self.accDreal / params.patchD_map_dim[self.d_nlayers]
            self.accDfake = self.accDfake / params.patchD_map_dim[self.d_nlayers]
        elif "multiLayerD_patchD" == self.d_model:
            self.accG = self.accG / params.get_multiLayerD_map_dim(num_D=self.num_D, d_nlayers=self.d_nlayers)
            self.accDreal = self.accDreal / params.get_multiLayerD_map_dim(num_D=self.num_D, d_nlayers=self.d_nlayers)
            self.accDfake = self.accDfake / params.get_multiLayerD_map_dim(num_D=self.num_D, d_nlayers=self.d_nlayers)
        elif "multiLayerD_dcgan" == self.d_model or "multiLayerD_simpleD" == self.d_model:
            self.accG = self.accG / self.num_D
            self.accDreal = self.accDreal / self.num_D
            self.accDfake = self.accDfake / self.num_D

        self.G_accuracy.append(self.accG)
        self.D_accuracy_real.append(self.accDreal)
        self.D_accuracy_fake.append(self.accDfake)

    def print_epoch_summary(self, epoch, start):
        print("Single [[epoch]] iteration took [%.4f] seconds\n" % (time.time() - start))
        if self.train_with_D:
            printer.print_epoch_losses_summary(epoch, self.num_epochs, self.errD.item(), self.errD_real.item(),
                                               self.errD_fake.item(), self.loss_g_d_factor, self.errG_d,
                                               self.struct_loss_factor, self.errG_struct, self.errG_intensity,
                                               self.errG_mu)
        else:
            printer.print_epoch_losses_summary(epoch, self.num_epochs, 0, 0, 0, 0, 0,
                                               self.struct_loss_factor, self.errG_struct, self.errG_intensity,
                                               self.errG_mu)
        printer.print_epoch_acc_summary(epoch, self.num_epochs, self.accDfake, self.accDreal, self.accG)
        if epoch % self.epoch_to_save == 0:
            self.tester.save_test_images(epoch, self.output_dir, self.input_images_mean, self.netD, self.netG,
                                         self.mse_loss, self.struct_loss, self.num_epochs, self.to_crop)
            self.save_loss_plot(epoch, self.output_dir)
            self.tester.save_images_for_model(self.netG, self.output_dir, epoch)
        print(self.output_dir)
        if epoch == self.final_epoch:
            model_save_util.save_model(params.models_save_path, epoch, self.output_dir, self.netG, self.optimizerG,
                                       self.netD, self.optimizerD)
            self.save_data_for_assessment()

    def save_gradient_flow(self, epoch):
        new_out_dir = os.path.join(self.output_dir, "gradient_flow")
        plt.savefig(os.path.join(new_out_dir, "gradient_flow_epoch=" + str(epoch)))
        plt.close()

    def save_data_for_assessment(self):
        model_params = model_save_util.get_model_params(self.output_dir,
                                                        train_settings_path=os.path.join(self.output_dir,
                                                                                         "run_settings.npy"))
        model_params["test_mode_f_factor"] = False
        model_params["test_mode_frame"] = True
        net_path = os.path.join(self.output_dir, "models", "net_epoch_" + str(self.final_epoch) + ".pth")
        # self.run_model_on_path("open_exr_exr_format", "exr", model_params, net_path)
        # self.run_model_on_path("npy_pth", "npy", model_params, net_path)
        self.run_model_on_path("test_source", "exr", model_params, net_path)


    def run_model_on_path(self, data_source, data_format, model_params, net_path):
        input_images_path = model_save_util.get_hdr_source_path(data_source)
        f_factor_path = model_save_util.get_f_factor_path(data_source, self.gamma_log, self.use_new_f, self.use_hist_fit)
        output_images_path = os.path.join(self.output_dir, data_format + "_" + str(self.final_epoch))
        if not os.path.exists(output_images_path):
            os.mkdir(output_images_path)
        output_images_path_color_stretch = os.path.join(output_images_path, "color_stretch")
        output_images_path_gray_stretch = os.path.join(output_images_path, "gray_stretch")
        if not os.path.exists(output_images_path_color_stretch):
            os.mkdir(output_images_path_color_stretch)
        if not os.path.exists(output_images_path_gray_stretch):
            os.mkdir(output_images_path_gray_stretch)
        model_save_util.run_model_on_path(model_params, self.device, net_path, input_images_path,
                                          output_images_path, f_factor_path, self.netG, input_images_path)
        if data_format == "npy":
            fid_res_color_stretch = fid_score.calculate_fid_given_paths([self.fid_real_path, output_images_path_color_stretch],
                                                batch_size=20, cuda=False, dims=768)
            fid_res_gray_stretch = fid_score.calculate_fid_given_paths([self.fid_real_path, output_images_path_gray_stretch],
                                                batch_size=20, cuda=False, dims=768)
            if os.path.exists(self.fid_res_path):
                data = np.load(self.fid_res_path, allow_pickle=True)[()]
                data[model_params["model_name"]] = {"fid_res_color_stretch": fid_res_color_stretch, "fid_res_gray_stretch": fid_res_gray_stretch}
                np.save(self.fid_res_path, data)
            else:
                my_res = {model_params["model_name"]: {"fid_res_color_stretch": fid_res_color_stretch, "fid_res_gray_stretch": fid_res_gray_stretch}}
                np.save(self.fid_res_path, my_res)
