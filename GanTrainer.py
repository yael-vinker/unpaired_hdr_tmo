from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import torch.utils.data
from torch import autograd

import Tester
# matplotlib.use('Agg')
import params
import printer
import ssim
import tranforms as custom_transform
import utils.data_loader_util as data_loader_util
import utils.model_save_util as model_save_util
import utils.plot_util as plot_util
import torch.nn.functional as F

class GanTrainer:
    def __init__(self, opt, t_netG, t_netD, t_optimizerG, t_optimizerD, lr_scheduler_G, lr_scheduler_D):
        self.pyramid_loss = opt.pyramid_loss
        self.pyramid_weight_list = opt.pyramid_weight_list
        self.to_crop = opt.add_frame
        self.output_dir = opt.output_dir
        # self.writer = writer_
        self.batch_size = opt.batch_size
        self.num_epochs = opt.num_epochs
        self.device = opt.device
        self.netG = t_netG
        self.netD = t_netD
        self.optimizerD = t_optimizerD
        self.optimizerG = t_optimizerG
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.epoch_to_save = opt.epoch_to_save

        self.real_label, self.fake_label = 1, 0
        self.epoch, self.num_iter, self.test_num_iter = 0, 0, 0

        self.errG_d, self.errG_ssim = None, None
        self.best_accG = 0
        self.errD_real, self.errD_fake, self.errD = None, None, None
        self.accG, self.accD, self.accDreal, self.accDfake = None, None, None, None
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy, self.D_accuracy_real, self.D_accuracy_fake = [], [], []
        self.G_loss_ssim, self.G_loss_d = [], []
        self.D_losses, self.D_loss_fake, self.D_loss_real = [], [], []

        self.train_data_loader_npy, self.train_data_loader_ldr = \
            data_loader_util.load_data(opt.data_root_npy, opt.data_root_ldr,
                                       self.batch_size, addFrame=opt.add_frame, title="train")

        self.input_dim = opt.input_dim
        self.input_images_mean = opt.input_images_mean
        self.isCheckpoint = opt.isCheckpoint
        self.checkpoint = None
        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss_name = opt.ssim_loss
        if opt.ssim_loss == params.ssim_tmqi:
            self.ssim_loss = ssim.TMQI_SSIM(window_size=opt.ssim_window_size)
        else:
            self.ssim_loss = ssim.OUR_CUSTOM_SSIM(window_size=opt.ssim_window_size)

        self.loss_g_d_factor = opt.loss_g_d_factor
        self.ssim_loss_g_factor = opt.ssim_loss_factor
        self.log_factor = opt.log_factor
        self.transform_exp = custom_transform.Exp(opt.log_factor)
        self.normalize = custom_transform.Normalize(0.5, 0.5)
        self.use_transform_exp = opt.use_transform_exp
        self.tester = Tester.Tester(opt.test_dataroot_npy, opt.test_dataroot_ldr, opt.test_dataroot_original_hdr,
                                    opt.batch_size,
                                    self.device, self.loss_g_d_factor, self.ssim_loss_g_factor, opt.use_transform_exp,
                                    self.transform_exp, self.log_factor, opt.add_frame)
        # self.writer = self.init_writer("writer", "a")

    # def init_writer(self, log_dir, run_name):
    #     log_dir = os.path.join(log_dir, run_name, "train")
    #     os.makedirs(log_dir)
    #     # writer = SummaryWriter(log_dir)
    #     return writer

    def update_accuracy(self):
        len_hdr_train_dset = len(self.train_data_loader_npy.dataset)
        len_ldr_train_dset = len(self.train_data_loader_ldr.dataset)

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
        self.errD_real = self.mse_loss(output_on_real, label)
        self.errD_real.backward()

        # Train with all-fake batch
        # Generate fake image batch with G
        if self.use_transform_exp:
            fake = self.transform_exp(self.netG(hdr_input))
            fake = self.normalize(fake)
        else:
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

    def update_g_d_loss(self, output_on_fake, label):
        self.errG_d = self.loss_g_d_factor * (self.mse_loss(output_on_fake, label))
        self.errG_d.backward(retain_graph=True)
        self.G_loss_d.append(self.errG_d.item())



    def update_ssim_loss(self, hdr_input, fake):
        fake_rgb_n = fake + 1
        hdr_input_rgb_n = hdr_input + 1
        ssim_loss = 0
        if self.pyramid_loss:
            mssim = []
            for i in range(len(self.pyramid_weight_list)):
                import numpy as np
                plt.subplot(1,2,1)
                plt.imshow(np.squeeze(fake_rgb_n[0].clone().permute(1, 2, 0).detach().cpu().numpy()), cmap='gray')
                plt.subplot(1, 2, 2)
                plt.imshow(np.squeeze(hdr_input_rgb_n[0].clone().permute(1, 2, 0).detach().cpu().numpy()), cmap='gray')
                plt.show()
                ssim_loss = ssim_loss + (self.pyramid_weight_list[i] * (1 - self.ssim_loss(fake_rgb_n, hdr_input_rgb_n)))
                # mssim.append(sim)

                fake_rgb_n = F.avg_pool2d(fake_rgb_n, (2, 2))
                hdr_input_rgb_n = F.avg_pool2d(hdr_input_rgb_n, (2, 2))
            self.errG_ssim = self.ssim_loss_g_factor * ssim_loss
            print(self.errG_ssim)
            print(self.errG_ssim.sum())
        else:
            self.errG_ssim = self.ssim_loss_g_factor * (1 - self.ssim_loss(fake_rgb_n, hdr_input_rgb_n))
        self.errG_ssim.backward()
        self.G_loss_ssim.append(self.errG_ssim.item())

    def update_best_G_acc(self):
        if self.accG > self.best_accG:
            self.best_accG = self.accG
            printer.print_best_acc_error(self.best_accG, self.epoch)
            model_save_util.save_best_model(self.netG, self.output_dir, self.optimizerG)
            self.tester.save_images_for_best_model(self.netG, self.output_dir, self.epoch)

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
        if self.use_transform_exp:
            fake = self.transform_exp(self.netG(hdr_input))
            fake = self.normalize(fake)
        else:
            fake = self.netG(hdr_input)
        printer.print_g_progress(fake)
        output_on_fake = self.netD(fake).view(-1)
        # Real label = 1, so wo count number of samples on which G tricked D
        self.accG_counter += (output_on_fake > 0.5).sum().item()
        # updates all G's losses
        self.update_g_d_loss(output_on_fake, label)
        if self.to_crop:
            hdr_input = data_loader_util.crop_input_hdr_batch(hdr_input)
        self.update_ssim_loss(hdr_input, fake)
        self.optimizerG.step()

    def train_epoch(self):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        for (h, data_hdr), (l, data_ldr) in zip(enumerate(self.train_data_loader_npy, 0),
                                                enumerate(self.train_data_loader_ldr, 0)):
            self.num_iter += 1
            with autograd.detect_anomaly():
                real_ldr_cpu = data_ldr[params.gray_input_image_key].to(self.device)
                hdr_input = data_hdr[params.gray_input_image_key].to(self.device)
                hdr_input_display = data_hdr[params.color_image_key].to(self.device)
                b_size = hdr_input.size(0)
                label = torch.full((b_size,), self.real_label, device=self.device)
                self.train_D(hdr_input, real_ldr_cpu)
                self.train_G(label, hdr_input, hdr_input_display)
                # self.writer.add_scalars('Loss/train', {'err_g_ssim': self.errG_ssim.item(),
                #                                       'err_g_d': self.errG_d.item(),
                #                                       'err_d_real': self.errD_real.item(),
                #                                       'err_d_fake': self.errD_fake.item()}, self.num_iter)

                # self.writer.add_scalar("g_d", self.errG_d.item(), self.num_iter)
                plot_util.plot_grad_flow(self.netG.named_parameters(), self.output_dir, 1)
        self.update_accuracy()
        if self.epoch > 20:
            self.update_best_G_acc()

    def verify_checkpoint(self):
        if self.isCheckpoint:
            print("Loading model...")
            self.load_model()
            print("Model was loaded")
            print()

    def save_loss_plot(self, epoch, output_dir):
        loss_path = os.path.join(output_dir, "loss_plot")
        acc_path = os.path.join(output_dir, "accuracy")
        plot_util.plot_general_losses(self.G_loss_d, self.G_loss_ssim, self.D_loss_fake,
                                      self.D_loss_real, "summary epoch_=_" + str(epoch), self.num_iter, loss_path,
                                      (self.loss_g_d_factor != 0), (self.ssim_loss_g_factor != 0))
        plot_util.plot_general_accuracy(self.G_accuracy, self.D_accuracy_fake, self.D_accuracy_real,
                                        "accuracy epoch = " + str(epoch),
                                        self.epoch, acc_path)

    def train(self):
        printer.print_cuda_details(self.device.type)
        self.verify_checkpoint()
        start_epoch = self.epoch
        print("Starting Training Loop...")
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch += 1
            start = time.time()

            self.train_epoch()
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            new_out_dir = os.path.join(self.output_dir, "gradient_flow")
            plt.savefig(os.path.join(new_out_dir, "gradient_flow_epoch=" + str(epoch)))
            plt.close()

            print("Single [[epoch]] iteration took [%.4f] seconds\n" % (time.time() - start))

            printer.print_epoch_losses_summary(epoch, self.num_epochs, self.errD.item(), self.errD_real.item(),
                                               self.errD_fake.item(), self.loss_g_d_factor, self.errG_d,
                                               self.ssim_loss_g_factor, self.errG_ssim)
            printer.print_epoch_acc_summary(epoch, self.num_epochs, self.accDfake, self.accDreal, self.accG,
                                            self.best_accG)
            printer.print_best_acc_error(self.best_accG, self.epoch)
            if epoch % self.epoch_to_save == 0:
                model_save_util.save_model(params.models_save_path, epoch, self.output_dir, self.netG, self.optimizerG,
                                           self.netD, self.optimizerD)
                self.tester.save_test_images(epoch, self.output_dir, self.input_images_mean, self.netD, self.netG,
                                             self.mse_loss, self.ssim_loss, self.num_epochs)
                self.save_loss_plot(epoch, self.output_dir)
                self.tester.save_images_for_model(self.netG, self.output_dir, self.epoch)
                # self.tester.update_TMQI(self.netG, output_dir, epoch)
        # self.writer.close()

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

#
# if __name__ == '__main__':
#     import config
#
#     # batch_size, num_epochs, model, con_operator, filters, G_lr, D_lr, train_data_root_npy, train_data_root_ldr, isCheckpoint_str, \
#     # test_data_root_npy, test_data_root_ldr, result_dir_pref, input_dim, loss_g_d_factor, \
#     # ssim_loss_factor, input_images_mean, use_transform_exp, log_factor, test_dataroot_original_hdr, \
#     # epoch_to_save, depth, add_frame, ssim_loss, ssim_window_size, decay_epoch = parse_arguments()
#     torch.manual_seed(params.manualSeed)
#     opt = config.get_opt()
#
#     print("=====================")
#     print("BATCH SIZE:", opt.batch_size)
#     print("EPOCHS:", opt.num_epochs)
#     print("MODEL:", opt.model, opt.con_operator)
#     print("UNET DEPTH: ", opt.unet_depth)
#     print("ADD FRAME: ", bool(opt.add_frame))
#     print("G LR: ", opt.G_lr)
#     print("D LR: ", opt.D_lr)
#     print("CHECK POINT:", opt.isCheckpoint)
#     print("INPUT DIM:", opt.input_dim)
#     print("INPUT IMAGES MEAN:", opt.input_images_mean)
#     print("LOSS G D FACTOR:", opt.loss_g_d_factor)
#     print("SSIM LOSS FACTOR:", opt.ssim_loss_factor)
#     print("SSIM METRIC:", opt.ssim_loss)
#     print("SSIM WINDOW SIZE:", opt.ssim_window_size)
#     print("LOG FACTOR:", opt.log_factor)
#     print("DEVICE:", opt.device)
#     print("=====================\n")
#
#     # net_G = create_net("G_" + "unet3_layer", device, isCheckpoint, input_dim, input_images_mean)
#     net_G = model_save_util.create_net("G", opt.model, opt.device, opt.isCheckpoint, opt.input_dim,
#                                        opt.input_images_mean, opt.filters, opt.con_operator, opt.depth, opt.add_frame)
#
#     print("=================  NET G  ==================")
#     print(net_G)
#     summary(net_G, (opt.input_dim, 346, 346), device="cpu")
#     print()
#
#     net_D = model_save_util.create_net("D", "D", opt.device, opt.isCheckpoint, opt.input_dim, opt.input_images_mean)
#     print("=================  NET D  ==================")
#     print(net_D)
#     summary(net_D, (opt.input_dim, 256, 256), device="cpu")
#     print()
#
#     # Setup Adam optimizers for both G and D
#     optimizer_D = optim.Adam(net_D.parameters(), lr=opt.D_lr, betas=(params.beta1, 0.999))
#     optimizer_G = optim.Adam(net_G.parameters(), lr=opt.G_lr, betas=(params.beta1, 0.999))
#
#     # Learning rate update schedulers
#     lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
#         optimizer_G, lr_lambda=LambdaLR(opt.num_epochs, 0, opt.decay_epoch).step
#     )
#     lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
#         optimizer_D, lr_lambda=LambdaLR(opt.num_epochs, 0, opt.decay_epoch).step
#     )
#
#     # writer = Writer.Writer(g_t_utils.get_loss_path(result_dir_pref, model, params.loss_path))
#     writer = 1
#     output_dir = g_t_utils.create_dir(opt.result_dir_pref + "_log_" + str(opt.log_factor), opt.model, opt.con_operator,
#                                       params.models_save_path,
#                                       params.loss_path, params.results_path, opt.depth)
#
#     gan_trainer = GanTrainer(device, batch_size, num_epochs, train_data_root_npy, train_data_root_ldr,
#                              test_data_root_npy, test_data_root_ldr, isCheckpoint,
#                              net_G, net_D, optimizer_G, optimizer_D, input_dim, loss_g_d_factor,
#                              ssim_loss_factor, input_images_mean, writer, use_transform_exp, log_factor,
#                              test_dataroot_original_hdr, epoch_to_save, add_frame, ssim_loss, ssim_window_size,
#                              lr_scheduler_G, lr_scheduler_D)
#
#     gan_trainer.train(output_dir)
