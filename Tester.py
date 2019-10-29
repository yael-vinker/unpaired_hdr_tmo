import torch
import printer
import ProcessedDatasetFolder
import params
import os
import gan_trainer_utils as g_t_utils

class Tester:
    def __init__(self, test_dataroot_npy, test_dataroot_ldr, batch_size, device,
                 loss_g_d_factor_, ssim_loss_g_factor_, use_transform_exp_, transform_exp_):
        self.test_data_loader_npy, self.test_data_loader_ldr = \
            g_t_utils.load_data(test_dataroot_npy, test_dataroot_ldr, batch_size, testMode=True, title="test")
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy_test, self.D_accuracy_real_test, self.D_accuracy_fake_test = [], [], []
        self.real_label, self.fake_label = 1, 0
        self.device = device
        self.test_D_losses, self.test_D_loss_fake, self.test_D_loss_real = [], [], []
        self.test_G_losses_d, self.test_G_loss_ssim = [], []
        self.use_transform_exp = use_transform_exp_
        self.ssim_loss_g_factor = ssim_loss_g_factor_
        self.loss_g_d_factor = loss_g_d_factor_
        self.test_num_iter = 0
        self.transform_exp = transform_exp_

    def update_test_loss(self, netD, criterion, ssim_loss, b_size, num_epochs, first_b_tonemap, fake, hdr_input, epoch):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        with torch.no_grad():
            real_label = torch.full((b_size,), self.real_label, device=self.device)
            test_D_output_on_real = netD(first_b_tonemap.detach()).view(-1)
            self.accDreal_counter += (test_D_output_on_real > 0.5).sum().item()

            test_errD_real = criterion(test_D_output_on_real, real_label)
            self.test_D_loss_real.append(test_errD_real.item())

            fake_label = torch.full((b_size,), self.fake_label, device=self.device)
            if self.use_transform_exp:
                output_on_fake = netD(self.transform_exp(fake.detach())).view(-1)
            else:
                output_on_fake = netD(fake.detach()).view(-1)
            self.accDfake_counter += (output_on_fake <= 0.5).sum().item()

            test_errD_fake = criterion(output_on_fake, fake_label)
            test_loss_D = test_errD_real + test_errD_fake
            self.test_D_loss_fake.append(test_errD_fake.item())
            self.test_D_losses.append(test_loss_D.item())

            # output_on_fakake = self.netD(fake.detach()).view(-1)
            self.accG_counter += (output_on_fake > 0.5).sum().item()
            # if self.loss_g_d_factor != 0:
            test_errGd = criterion(output_on_fake, real_label)
            self.test_G_losses_d.append(test_errGd.item())
            if self.ssim_loss_g_factor != 0:
                test_errGssim = self.ssim_loss_g_factor * (1 - ssim_loss(fake, hdr_input))
                self.test_G_loss_ssim.append(test_errGssim.item())
            self.update_accuracy()
            printer.print_test_epoch_losses_summary(num_epochs, epoch, test_loss_D, test_errGd, self.accDreal_test,
                                                    self.accDfake_test, self.accG_test)

    @staticmethod
    def get_fake_test_images(first_b_hdr, netG):
        with torch.no_grad():
            fake = netG(first_b_hdr)
            return fake

    def update_accuracy(self):
        len_hdr_test_dset = len(self.test_data_loader_npy.dataset)
        len_ldr_test_dset = len(self.test_data_loader_ldr.dataset)

        self.accG_test = self.accG_counter / len_hdr_test_dset
        self.accDreal_test = self.accDreal_counter / len_ldr_test_dset
        self.accDfake_test = self.accDfake_counter / len_ldr_test_dset
        self.G_accuracy_test.append(self.accG_test)
        self.D_accuracy_real_test.append(self.accDreal_test)
        self.D_accuracy_fake_test.append(self.accDfake_test)

    def save_test_loss(self, epoch, out_dir):
        acc_path = os.path.join(out_dir, "accuracy")
        loss_path = os.path.join(out_dir, "loss_plot")
        g_t_utils.plot_general_losses(self.test_G_losses_d, self.test_G_loss_ssim,
                                      self.test_D_loss_fake, self.test_D_loss_real,
                                      "TEST epoch loss = " + str(epoch), self.test_num_iter, loss_path,
                                      (self.loss_g_d_factor != 0), (self.ssim_loss_g_factor != 0))

        g_t_utils.plot_general_accuracy(self.G_accuracy_test, self.D_accuracy_fake_test, self.D_accuracy_real_test,
                                        "TEST epoch acc = " + str(epoch), epoch, acc_path)

    def save_test_images(self, epoch, out_dir, input_images_mean, netD, netG, criterion, ssim_loss, num_epochs):
        out_dir = os.path.join(out_dir, "result_images")
        new_out_dir = os.path.join(out_dir, "images_epoch=" + str(epoch))

        if not os.path.exists(new_out_dir):
            os.mkdir(new_out_dir)

        self.test_num_iter += 1
        test_real_batch = next(iter(self.test_data_loader_ldr))
        test_real_first_b = test_real_batch["input_im"].to(self.device)

        test_hdr_batch = next(iter(self.test_data_loader_npy))
        # test_hdr_batch_image = test_hdr_batch[params.image_key].to(self.device)
        test_hdr_batch_image = test_hdr_batch["input_im"].to(self.device)

        fake = self.get_fake_test_images(test_hdr_batch_image, netG)
        fake_ldr = self.get_fake_test_images(test_real_batch["input_im"].to(self.device), netG)

        if self.use_transform_exp:
            g_t_utils.save_groups_images(test_hdr_batch, test_real_batch, self.transform_exp(fake), fake_ldr,
                                         new_out_dir, len(self.test_data_loader_npy.dataset), epoch,
                                         input_images_mean)
        else:
            g_t_utils.save_groups_images(test_hdr_batch, test_real_batch, fake, fake_ldr,
                                         new_out_dir, len(self.test_data_loader_npy.dataset), epoch,
                                         input_images_mean)
        self.update_test_loss(netD, criterion, ssim_loss, test_real_first_b.size(0), num_epochs,
                              test_real_first_b, fake, test_hdr_batch_image, epoch)
        # netD, criterion, ssim_loss, b_size, num_epochs, first_b_tonemap, fake, hdr_input, epoch):