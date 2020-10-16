import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import tranforms as custom_transform
from utils import printer
import tranforms
import utils.data_loader_util as data_loader_util
import utils.hdr_image_util as hdr_image_util
import utils.plot_util as plot_util
import data_generator.create_dng_npy_data as create_dng_npy_data
import torch.nn.functional as F


class Tester:
    def __init__(self, device, loss_g_d_factor_, ssim_loss_g_factor_, log_factor_, args):
        self.args = args
        self.to_crop = args.add_frame
        self.data_trc = args.data_trc
        self.test_data_loader_npy, self.test_data_loader_ldr = \
            data_loader_util.load_test_data(args.dataset_properties, title="test")
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        self.G_accuracy_test, self.D_accuracy_real_test, self.D_accuracy_fake_test = [], [], []
        self.real_label, self.fake_label = 1, 0
        self.device = device
        self.test_D_losses, self.test_D_loss_fake, self.test_D_loss_real = [], [], []
        self.test_G_losses_d, self.test_G_loss_ssim = [], []
        self.ssim_loss_g_factor = ssim_loss_g_factor_
        self.loss_g_d_factor = loss_g_d_factor_
        self.test_num_iter = 0
        self.Q_arr, self.S_arr, self.N_arr = [], [], []
        self.log_factor = log_factor_
        self.use_contrast_ratio_f = args.use_contrast_ratio_f
        self.test_original_hdr_images = self.load_original_test_hdr_images(args.test_dataroot_original_hdr)
        self.max_normalization = custom_transform.MaxNormalization()
        self.min_max_normalization = custom_transform.MinMaxNormalization()
        self.clip_transform = custom_transform.Clip()
        self.apply_wind_norm = args.apply_wind_norm
        self.wind_size = args.ssim_window_size
        self.std_norm_factor = args.std_norm_factor
        self.manual_d_training = args.manual_d_training
        self.d_weight_mul_mode = args.d_weight_mul_mode


    def load_original_test_hdr_images(self, root):
        print("using input loader number ", self.args.wind_norm_option)
        original_hdr_images = []
        counter = 1
        for img_name in os.listdir(root):
            im_path = os.path.join(root, img_name)
            print(img_name)
            rgb_img, gray_im_log, f_factor = \
                create_dng_npy_data.hdr_preprocess(im_path,
                                                   self.args.factor_coeff, train_reshape=False,
                                                   gamma_log=self.args.gamma_log,
                                                   f_factor_path=self.args.f_factor_path,
                                                   use_new_f=self.args.use_new_f,
                                                   data_trc=self.data_trc, test_mode=False,
                                                   use_contrast_ratio_f=self.use_contrast_ratio_f)
            rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
            diffY, diffX = 0, 0
            if self.to_crop:
                print("original shape", gray_im_log.shape)
                diffY = hdr_image_util.closest_power(gray_im_log.shape[1]) - gray_im_log.shape[1]
                diffX = hdr_image_util.closest_power(gray_im_log.shape[2]) - gray_im_log.shape[2]
                gray_im_log = F.pad(gray_im_log.unsqueeze(dim=0), (diffX // 2, diffX - diffX // 2,
                                                                   diffY // 2, diffY - diffY // 2), mode='replicate')
                gray_im_log = torch.squeeze(gray_im_log, dim=0)
                print("new shape", gray_im_log.shape)
                # gray_im_log = data_loader_util.add_frame_to_im(gray_im_log)
            original_hdr_images.append({'im_name': os.path.splitext(img_name)[0],
                                        'im_hdr_original': rgb_img,
                                        'im_log_normalize_tensor': gray_im_log,
                                        'epoch': 0, 'diffX': diffX, 'diffY': diffY})
            counter += 1
        return original_hdr_images

    def get_test_image_special_factor(self, im_path, original_hdr_images, counter):
        special_factors = [self.args.factor_coeff * 0.01, 5, 10]
        for f in special_factors:
            rgb_img, gray_im_log = create_dng_npy_data.hdr_preprocess_change_f(im_path, self.args, f, reshape=True)
            rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
            if self.to_crop:
                gray_im_log = data_loader_util.add_frame_to_im(gray_im_log)
            original_hdr_images.append({'im_name': str(counter) + "_" + str(f),
                                        'im_hdr_original': rgb_img,
                                        'im_log_normalize_tensor': gray_im_log,
                                        'epoch': 0})

    def update_test_loss(self, netD, criterion, ssim_loss, b_size, num_epochs, first_b_tonemap, fake, hdr_input, epoch):
        self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
        with torch.no_grad():
            test_D_output_on_real = netD(first_b_tonemap.detach()).view(-1)
            self.accDreal_counter += (test_D_output_on_real > 0.5).sum().item()

            real_label = torch.full(test_D_output_on_real.shape, self.real_label, device=self.device)
            test_errD_real = criterion(test_D_output_on_real, real_label)
            self.test_D_loss_real.append(test_errD_real.item())

            fake_label = torch.full(test_D_output_on_real.shape, self.fake_label, device=self.device)
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
                if self.to_crop:
                    hdr_input = data_loader_util.crop_input_hdr_batch(hdr_input)
                fake_rgb_n = fake + 1
                hdr_input_rgb_n = hdr_input + 1
                test_errGssim = self.ssim_loss_g_factor * (1 - ssim_loss(fake_rgb_n, hdr_input_rgb_n))
                self.test_G_loss_ssim.append(test_errGssim.item())
            self.update_accuracy()
            printer.print_test_epoch_losses_summary(num_epochs, epoch, test_loss_D, test_errGd, self.accDreal_test,
                                                    self.accDfake_test, self.accG_test)

    @staticmethod
    def get_fake_test_images(first_b_hdr, netG):
        with torch.no_grad():
            fake = netG(first_b_hdr.detach())
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
        plot_util.plot_general_losses(self.test_G_losses_d, self.test_G_loss_ssim,
                                      self.test_D_loss_fake, self.test_D_loss_real,
                                      "TEST epoch loss = " + str(epoch), self.test_num_iter, loss_path,
                                      (self.loss_g_d_factor != 0), (self.ssim_loss_g_factor != 0))

        plot_util.plot_general_accuracy(self.G_accuracy_test, self.D_accuracy_fake_test, self.D_accuracy_real_test,
                                        "TEST epoch acc = " + str(epoch), epoch, acc_path)

    def save_test_images(self, epoch, out_dir, input_images_mean, netD, netG,
                         criterion, ssim_loss, num_epochs, add_frame):
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
        if self.manual_d_training:
            additional_channel = 1.0
            weight_channel = torch.full(test_hdr_batch_image.shape, additional_channel).type_as(test_hdr_batch_image)
            test_hdr_batch_image1 = torch.cat([test_hdr_batch_image, weight_channel], dim=1)
            fake1 = self.get_fake_test_images(test_hdr_batch_image1, netG)
            fake0 = fake1
            if not self.d_weight_mul_mode == "single":
                additional_channel = 0.0
                weight_channel = torch.full(test_hdr_batch_image.shape, additional_channel).type_as(
                    test_hdr_batch_image)
                test_hdr_batch_image0 = torch.cat([test_hdr_batch_image, weight_channel], dim=1)
                fake0 = self.get_fake_test_images(test_hdr_batch_image0, netG)

        else:
            fake1 = self.get_fake_test_images(test_hdr_batch_image, netG)
            fake0 = fake1

        plot_util.save_groups_images(test_hdr_batch, test_real_batch, fake1, fake0,
                                     new_out_dir, len(self.test_data_loader_npy.dataset), epoch,
                                     input_images_mean)

    def save_images_for_model(self, netG, out_dir, epoch):
        out_dir = os.path.join(out_dir, "model_results", str(epoch))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print("Directory ", out_dir, " created")
        with torch.no_grad():
            for im_and_q in self.test_original_hdr_images:
                print(im_and_q["im_name"])
                im_hdr_original = im_and_q['im_hdr_original']
                im_log_normalize_tensor = im_and_q['im_log_normalize_tensor'].unsqueeze(0).to(self.device)
                printer.print_g_progress(im_log_normalize_tensor, "input_tester")
                if self.manual_d_training:
                    file_name = im_and_q["im_name"] + "_1"
                    self.run_model_on_im_and_save_res(im_log_normalize_tensor, netG, im_hdr_original, out_dir,
                                                      file_name, 1.0, im_and_q['diffX'], im_and_q['diffY'])
                    if not self.d_weight_mul_mode == "single":
                        file_name = im_and_q["im_name"] + "_0"
                        self.run_model_on_im_and_save_res(im_log_normalize_tensor, netG, im_hdr_original, out_dir,
                                                          file_name, 0.0, im_and_q['diffX'], im_and_q['diffY'])
                        file_name = im_and_q["im_name"] + "_0.5"
                        self.run_model_on_im_and_save_res(im_log_normalize_tensor, netG, im_hdr_original, out_dir,
                                                          file_name, 0.5, im_and_q['diffX'], im_and_q['diffY'])

                else:
                    file_name = im_and_q["im_name"]
                    self.run_model_on_im_and_save_res(im_log_normalize_tensor, netG, im_hdr_original, out_dir,
                                                      file_name, None, im_and_q['diffX'], im_and_q['diffY'])

    def run_model_on_im_and_save_res(self, im_log_normalize_tensor, netG, im_hdr_original,
                                     out_dir, file_name, additional_channel, diffX, diffY):
        if additional_channel is not None:
            weight_channel = torch.full(im_log_normalize_tensor.shape, additional_channel).type_as(
                im_log_normalize_tensor)
            im_log_normalize_tensor = torch.cat([im_log_normalize_tensor, weight_channel], dim=1)
        with torch.no_grad():
            print(im_log_normalize_tensor.shape)
            fake = netG(im_log_normalize_tensor.detach(), apply_crop=False)
            print(fake.shape)
            if self.to_crop:
                fake = fake[:, :, diffY // 2:fake.shape[2] - (diffY - diffY // 2),
                       diffX // 2:fake.shape[3] - (diffX - diffX // 2)]
                print(fake.shape)
        printer.print_g_progress(fake, file_name)

        fake_im_gray_stretch = (fake[0] - fake[0].min()) / (fake[0].max() - fake[0].min())
        print(im_hdr_original.shape)
        print(fake_im_gray_stretch.shape)
        fake_im_color = hdr_image_util.back_to_color_batch(im_hdr_original.unsqueeze(0),
                                                           fake_im_gray_stretch.unsqueeze(dim=0))
        hdr_image_util.save_color_tensor_as_numpy(fake_im_color[0], out_dir, file_name)
