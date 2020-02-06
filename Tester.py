import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

# from old_files import TMQI
from utils import printer
import tranforms
import utils.data_loader_util as data_loader_util
import utils.hdr_image_util as hdr_image_util
import utils.image_quality_assessment_util as image_quality_assessment_util
import utils.plot_util as plot_util


class Tester:
    def __init__(self, test_dataroot_npy, test_dataroot_ldr, test_dataroot_original_hdr, batch_size, device,
                 loss_g_d_factor_, ssim_loss_g_factor_, use_transform_exp_, transform_exp_, log_factor_, addFrame_,
                 args):
        self.args = args
        self.to_crop = addFrame_
        self.test_data_loader_npy, self.test_data_loader_ldr = \
            data_loader_util.load_data(test_dataroot_npy, test_dataroot_ldr, batch_size, addFrame=addFrame_,
                                       title="test")
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
        self.Q_arr, self.S_arr, self.N_arr = [], [], []
        self.log_factor = log_factor_
        self.test_original_hdr_images = self.load_original_test_hdr_images(test_dataroot_original_hdr)
        self.normalize = tranforms.Normalize(0.5, 0.5)


    def load_original_test_hdr_images(self, root):
        import data_generator.create_dng_npy_data as create_dng_npy_data
        original_hdr_images = []
        counter = 1
        for img_name in os.listdir(root):
            im_path = os.path.join(root, img_name)
            rgb_img, gray_im_log = create_dng_npy_data.hdr_preprocess(im_path, self.args, reshape=True)
            rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
            if self.to_crop:
                gray_im_log = data_loader_util.add_frame_to_im(gray_im_log)
            text = ""
            original_hdr_images.append({'im_name': str(counter),
                                        'im_hdr_original': rgb_img,
                                        'im_log_normalize_tensor': gray_im_log,
                                        'Q_arr': [],
                                        'N_arr': [],
                                        'S_arr': [],
                                        'best_Q': 0,
                                        'Q_gray': 0,
                                        'epoch': 0,
                                        'other_results': text})
            counter += 1
        return original_hdr_images

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
                output_on_fake = netD(self.normalize(fake.detach())).view(-1)
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
        plot_util.plot_general_losses(self.test_G_losses_d, self.test_G_loss_ssim,
                                      self.test_D_loss_fake, self.test_D_loss_real,
                                      "TEST epoch loss = " + str(epoch), self.test_num_iter, loss_path,
                                      (self.loss_g_d_factor != 0), (self.ssim_loss_g_factor != 0))

        plot_util.plot_general_accuracy(self.G_accuracy_test, self.D_accuracy_fake_test, self.D_accuracy_real_test,
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
            fake = self.transform_exp(fake)
        plot_util.save_groups_images(test_hdr_batch, test_real_batch, fake, fake_ldr,
                                     new_out_dir, len(self.test_data_loader_npy.dataset), epoch,
                                     input_images_mean)
        self.update_test_loss(netD, criterion, ssim_loss, test_real_first_b.size(0), num_epochs,
                              test_real_first_b, fake, test_hdr_batch_image, epoch)

    def display_graph_and_image(self, im, im_and_q, graph_Q, graph_N, graph_S, path):
        plt.figure(figsize=(15, 15))
        plt.subplot(2, 1, 1)
        plt.axis("off")
        title = self.get_tmqi_graph_title(im_and_q)
        plt.title(title)
        plt.imshow(im)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(graph_Q)), graph_Q, '-r', label='Q')

        plt.plot(range(len(graph_N)), graph_N, '-b', label='N')
        plt.plot(range(len(graph_S)), graph_S, '-y', label='S')

        plt.xlabel("n iteration")
        plt.legend(loc='upper left')

        # save image
        plt.savefig(path)  # should before show method
        plt.close()

    @staticmethod
    def update_tmqi_arr(im_and_q, Q, S, N):
        if not np.isnan(Q):
            im_and_q['Q_arr'].append(Q)
        else:
            im_and_q['Q_arr'].append(0)
        if not np.isnan(N):
            im_and_q['N_arr'].append(N)
        else:
            im_and_q['N_arr'].append(0)
        if not np.isnan(S):
            im_and_q['S_arr'].append(S)
        else:
            im_and_q['S_arr'].append(0)

    def get_tmqi_plt_im_title(self, Q, S, N, epoch, im):
        title = 'Q = ' + str(Q) + 'S = ' + str(S) + 'N = ' + str(N) + " epoch = " + str(epoch) + \
                "min = " + str(np.min(im)) + " max = " + str(np.max(im))
        return title

    def get_tmqi_graph_title(self, im_and_q):
        return "best Q = " + str(im_and_q['best_Q']) + " epoch = " + str(im_and_q['epoch'])

    def save_tmqi_plt_result(self, out_dir, im_and_q, Q, S, N, epoch, im, color='rgb'):
        plt.figure(figsize=(15, 15))
        plt.axis("off")
        title = self.get_tmqi_plt_im_title(Q, S, N, epoch, im)
        plt.title(title)
        if color == 'gray':
            plt.imshow(im, cmap='gray')
        else:
            plt.imshow(im)
        plt.savefig(os.path.join(out_dir, im_and_q["im_name"] + "_plt_" + color))
        plt.close()

    def save_tmqi_result_imageio(self, out_dir, im_and_q, im, color):
        file_name = self.get_rgb_imageio_im_file_name(im_and_q, color)
        im = (im * 255).astype('uint8')
        imageio.imwrite(os.path.join(out_dir, file_name), im, format='PNG-FI')

    def save_best_acc_result_imageio(self, out_dir, im_and_q, im, epoch, color):
        file_name = im_and_q["im_name"] + "_epoch_" + str(epoch) + "_" + color + ".png"
        im = (im * 255).astype('uint8')
        imageio.imwrite(os.path.join(out_dir, file_name), im, format='PNG-FI')

    def get_rgb_imageio_im_file_name(self, im_and_q, color):
        return im_and_q["im_name"] + "_imageio_" + color + ".png"

    def update_best_Q(self, im_and_q, Q, epoch, im):
        im_and_q['best_Q'] = Q
        im_and_q['epoch'] = epoch
        # hdr_image_utils.print_image_details(im, "fake_im_color")

    def update_TMQI(self, netG, out_dir, epoch):
        out_dir = os.path.join(out_dir, "tmqi")
        with torch.no_grad():
            for im_and_q in self.test_original_hdr_images:
                im_hdr_original = im_and_q['im_hdr_original']
                im_log_normalize_tensor = im_and_q['im_log_normalize_tensor'].to(self.device)
                fake = netG(im_log_normalize_tensor.unsqueeze(0).detach())
                if self.use_transform_exp:
                    fake = self.transform_exp(fake)
                fake_im_gray = torch.squeeze(fake, dim=0)
                fake_im_gray_numpy = fake_im_gray.clone().permute(1, 2, 0).detach().cpu().numpy()
                fake_im_color = hdr_image_util.back_to_color(im_hdr_original,
                                                             fake_im_gray_numpy)

                Q, S, N = TMQI.TMQI(im_hdr_original, hdr_image_util.to_0_1_range(fake_im_color))
                self.update_tmqi_arr(im_and_q, Q, S, N)
                self.display_graph_and_image(fake_im_color, im_and_q, im_and_q['Q_arr'], im_and_q['N_arr'],
                                             im_and_q['S_arr'], os.path.join(out_dir, (
                        im_and_q["im_name"]) + "_graph" + ".png"))
                image_quality_assessment_util.save_text_to_image(out_dir,
                                                                 im_and_q['other_results'], im_and_q["im_name"])
                if Q > im_and_q['best_Q']:
                    self.update_best_Q(im_and_q, Q, epoch, fake_im_color)
                    self.save_tmqi_plt_result(out_dir, im_and_q, Q, S, N, epoch,
                                              hdr_image_util.to_0_1_range(fake_im_color), color='rgb')
                    self.save_tmqi_result_imageio(out_dir, im_and_q, fake_im_color, color='rgb')
                    self.save_tmqi_result_imageio(out_dir, im_and_q, hdr_image_util.to_0_1_range(fake_im_color),
                                                  color='stretch_rgb')
                    printer.print_tmqi_update(Q, color='rgb')
                fake_im_gray_numpy_0_1 = np.squeeze(hdr_image_util.to_0_1_range(fake_im_gray_numpy))
                Q_gray, S_gray, N_gray = TMQI.TMQI(im_hdr_original, np.squeeze(fake_im_gray_numpy))
                if Q_gray > im_and_q['Q_gray']:
                    im_and_q['Q_gray'] = Q_gray
                    self.save_tmqi_plt_result(out_dir, im_and_q, Q_gray, S_gray, N_gray, epoch, fake_im_gray_numpy_0_1,
                                              color='gray')
                    self.save_tmqi_result_imageio(out_dir, im_and_q, fake_im_gray_numpy_0_1, color='gray')
                    printer.print_tmqi_update(Q_gray, color='gray')
                    # hdr_image_utils.print_image_details(fake_im_gray_numpy_0_1, "fake_im_gray_numpy")

    def save_images_for_best_model(self, netG, out_dir, epoch):
        out_dir = os.path.join(out_dir, "best_acc_images")
        with torch.no_grad():
            for im_and_q in self.test_original_hdr_images:
                im_hdr_original = im_and_q['im_hdr_original']
                im_log_normalize_tensor = im_and_q['im_log_normalize_tensor'].to(self.device)
                fake = netG(im_log_normalize_tensor.unsqueeze(0).detach())
                if self.use_transform_exp:
                    fake = self.transform_exp(fake)
                fake_im_gray = torch.squeeze(fake, dim=0)
                fake_im_gray_numpy = fake_im_gray.clone().permute(1, 2, 0).detach().cpu().numpy()
                im_hdr_original = im_hdr_original.clone().permute(1, 2, 0).detach().cpu().numpy()
                fake_im_color = hdr_image_util.back_to_color(im_hdr_original,
                                                             fake_im_gray_numpy, self.args.use_normalization)
                fake_im_color = (255 * fake_im_color).astype('uint8')
                self.save_best_acc_result_imageio(out_dir, im_and_q, fake_im_color, epoch, color='rgb')
                fake_im_gray_numpy_0_1 = np.squeeze(hdr_image_util.to_0_1_range(fake_im_gray_numpy))
                self.save_best_acc_result_imageio(out_dir, im_and_q, fake_im_gray_numpy_0_1, epoch, color='gray')

    def save_images_for_model(self, netG, out_dir, epoch):
        out_dir = os.path.join(out_dir, "model_results", str(epoch))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print("Directory ", out_dir, " created")
        with torch.no_grad():
            for im_and_q in self.test_original_hdr_images:
                im_hdr_original = im_and_q['im_hdr_original']
                im_log_normalize_tensor = im_and_q['im_log_normalize_tensor'].to(self.device)
                fake = netG(im_log_normalize_tensor.unsqueeze(0).detach())
                if self.use_transform_exp:
                    fake = self.transform_exp(fake)
                fake_im_gray = torch.squeeze(fake, dim=0)
                fake_im_gray_numpy = fake_im_gray.clone().permute(1, 2, 0).detach().cpu().numpy()
                im_hdr_original = im_hdr_original.clone().permute(1, 2, 0).detach().cpu().numpy()
                fake_im_color = hdr_image_util.back_to_color(im_hdr_original,
                                                             fake_im_gray_numpy, self.args.use_normalization)
                fake_im_color = (255 * fake_im_color).astype('uint8')
                self.save_best_acc_result_imageio(out_dir, im_and_q, fake_im_color, epoch, color='rgb')
                fake_im_gray_numpy_0_1 = np.squeeze(hdr_image_util.to_0_1_range(fake_im_gray_numpy))
                self.save_best_acc_result_imageio(out_dir, im_and_q, fake_im_gray_numpy_0_1, epoch, color='gray')
