# import torch
#
# class Tester:
#     def __init__(self):
#
#     def update_test_loss(self, b_size, first_b_tonemap, fake, hdr_input, epoch):
#         self.accG_counter, self.accDreal_counter, self.accDfake_counter = 0, 0, 0
#         with torch.no_grad():
#             real_label = torch.full((b_size,), self.real_label, device=self.device)
#             test_D_output_on_real = self.netD(first_b_tonemap.detach()).view(-1)
#             self.accDreal_counter += (test_D_output_on_real > 0.5).sum().item()
#
#             test_errD_real = self.criterion(test_D_output_on_real, real_label)
#             self.test_D_loss_real.append(test_errD_real.item())
#
#             fake_label = torch.full((b_size,), self.fake_label, device=self.device)
#             output_on_fake = self.netD(fake.detach()).view(-1)
#             self.accDfake_counter += (output_on_fake <= 0.5).sum().item()
#
#             test_errD_fake = self.criterion(output_on_fake, fake_label)
#             test_loss_D = test_errD_real + test_errD_fake
#             self.test_D_loss_fake.append(test_errD_fake.item())
#             self.test_D_losses.append(test_loss_D.item())
#
#             output_on_fake = self.netD(fake.detach()).view(-1)
#             self.accG_counter += (output_on_fake > 0.5).sum().item()
#             # if self.loss_g_d_factor != 0:
#             test_errGd = self.criterion(output_on_fake, real_label)
#             self.test_G_losses_d.append(test_errGd.item())
#             if self.ssim_loss_g_factor != 0:
#                 test_errGssim = self.ssim_loss_g_factor * (1 - self.ssim_loss(fake, hdr_input))
#                 self.test_G_loss_ssim.append(test_errGssim.item())
#             if self.rgb_l2_loss_g_factor != 0:
#                 new_fake = g_t_utils.get_rgb_normalize_im_batch(fake)
#                 new_hdr_unput = g_t_utils.get_rgb_normalize_im_batch(hdr_input)
#                 test_errRGBl2 = self.rgb_l2_loss_g_factor * (self.mse_loss(new_fake, new_hdr_unput) / (new_fake.shape[1] * new_fake.shape[2] * new_fake.shape[3]))
#                 self.test_G_loss_rgb_l2.append(test_errRGBl2.item())
#             self.update_accuracy(isTest=True)
#             printer.print_test_epoch_losses_summary(self.num_epochs, epoch, test_loss_D, test_errGd, self.accDreal_test,
#                                                     self.accDfake_test, self.accG_test)
