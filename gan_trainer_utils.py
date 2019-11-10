import torchvision.transforms as transforms
import params
import torchvision.utils as vutils
import torch
import pathlib
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import exposure
import math
import cv2
from PIL import Image
import imageio
import torch
import tranforms as transforms_
# import hdr_image_utils
# import Writer



def custom_loss(output, target):
    b_size = target.shape[0]
    loss = ((output - target) ** 2).sum() / b_size
    return loss

def get_loss_path(result_dir_pref, model_name, loss_graph_path):
    output_dir = os.path.join(result_dir_pref + "_" + model_name)
    loss_graph_path = os.path.join(output_dir, loss_graph_path)


def create_dir(result_dir_pref, model_name, model_path, loss_graph_path, result_path, model_depth):
    # output_dir = os.path.join("/cs","labs","raananf","yael_vinker","29_07",result_dir_pref + "_" + model_name)
    output_dir = os.path.join(result_dir_pref + "_" + model_name + "_depth_" + str(model_depth))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Directory ", output_dir, " created")

    model_path = os.path.join(output_dir, model_path)
    models_250_save_path = os.path.join("models_250", "models_250_net.pth")
    model_path_250 = os.path.join(output_dir, models_250_save_path)
    loss_graph_path = os.path.join(output_dir, loss_graph_path)
    result_path = os.path.join(output_dir, result_path)
    acc_path = os.path.join(output_dir, "accuracy")
    tmqi_path = os.path.join(output_dir, "tmqi")
    gradient_flow_path = os.path.join(output_dir, params.gradient_flow_path, "g")

    if not os.path.exists(os.path.dirname(gradient_flow_path)):
        os.makedirs(os.path.dirname(gradient_flow_path))
        print("Directory ", gradient_flow_path, " created")

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print("Directory ", model_path, " created")

    if not os.path.exists(os.path.dirname(model_path_250)):
        os.makedirs(os.path.dirname(model_path_250))
        print("Directory ", model_path_250, " created")

    if not os.path.exists(loss_graph_path):
        os.mkdir(loss_graph_path)

        print("Directory ", loss_graph_path, " created")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print("Directory ", result_path, " created")
    if not os.path.exists(acc_path):
        os.mkdir(acc_path)
        print("Directory ", acc_path, " created")

    if not os.path.exists(tmqi_path):
        os.mkdir(tmqi_path)
        print("Directory ", tmqi_path, " created")

    return output_dir



def get_rgb_normalize_im(im):
    new_im = im.clone()
    norm_rgb = torch.norm(im, dim=0) + params.epsilon
    indices = (norm_rgb == 0).nonzero()
    if indices.shape[0] != 0:
        print("TORCH.SUM CONTAINS ZEROS")
    for i in range(im.shape[0]):
        new_im[i, :, :] = im[i, :, :] / norm_rgb
    return new_im

def get_rgb_normalize_im_batch(batch):
    batch_normalize = torch.stack([get_rgb_normalize_im(im_x) for im_x in batch])
    return batch_normalize

if __name__ == '__main__':

    data = np.load("/Users/yaelvinker/PycharmProjects/lab/data/hdr_log_data/hdr_log_data/belgium_10002.npy", allow_pickle=True)
    # # if testMode:
    # input_im = data[()]["input_image"]
    # print(input_im.shape)
    # color_im = data[()]["display_image"]
    # im1 = imageio.imread("data/hdr_data/hdr_data/S0010.hdr").astype('float32')
    # im1 = (im1 / np.max(im1))
    # imageio.imwrite(os.path.join("","im1.png"), im1, format='PNG-FI')
    #
    # im_gray = np.dot(im1[..., :3], [0.299, 0.587, 0.114])
    # transform_custom_ = transforms.Compose([
    #     transforms_.Scale(params.input_size),
    #     transforms_.CenterCrop(params.input_size),
    #     transforms_.ToTensor(),
    # ])
    # transform_custom_gray = transforms.Compose([
    #     transforms_.Scale(params.input_size),
    #     transforms_.CenterCrop(params.input_size),
    #     transforms_.ToTensor(),
    #     transforms_.Normalize(0.5, 0.5),
    # ])
    # im_transform = transform_custom_(im1)
    # gray_t = im_transform.sum(axis=0)
    # gray_fake = transform_custom_gray(im_gray)
    # gray_fake_0_1 = gray_fake + 1
    # # gray_np = gray_fake_0_1.clone().permute(1, 2, 0).detach().cpu().numpy()[:,:,0]
    # # gray_np_gamma = torch.pow(gray_fake_0_1[0, :, :], 0.5)
    # # plt.imshow(gray_np_gamma, cmap='gray')
    # # plt.show()
    # # gray_fake[0, :, :] = gray_np_gamma
    #
    # im5 = back_to_color(im_transform.clone().permute(1, 2, 0).detach().cpu().numpy(), gray_t, gray_fake_0_1.clone().permute(1, 2, 0).detach().cpu().numpy())
    # plt.imshow(im5)
    # plt.show()
    #
    # im2 = im_transform.clone()
    # print(im2.shape)
    # print(gray_t.shape)
    # # im2[0, :, :].div(gray_t)
    # im2[0, :, :] = im_transform[0, :, :] / gray_t
    # im2[1, :, :] = im_transform[1, :, :] / gray_t
    # im2[2, :, :] = im_transform[2, :, :] / gray_t
    # print(im2[0,0,0])
    # print(im_transform[0,0,0] / (gray_t[0,0]))
    # im3 = torch.pow(im2, 0.4) * gray_fake_0_1
    #
    # im_display = im3.clone().permute(1, 2, 0).detach().cpu().numpy()
    # im_display = to_0_1_range(im_display)
    # im2_display = (np.exp(im_display)) / (np.exp(np.max(im_display)))
    # # im3_d = im2_display / np.max(im2_display)
    # im3_d = im2_display
    # plt.imshow(im_display)
    #
    # # plt.imsave("im4.jpg", im_display)
    # plt.show()
    # plt.imshow(im3_d)
    # # plt.imsave("im4.jpg", im_display)
    # plt.show()
    #
    # # hdr_image_utils.print_image_details(im3, "pow")
    # # plt.imshow(im3)
    # # plt.show()


#
# def normalize_im_by_windows(im, window_size):
#     channeles_by_wind_im = get_windows_to_channels_im(im, window_size)
#     mean_matrix = np.mean(channeles_by_wind_im, axis=2)
#     mean_matrix_broadcasted = mean_matrix.reshape((mean_matrix.shape[0], mean_matrix.shape[1], 1))
#     std_matrix = np.sqrt(np.mean(np.power((channeles_by_wind_im - mean_matrix_broadcasted), 2), axis=2))
#     std_matrix_broadcasted = std_matrix.reshape((std_matrix.shape[0], std_matrix.shape[1], 1))
#
#     normalized_im_by_wind = (channeles_by_wind_im - mean_matrix_broadcasted) / std_matrix_broadcasted
#     return normalized_im_by_wind


# def get_windows_to_channels_im(im, window_size):
#     patches = image.extract_patches_2d(im, (window_size, window_size))
#     flat_patches = patches.flatten()
#     new_height, new_width = im.shape[0] - 2 * int(window_size / 2), im.shape[1] - 2 * int(window_size / 2)
#     return flat_patches.reshape((new_height, new_width, int(window_size ** 2) * im.shape[-1]))

#
# def get_tensor_normalized_images_for_windows_loss(im1_tensor, im2_tensor, window_size):
#     im1_tensor = im1_tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
#     normalized_im_by_wind = normalize_im_by_windows(im1_tensor, window_size)
#     print(normalized_im_by_wind.shape)
#     normalized_im_by_wind_reshaped = normalized_im_by_wind.reshape((normalized_im_by_wind.shape[0] * normalized_im_by_wind.shape[1], normalized_im_by_wind.shape[2]))
#
#     im2_tensor = im2_tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
#     normalized_im_by_wind_2 = normalize_im_by_windows(im2_tensor, window_size)
#     normalized_im_by_wind_reshaped_2 = normalized_im_by_wind_2.reshape((normalized_im_by_wind_2.shape[0] * normalized_im_by_wind_2.shape[1], normalized_im_by_wind_2.shape[2]))
#
#     return torch.from_numpy(normalized_im_by_wind_reshaped), torch.from_numpy(normalized_im_by_wind_reshaped_2)

# def windows_l2_normalized_loss(fake_im_batch, hdr_im_batch):
#     b_size = hdr_im_batch.shape[0]
#     loss = 0
#     for i in range(b_size):
#         fake_im, hdr_im = fake_im_batch[i], hdr_im_batch[i]
#         fake_im_normalize, hdr_im_normalize = get_tensor_normalized_images_for_windows_loss(fake_im, hdr_im, window_size=5)
#         loss += mse_loss(fake_im_normalize, hdr_im_normalize)
#     return loss / b_size


