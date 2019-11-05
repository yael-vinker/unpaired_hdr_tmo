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

def read_ldr_image(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    im = im_origin / 255
    return im

def read_hdr_image(path):
    path_lib_path = pathlib.Path(path)
    file_extension = os.path.splitext(path)[1]
    if file_extension == ".hdr":
        im = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
    elif file_extension == ".dng":
        im = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
    else:
        raise Exception('invalid hdr file format: {}'.format(file_extension))
    return im

def hdr_log_loader_factorize(path, range_factor):
    im_origin = read_hdr_image(path)
    max_origin = np.max(im_origin)
    image_new_range = (im_origin / max_origin) * range_factor
    im_log = np.log(image_new_range + 1)
    im = (im_log / np.log(range_factor + 1)).astype('float32')
    return im

def custom_loss(output, target):
    b_size = target.shape[0]
    loss = ((output - target) ** 2).sum() / b_size
    return loss

def plot_general_losses(G_loss_d, G_loss_ssim, loss_D_fake, loss_D_real, title, iters_n, path, use_g_d_loss, use_g_ssim_loss):
    if use_g_ssim_loss or use_g_d_loss:
        plt.figure()
        plt.plot(range(iters_n), loss_D_fake, '-r', label='loss D fake')
        plt.plot(range(iters_n), loss_D_real, '-b', label='loss D real')
        if use_g_d_loss:
            plt.plot(range(iters_n), G_loss_d, '-g', label='loss G')
        if use_g_ssim_loss:
            plt.plot(range(iters_n), G_loss_ssim, '-y', label='loss G SSIM')
        plt.xlabel("n iteration")
        plt.legend(loc='upper left')
        plt.title(title)

        # save image
        plt.savefig(os.path.join(path, title + "all.png"))  # should before show method
        plt.close()

    plt.figure()
    plt.plot(range(iters_n), loss_D_fake, '-r', label='loss D fake')
    plt.plot(range(iters_n), loss_D_real, '-b', label='loss D real')
    if use_g_d_loss:
        plt.plot(range(iters_n), G_loss_d, '-g', label='loss G')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(os.path.join(path, title + ".png"))  # should before show method
    plt.close()


def plot_general_accuracy(acc_G, acc_D_fake, acc_D_real, title, iters_n, path):
    plt.figure()
    plt.plot(range(iters_n), acc_D_fake, '-r', label='acc D fake')
    plt.plot(range(iters_n), acc_D_real, '-b', label='acc D real')
    # plt.plot(range(iters_n), acc_G, '-g', label='acc G')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(os.path.join(path, title + ".png"))  # should before show method
    plt.close()


def get_loss_path(result_dir_pref, model_name, loss_graph_path):
    output_dir = os.path.join(result_dir_pref + "_" + model_name)
    loss_graph_path = os.path.join(output_dir, loss_graph_path)


def create_dir(result_dir_pref, model_name, model_path, loss_graph_path, result_path):
    # output_dir = os.path.join("/cs","labs","raananf","yael_vinker","29_07",result_dir_pref + "_" + model_name)
    output_dir = os.path.join(result_dir_pref + "_" + model_name)
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


def log100_normalization(im, isHDR):
    IMAGE_SCALE = 100
    IMAGE_MAX_VALUE = 255
    if isHDR:
        norm_im = (np.exp(im) - 1) / IMAGE_SCALE
        if norm_im.shape[2] == 1:
            gamma_corrected = exposure.adjust_gamma(norm_im, 0.5)
            im1 = (gamma_corrected * IMAGE_MAX_VALUE).astype("uint8")
            # norm_im = im1[:, :, 0]
            norm_im = im1
        else:
            tone_map1 = cv2.createTonemapReinhard(1.5, 0, 0, 0)
            im1_dub = tone_map1.process(norm_im.copy()[:, :, ::-1])
            im1 = (im1_dub * IMAGE_MAX_VALUE).astype("uint8")
            norm_im = im1

    else:
        norm_im = (((np.exp(im) - 1) / IMAGE_SCALE) * IMAGE_MAX_VALUE).astype("uint8")
    # norm_im_clamp = np.clip(norm_im, 0, 255)
    return norm_im


def uint_normalization(im):
    # norm_im = ((im / np.max(im)) * 255).astype("uint8")
    norm_im = (im * 255).astype("uint8")
    # norm_im_clamp = np.clip(norm_im, 0, 255)
    # norm_im_clamp = norm_im
    norm_im = np.clip(norm_im, 0, 255)
    return norm_im


def to_0_1_range(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def to_0_1_range_tensor(im):
    return (im - im.min()) / (im.max() - im.min())

def back_to_color(im_hdr, fake):
    im_gray_ = np.sum(im_hdr, axis=2)
    fake = to_0_1_range(fake)
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / im_gray_
    norm_im[:, :, 1] = im_hdr[:, :, 1] / im_gray_
    norm_im[:, :, 2] = im_hdr[:, :, 2] / im_gray_
    output_im = np.power(norm_im, 0.5) * fake
    return output_im

def back_to_color_exp(im_hdr, fake):
    im_gray_ = np.sum(im_hdr, axis=2)
    fake = to_0_1_range(fake)
    fake = exp_normalization(fake)
    norm_im = np.zeros(im_hdr.shape)
    norm_im[:, :, 0] = im_hdr[:, :, 0] / im_gray_
    norm_im[:, :, 1] = im_hdr[:, :, 1] / im_gray_
    norm_im[:, :, 2] = im_hdr[:, :, 2] / im_gray_
    output_im = np.power(norm_im, 0.5) * fake
    return output_im

def back_to_color_batch(im_hdr_batch, fake_batch):
    b_size = im_hdr_batch.shape[0]
    output = []
    for i in range(b_size):
        im_hdr = im_hdr_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        fake = fake_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        norm_im = back_to_color(im_hdr, fake)
        output.append(torch.from_numpy(norm_im.transpose((2, 0, 1))).float())
    return torch.stack(output)

def back_to_color_exp_batch(im_hdr_batch, fake_batch):
    b_size = im_hdr_batch.shape[0]
    output = []
    for i in range(b_size):
        im_hdr = im_hdr_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        fake = fake_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        norm_im = back_to_color_exp(im_hdr, fake)
        output.append(torch.from_numpy(norm_im.transpose((2, 0, 1))).float())
    return torch.stack(output)

def exp_normalization(im):
    return (np.exp(im) - 1) / (np.exp(np.max(im)) - 1)


def display_batch_as_grid(batch, ncols_to_display, normalization, nrow=8, pad_value=0.0, isHDR=False,
                          batch_start_index=0, toPrint=False):
    batch = batch[batch_start_index:ncols_to_display]
    b_size = batch.shape[0]
    output = []
    for i in range(b_size):
        cur_im = batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        if normalization == "uint":
            norm_im = uint_normalization(cur_im)
        elif normalization == "0_1":
            norm_im = to_0_1_range(cur_im)
        elif normalization == "log100":
            norm_im = log100_normalization(cur_im, isHDR)
        elif normalization == "uint_0_1":
            im = to_0_1_range(cur_im)
            norm_im = uint_normalization(im)
        elif normalization == "log1_uint_0_1":
            im = np.exp(cur_im) - 1
            im = to_0_1_range(im)
            norm_im = uint_normalization(im)
        elif normalization == "exp":
            im = to_0_1_range(cur_im)
            norm_im = exp_normalization(im)
        elif normalization == "none":
            norm_im = cur_im
        else:
            raise Exception('ERROR: Not valid normalization for display')
        if i == 0 and toPrint:
            print("fake display --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]" %
                  (float(np.max(norm_im)), float(np.min(norm_im)),
                   norm_im.dtype, str(norm_im.shape)))
        output.append(norm_im)
    norm_batch = np.asarray(output)
    nmaps = norm_batch.shape[0]
    xmaps = min(ncols_to_display, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(norm_batch.shape[1]), int(norm_batch.shape[2])
    if norm_im.shape[2] == 1:
        grid = np.full((height * ymaps, width * xmaps), pad_value)
    else:
        grid = np.full((height * ymaps, width * xmaps, norm_batch.shape[3]), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            if norm_im.shape[2] == 1:
                im_for_grid = norm_batch[k][:, :, 0]
            else:
                im_for_grid = norm_batch[k]
            grid[(y * height):(y * height + height), x * width : x * width + width] = im_for_grid
            k = k + 1
    return grid

def save_groups_images(test_hdr_batch, test_real_batch, fake, fake_ldr, new_out_dir, batch_size, epoch, image_mean):
    test_ldr_batch = test_real_batch["input_im"]
    color_test_ldr_batch = test_real_batch["color_im"]
    test_hdr_image = test_hdr_batch["input_im"]
    color_test_hdr_image = test_hdr_batch["color_im"]

    normalization_string = "0_1"
    if image_mean == 0.5:
        normalization_string = "none"

    output_len = int(batch_size / 4)
    display_group = [test_ldr_batch, fake_ldr, test_hdr_image, fake]
    # titles = ["Input Images", "Fake Images", "Exp Images"]
    titles = ["Real (LDR) Images", "G(LDR)", "Input (HDR) Images", "Fake Images"]
    normalization_string_arr = ["0_1", "0_1", "0_1", "0_1"]
    for i in range(output_len):
        plt.figure(figsize=(15, 15))
        for j in range(4):
            display_im = display_batch_as_grid(display_group[j], ncols_to_display=(i + 1) * 4, normalization=normalization_string_arr[j],
                                                 isHDR=False, batch_start_index=i * 4)
            plt.subplot(4, 1, j + 1)
            plt.axis("off")
            plt.title(titles[j])
            if display_im.ndim == 2:
                plt.imshow(display_im, cmap='gray')
            else:
                plt.imshow(display_im)
        plt.savefig(os.path.join(new_out_dir, "set " + str(i)))
        plt.close()

    color_fake = back_to_color_batch(color_test_hdr_image, fake)
    color_display_group = [color_test_ldr_batch, color_test_hdr_image, color_fake]
    # color_fake = back_to_color_batch(color_test_hdr_image, fake)
    # color_fake_exp = back_to_color_exp_batch(color_test_hdr_image, fake)
    # color_display_group = [color_test_ldr_batch, color_test_hdr_image, color_fake]
    # titles = ["Input Images", "Fake Images", "Exp Images"]
    titles = ["Real (LDR) Images", "Input (HDR) Images", "Fake Images"]
    # normalization_string_arr = ["0_1", "0_1", "exp"]
    for i in range(output_len):
        plt.figure(figsize=(15, 15))
        for j in range(3):
            if j == 2:
                normalization_string = "none"
            display_im = display_batch_as_grid(color_display_group[j], ncols_to_display=(i + 1) * 4, normalization=normalization_string,
                                                 isHDR=False, batch_start_index=i * 4)
            plt.subplot(3, 1, j + 1)
            plt.axis("off")
            plt.title(titles[j])
            plt.imshow(display_im)
        plt.savefig(os.path.join(new_out_dir, "color set " + str(i)))
        plt.close()

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
#
# def get_no(fake, hdr_input):
#     new_fake = get_rgb_normalize_im(fake)
#     new_hdr_unput = get_rgb_normalize_im(hdr_input)
#

def back_to_color_tensor(fake, im_hdr_display):
    """

    :param fake: range [-1, -] gray
    :param im_hdr: range [-1, 1] gray
    :param im_hdr_display: range [0,1]
    :return:
    """
    gray_im = im_hdr_display.sum(dim=0)
    rgb_hdr_copy = im_hdr_display.clone()
    rgb_hdr_copy[0, :, :] = rgb_hdr_copy[0, :, :] / gray_im
    rgb_hdr_copy[1, :, :] = rgb_hdr_copy[1, :, :] / gray_im
    rgb_hdr_copy[2, :, :] = rgb_hdr_copy[2, :, :] / gray_im
    gray_fake_to_0_1 = to_0_1_range_tensor(fake)
    output_im = torch.pow(rgb_hdr_copy, 0.5) * gray_fake_to_0_1
    # display_tensor(output_im)
    return output_im


def back_to_color_batch_tensor(fake_batch, hdr_input_display_batch):
    b_size = fake_batch.shape[0]
    output = [back_to_color_tensor(fake_batch[i], hdr_input_display_batch[i]) for i in range(b_size)]
    return torch.stack(output)


def display_tensor(tensor):
    im_display = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    im_display = to_0_1_range(im_display)
    plt.imshow(im_display)
    plt.show()

# ----- data loader -------
def load_data_set(data_root, batch_size_, shuffle, testMode):
    import ProcessedDatasetFolder
    npy_dataset = ProcessedDatasetFolder.ProcessedDatasetFolder(root=data_root, testMode=testMode)
    dataloader = torch.utils.data.DataLoader(npy_dataset, batch_size=batch_size_,
                                             shuffle=shuffle, num_workers=params.workers)
    return dataloader


def load_data(train_root_npy, train_root_ldr, batch_size, testMode, title):
    import printer
    """
    :param isHdr: True if images in "dir_root" are in .hdr format, False otherwise.
    :param dir_root: path to wanted directory
    :param b_size: batch size
    :return: DataLoader object of images in "dir_root"
    """
    train_hdr_dataloader = load_data_set(train_root_npy, batch_size, shuffle=True, testMode=testMode)
    train_ldr_dataloader = load_data_set(train_root_ldr, batch_size, shuffle=True, testMode=testMode)

    printer.print_dataset_details([train_hdr_dataloader, train_ldr_dataloader],
                                  [train_root_npy, train_root_ldr],
                                  [title + "_hdr_dataloader", title + "_ldr_dataloader"],
                                  [True, False],
                                  [True, True])

    printer.load_data_dict_mode(train_hdr_dataloader, train_ldr_dataloader, title, images_number=2)
    return train_hdr_dataloader, train_ldr_dataloader



def to_gray(im):
    return np.dot(im[...,:3], [0.299, 0.587, 0.114]).astype('float32')




if __name__ == '__main__':
    data = np.load("/Users/yaelvinker/PycharmProjects/lab/data/hdr_log_data/hdr_log_data/belgium_1000.npy", allow_pickle=True)
    # if testMode:
    input_im = data[()]["input_image"]
    print(input_im.shape)
    color_im = data[()]["display_image"]
    im1 = imageio.imread("data/hdr_data/hdr_data/S0010.hdr").astype('float32')
    im1 = (im1 / np.max(im1))
    imageio.imwrite(os.path.join("","im1.png"), im1, format='PNG-FI')
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


