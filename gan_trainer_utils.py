import pathlib
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import exposure
import math
import cv2
from PIL import Image
import imageio
import torch
import hdr_image_utils


def custom_loss(output, target):
    b_size = target.shape[0]
    loss = ((output - target) ** 2).sum() / b_size
    return loss


def plot_general_losses(loss_G, loss_G_wind, loss_D_fake, loss_D_real, title, iters_n, path, use_g_ssim_loss):
    if use_g_ssim_loss:
        plt.figure()
        plt.plot(range(iters_n), loss_D_fake, '-r', label='loss D fake')
        plt.plot(range(iters_n), loss_D_real, '-b', label='loss D real')
        plt.plot(range(iters_n), loss_G, '-g', label='loss G')
        plt.plot(range(iters_n), loss_G_wind, '-y', label='loss G SSIM')

        plt.xlabel("n iteration")
        plt.legend(loc='upper left')
        plt.title(title)

        # save image
        plt.savefig(os.path.join(path, title + "all.png"))  # should before show method
        plt.close()

    plt.figure()
    plt.plot(range(iters_n), loss_D_fake, '-r', label='loss D fake')
    plt.plot(range(iters_n), loss_D_real, '-b', label='loss D real')
    plt.plot(range(iters_n), loss_G, '-g', label='loss G')

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

def create_dir(result_dir_pref, model_name, model_path, loss_graph_path, result_path):
    # output_dir = os.path.join("/cs","labs","raananf","yael_vinker","29_07",result_dir_pref + "_" + model_name)
    output_dir = os.path.join(result_dir_pref + "_" + model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Directory ", output_dir, " created")

    model_path = os.path.join(output_dir, model_path)
    loss_graph_path = os.path.join(output_dir, loss_graph_path)
    result_path = os.path.join(output_dir, result_path)
    acc_path = os.path.join(output_dir, "accuracy")

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print("Directory ", model_path, " created")
    if not os.path.exists(loss_graph_path):
        os.mkdir(loss_graph_path)
        print("Directory ", loss_graph_path, " created")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print("Directory ", result_path, " created")
    if not os.path.exists(acc_path):
        os.mkdir(acc_path)
        print("Directory ", acc_path, " created")
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
        print(norm_im[:,:,0].shape)
        plt.imshow(norm_im[:,:,0], cmap='gray')
        plt.show()
    else:
        norm_im = (((np.exp(im) - 1) / IMAGE_SCALE) * IMAGE_MAX_VALUE).astype("uint8")
    # norm_im_clamp = np.clip(norm_im, 0, 255)
    return norm_im

def uint_normalization(im):
    norm_im = ((im / np.max(im)) * 255).astype("uint8")
    # norm_im_clamp = np.clip(norm_im, 0, 255)
    # norm_im_clamp = norm_im
    return norm_im

def to_0_1_range(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def display_batch_as_grid(batch, ncols_to_display, normalization="log100", nrow=8, pad_value=0, isHDR=False, batch_start_index=0, toPrint=False):
    batch = batch[batch_start_index:ncols_to_display]
    b_size = batch.shape[0]
    output = []
    for i in range(b_size):
        cur_im = batch[i].clone().permute(1, 2, 0).detach().cpu().numpy()
        if normalization == "uint":
            norm_im = uint_normalization(cur_im)
        elif normalization == "log100":
            norm_im = log100_normalization(cur_im, isHDR)
        elif normalization == "uint_0_1":
            im = to_0_1_range(cur_im)
            norm_im = uint_normalization(im)
        elif normalization == "log1_uint_0_1":
            im = np.exp(cur_im) - 1
            im = to_0_1_range(im)
            norm_im = uint_normalization(im)
        else:
            raise Exception('ERROR: Not valid normalization for display')
        norm_im = np.clip(norm_im, 0, 255)
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
    grid = np.full((height * ymaps, width * xmaps, norm_batch.shape[3]), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[(y * height):(y * height + height), x * width : x * width + width] = norm_batch[k]
            k = k + 1
    return grid

def get_single_ldr_im(ldr_data_root, images_number=1):
    images = []
    x = next(os.walk(ldr_data_root))[1][0]
    dir_path = os.path.join(ldr_data_root, x)
    for i in range(images_number):
        im_name = os.listdir(dir_path)[i]
        im_path = os.path.join(dir_path, im_name)
        with open(im_path, 'rb') as f:
            img = Image.open(f)
            images.append((im_name, np.asarray(img.convert('RGB'))))
    return images

def get_single_hdr_im(hdr_data_root, images_number=1, isNpy=False):
    images = []
    # if isNpy:
    #     x = next(os.walk(hdr_data_root))[1][0]
    #     dir_path = os.path.join(hdr_data_root, x)
    #     for i in range(images_number):
    #         im_name = os.listdir(dir_path)[i]
    #         im_path = os.path.join(dir_path, im_name)
    #         data = np.load(im_path)
    #         im_hdr = data[()][params.image_key]
    #         images.append((im_name, im_hdr))
    #     return images
    x = next(os.walk(hdr_data_root))[1][0]
    dir_path = os.path.join(hdr_data_root, x)
    for i in range(images_number):
        im_name = os.listdir(dir_path)[i]
        im_path = os.path.join(dir_path, im_name)
        # im_origin = imageio.imread(im_path, format='HDR-FI')
        im_origin = imageio.imread(im_path)
        # im_origin = cv2.imread(im_path)
        # print(im_origin.shape)
        images.append((im_name, im_origin))
    return images

def load_data_test_mode(test_hdr_dataloader, test_ldr_dataloader, images_number=1):
    # train_hdr_loader = next(iter(train_hdr_dataloader))[params.image_key]
    # train_hdr_loader = next(iter(train_hdr_dataloader))[0]
    # train_hdr_loader_single = np.asarray(train_hdr_loader[0])
    # print("train_hdr_dataloader --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]" %
    #       (float(np.max(train_hdr_loader_single)), float(np.min(train_hdr_loader_single)),
    #        train_hdr_loader_single.dtype, str(train_hdr_loader_single.shape)))
    # im_display = (((np.exp(train_hdr_loader_single) - 1) / 100) * 255).astype("uint8")
    # plt.imshow(np.transpose(im_display, (1, 2, 0)))
    # plt.show()
    #
    # train_ldr_loader = next(iter(train_ldr_dataloader))[0]
    # train_ldr_loader_single = np.asarray(train_ldr_loader[0])
    # print("train_ldr_dataloader --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]" %
    #       (float(np.max(train_ldr_loader_single)), float(np.min(train_ldr_loader_single)),
    #        train_ldr_loader_single.dtype, str(train_ldr_loader_single.shape)))
    # im_display = (((np.exp(train_ldr_loader_single) - 1) / 100) * 255).astype("uint8")
    # plt.imshow(np.transpose(im_display, (1, 2, 0)))
    # plt.show()

    # test_hdr_loader = next(iter(test_hdr_dataloader))[params.image_key]
    test_hdr_loader = next(iter(test_hdr_dataloader))[0]
    test_ldr_loader = next(iter(test_ldr_dataloader))[0]
    for i in range(images_number):
        test_hdr_loader_single = np.asarray(test_hdr_loader[i])
        print("test_hdr_dataloader --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]" %
              (float(np.max(test_hdr_loader_single)), float(np.min(test_hdr_loader_single)),
               test_hdr_loader_single.dtype, str(test_hdr_loader_single.shape)))
        # im_display = (((np.exp(test_hdr_loader_single) - 1) / 100) * 255).astype("uint8")
    # plt.imshow(np.transpose(im_display, (1, 2, 0)))
    # plt.show()


        test_ldr_loader_single = np.asarray(test_ldr_loader[i])
        print("test_ldr_dataloader --- max[%.4f]  min[%.4f]  dtype[%s]  shape[%s]" %
              (float(np.max(test_ldr_loader_single)), float(np.min(test_ldr_loader_single)),
               test_ldr_loader_single.dtype, str(test_ldr_loader_single.shape)))
    # im_display = (((np.exp(test_ldr_loader_single) - 1) / 100) * 255).astype("uint8")
    # plt.imshow(np.transpose(im_display, (1, 2, 0)))
    # plt.show()

def print_dataset_details(images_number, data_loader, batch):
    print("\ntrain_npy_dataset [%d] images" % (len(data_loader.dataset)))
    for i in range(images_number):
        sample = batch[i]
        im_name, im = sample[0], sample[1]
        print(im_name + "    max[%.4f]  min[%.4f]  dtype[%s]" % (float(np.max(im)), float(np.min(im)), im.dtype))




def normalize_im_by_windows(im, window_size):
    channeles_by_wind_im = get_windows_to_channels_im(im, window_size)
    mean_matrix = np.mean(channeles_by_wind_im, axis=2)
    mean_matrix_broadcasted = mean_matrix.reshape((mean_matrix.shape[0], mean_matrix.shape[1], 1))
    std_matrix = np.sqrt(np.mean(np.power((channeles_by_wind_im - mean_matrix_broadcasted), 2), axis=2))
    std_matrix_broadcasted = std_matrix.reshape((std_matrix.shape[0], std_matrix.shape[1], 1))

    normalized_im_by_wind = (channeles_by_wind_im - mean_matrix_broadcasted) / std_matrix_broadcasted
    return normalized_im_by_wind


def get_windows_to_channels_im(im, window_size):
    patches = image.extract_patches_2d(im, (window_size, window_size))
    flat_patches = patches.flatten()
    new_height, new_width = im.shape[0] - 2 * int(window_size / 2), im.shape[1] - 2 * int(window_size / 2)
    return flat_patches.reshape((new_height, new_width, int(window_size ** 2) * im.shape[-1]))


def get_tensor_normalized_images_for_windows_loss(im1_tensor, im2_tensor, window_size):
    im1_tensor = im1_tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    normalized_im_by_wind = normalize_im_by_windows(im1_tensor, window_size)
    print(normalized_im_by_wind.shape)
    normalized_im_by_wind_reshaped = normalized_im_by_wind.reshape((normalized_im_by_wind.shape[0] * normalized_im_by_wind.shape[1], normalized_im_by_wind.shape[2]))

    im2_tensor = im2_tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    normalized_im_by_wind_2 = normalize_im_by_windows(im2_tensor, window_size)
    normalized_im_by_wind_reshaped_2 = normalized_im_by_wind_2.reshape((normalized_im_by_wind_2.shape[0] * normalized_im_by_wind_2.shape[1], normalized_im_by_wind_2.shape[2]))

    return torch.from_numpy(normalized_im_by_wind_reshaped), torch.from_numpy(normalized_im_by_wind_reshaped_2)

# def windows_l2_normalized_loss(fake_im_batch, hdr_im_batch):
#     b_size = hdr_im_batch.shape[0]
#     loss = 0
#     for i in range(b_size):
#         fake_im, hdr_im = fake_im_batch[i], hdr_im_batch[i]
#         fake_im_normalize, hdr_im_normalize = get_tensor_normalized_images_for_windows_loss(fake_im, hdr_im, window_size=5)
#         loss += mse_loss(fake_im_normalize, hdr_im_normalize)
#     return loss / b_size

if __name__ == '__main__':
    path = "hdr05.hdr"
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    im_origin = im_origin / np.max(im_origin)
    hdr_image_utils.print_image_details(im_origin, "hdr")

    new_im = (im_origin - np.mean(im_origin)) / np.std(im_origin)
    hdr_image_utils.print_image_details(new_im, "new_im")