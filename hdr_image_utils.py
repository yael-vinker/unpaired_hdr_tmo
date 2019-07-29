import os
import torch

import cv2
import pathlib
import skimage

import imageio
import numpy as np
import matplotlib


WINDOW_SIZE_FACTOR = 20
CHANGE_MATRIX = (np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523, 0.311]])).T  # Create the change of basis matrix

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def save_results(res_im, path_name):
    matplotlib.image.imsave(path_name, res_im)


def log_transform(im):
    log_im = ((np.log(im + 1)) / (np.log(1 + np.max(im))))
    return log_im


def draw_patch_from_center(color, window_height, window_width, ax, im, x, y, text):
    import matplotlib.patches as patches
    half_height = int(window_height / 2)
    half_width = int(window_width / 2)
    ax.imshow(im, cmap='gray')
    rect = patches.Rectangle((x - half_width, y - half_height), window_width, window_height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def print_image_details(im, title):
    print(title)
    print("shape : ", im.shape)
    print("max : ", np.max(im), "  min : ", np.min(im), "mean : ", np.mean(im))
    print("type : ", im.dtype)


def RGB2YUV(rgb_im):

    m = np.array([[0.29900, -0.16874,  0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    y = np.dot(rgb_im, m)[:, :, 0]
    return y


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space.
    :param imRGB: image as a matrix with values in [0, 1].
    :return: transformed image.
    """
    yiq_matrix = np.dot(imRGB, CHANGE_MATRIX)
    return yiq_matrix


def read_hdr_img_rgb(path_name):
    """

    :param path: .hdr extension of the wanted image path
    :return: RGB image with hdr values
    """
    path = pathlib.Path(path_name)
    im = imageio.imread(path, format='HDR-FI')
    im = skimage.exposure.rescale_intensity(im, out_range=(0, 1000))
    # im = im / 1000
    return im


def resize_image(im):
    if im.shape[1] > im.shape[0]:
        return cv2.resize(im, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT))


def read_img(path):
    """

    :param path:
    :return: RGB image with hdr values, scale dto the defined values
    """
    im_rgb = read_hdr_img_rgb(path)
    return resize_image(im_rgb)


def get_file_name(path):
    pre = os.path.splitext(path)[0]
    return str(pre)


def get_window_size(im_height, im_width):
    win_height = int(im_height / WINDOW_SIZE_FACTOR)
    win_width = int(im_width / WINDOW_SIZE_FACTOR)
    if win_height % 2 == 0:
        win_height += 1
    if win_width % 2 == 0:
        win_width += 1
    return win_height, win_width


def get_window_borders_from_center(max_width, max_height, x_center, half_width, y_center, half_height):
    y0, y1, x0, x1 = y_center - half_height, y_center + half_height + 1, x_center - half_width, x_center + half_width + 1
    if y0 < 0:
        y0 = 0
    if y1 > max_height:
        y1 = max_height
    if x0 < 0:
        x0 = 0
    if x1 > max_width:
        x1 = max_width
    return y0, y1, x0, x1


def get_window(im, boarders):
    y0, y1, x0, x1 = boarders[0], boarders[1], boarders[2], boarders[3]
    return im[y0: y1, x0: x1]


def get_half_windw_size(window_height, window_width):
    half_window_height, half_window_width = int(window_height / 2), int(window_width / 2)
    half_window_height = half_window_height + 1 if half_window_height % 2 == 0 else half_window_height
    half_window_width = half_window_width + 1 if half_window_width % 2 == 0 else half_window_width
    quarter_height, quarter_width = int(half_window_height / 2), int(half_window_width / 2)
    return half_window_height, half_window_width, quarter_height, quarter_width