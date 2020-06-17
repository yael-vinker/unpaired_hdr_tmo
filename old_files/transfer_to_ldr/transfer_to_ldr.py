import os
import pathlib

import cv2
import imageio
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage

WINDOW_SIZE_FACTOR = 20
CHANGE_MATRIX = (np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])).T  # Create the change of basis matrix

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def save_results(res_im, path_name):
    print(path_name)
    matplotlib.image.imsave(path_name, res_im)


def draw_results(im1, isHdr1, title1, im2, isHdr2, title2, title, is_gray, to_save, path=""):
    """

    :param im1: first image to draw
    :param isHdr1: boolean, that determine whether im1 is hdr (should be tone-mapped before display).
    :param title1: title for image1
    :param im2: second image to draw
    :param isHdr2: boolean, that determine whether im2 is hdr (should be tone-mapped before display).
    :param title2: title for image2
    :param title: title for the entire display
    :param is_gray: boolean, True if the images are grayscale, False otherwise.
    :param to_save: True if the image should be saved, False for display.
    :return:
    """
    # Tonemap HDR image
    if isHdr1:
        tone_map1 = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        im1_dub = tone_map1.process(im1.copy().astype(np.float32)[:, :, ::-1])
        im1 = np.clip(im1_dub, 0, 1).astype('float32')[:, :, ::-1]
    if isHdr2:
        tone_map2 = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        im2_dub = tone_map2.process(im2.copy().astype(np.float32)[:, :, ::-1])
        im2 = np.clip(im2_dub, 0, 1).astype('float32')[:, :, ::-1]
    f = plt.figure()
    plt.axis("off")
    plt.title(title)
    ax1 = f.add_subplot(121)
    ax1.title.set_text(title1)
    draw_patch_from_center('r', 13, 13, ax1, im1, int(im1.shape[1] / 2), int(im1.shape[0] / 2), "")
    if is_gray:
        plt.imshow(im1, cmap='gray')
    else:
        plt.imshow(im1)
    ax2 = f.add_subplot(122)
    ax2.title.set_text(title2)
    draw_patch_from_center('r', 13, 13, ax2, im2, int(im2.shape[1] / 2), int(im2.shape[0] / 2), "")

    if is_gray:
        plt.imshow(im2, cmap='gray')
    else:
        plt.imshow(im2)
    if to_save:
        plt.savefig(path)
    else:
        plt.show()


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


def normalize_im(hdr_img):
    # 2 * (x - x.min()) / (x.max() - x.min()) - 1
    # d = 2. * (a - np.min(a)) / np.ptp(a) - 1
    # (x - x.mean()) / x.std()
    return (hdr_img - 0.5) / 0.5


def read_hdr_img_rgb(path_name):
    """

    :param path: .hdr extension of the wanted image path
    :return: RGB image with hdr values
    """

    path_lib_path = pathlib.Path(path_name)
    file_extension = os.path.splitext(path_name)[1]
    if file_extension == ".hdr":
        im = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
    elif file_extension == ".dng":
        im = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
    elif file_extension == ".exr":
        im = imageio.imread(path_lib_path, format="EXR-FI").astype('float32')
    else:
        raise Exception('invalid hdr file format: {}'.format(file_extension))
    return im / np.max(im)
    # im = img_as_float(cv2.imread(path, -1))[:, :, ::-1]  # read the hdr image as float64
    # return im  # cv2 returns gbr


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
    return im_rgb
    # return resize_image(im_rgb)


def log_transform(im):
    log_im = ((np.log(im + 1)) / (np.log(1 + np.max(im))))
    return log_im


def get_derivative_threshold(grad_im):
    sorted_grad = np.sort(np.abs(grad_im), axis=None)
    thresh_index = sorted_grad.shape[0] * 0.999
    return sorted_grad[int(thresh_index)]


def to_binary(gradient_im, threshold):
    mask = np.abs(gradient_im) > threshold
    masked = np.zeros(gradient_im.shape)
    masked[mask] = 1
    return masked


def conv_to_binary(convolve_im, threshold):
    mask = convolve_im > threshold
    masked = np.ones(convolve_im.shape)
    masked[mask] = 0.5
    return masked


def get_window_borders_from_center(max_width, max_height, x_center, win_width, y_center, win_height):
    half_height = int(win_height / 2)
    half_width = int(win_width / 2)
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


def get_window(im, x0, width, y0, height):
    return im[y0: height, x0: width]


def draw_patch_from_center(color, window_height, window_width, ax, im, x, y, text):
    half_height = int(window_height / 2)
    half_width = int(window_width / 2)
    ax.imshow(im, cmap='gray')
    rect = patches.Rectangle((x - half_width, y - half_height), window_width, window_height, linewidth=1,
                             edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def get_gradient(im):
    im = cv2.GaussianBlur(im, (5, 5), 0)
    return cv2.Laplacian(im, cv2.CV_64F)


def filter_relevant_pixels_quarter(im, window_width, window_height):
    half_window_height, half_window_width = int(window_height / 2), int(window_width / 2)
    if half_window_height % 2 == 0:
        half_window_height += 1
    if half_window_width % 2 == 0:
        half_window_width += 1
    quarter_height, quarter_width = int(half_window_height / 2), int(half_window_width / 2)
    new_wind = np.zeros((half_window_height, half_window_width))
    new_wind[quarter_height, quarter_width] = 1
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            cur_wind = np.copy(new_wind)
            y0, y1, x0, x1 = i - quarter_height, i + quarter_height + 1, j - quarter_width, j + quarter_width + 1
            if im[i, j] == 1:
                if i < quarter_height:
                    cur_wind = cur_wind[quarter_height - i:, :]
                    y0 = 0
                if i >= (im.shape[0] - quarter_height):
                    cur_wind = cur_wind[: half_window_height - (y1 - im.shape[0]), :]
                    y1 = im.shape[0] + 1
                if j < quarter_width:
                    cur_wind = cur_wind[:, quarter_width - j:]
                    x0 = 0
                if j >= (im.shape[1] - quarter_width):
                    cur_wind = cur_wind[:, : half_window_width - (x1 - im.shape[1])]
                    x1 = im.shape[1] + 1
                im[y0: y1, x0:x1] = cur_wind
    return im


def filter_relevant_pixels(im, window_width, window_height):
    half_height, half_width = int(window_height / 2), int(window_width / 2)
    new_wind = np.zeros((window_height, window_width))
    new_wind[half_height, half_width] = 1
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            cur_wind = np.copy(new_wind)
            y0, y1, x0, x1 = i - half_height, i + half_height + 1, j - half_width, j + half_width + 1
            if im[i, j] == 1:
                if i < half_height:
                    cur_wind = cur_wind[half_height - i:, :]
                    y0 = 0
                if i >= (im.shape[0] - half_height):
                    cur_wind = cur_wind[: window_height - (y1 - im.shape[0]), :]
                    y1 = im.shape[0] + 1
                if j < half_width:
                    cur_wind = cur_wind[:, half_width - j:]
                    x0 = 0
                if j >= (im.shape[1] - half_width):
                    cur_wind = cur_wind[:, : window_width - (x1 - im.shape[1])]
                    x1 = im.shape[1] + 1
                im[y0: y1, x0:x1] = cur_wind
    return im


def print_image_details(im, title):
    print(title)
    print("shape : ", im.shape)
    print("max : ", np.nanmax(im), "  min : ", np.nanmin(im), "mean : ", np.nanmean(im))
    print("type : ", im.dtype)
    print("is nan = ", np.isnan(im).any())


def get_window_size(im_height, im_width):
    WINDOW_SIZE_FACTOR = min(im_height / 10, im_width / 10)
    win_height = int(im_height / WINDOW_SIZE_FACTOR)
    win_width = int(im_width / WINDOW_SIZE_FACTOR)
    if win_height % 2 == 0:
        win_height += 1
    if win_width % 2 == 0:
        win_width += 1
    print("wind h ", win_height, "wind width ", win_width)
    return win_height, win_width


def get_convolved_im(win_height, win_width, bin_im, draw):
    conv = np.ones((win_height, win_width))
    convolve_im = ndimage.convolve(bin_im, conv, mode='constant', cval=0.0)
    convolve_im = convolve_im / (win_height * win_width)
    if draw:
        draw_results(bin_im, False, "binary", convolve_im, False, "convolve", "convolve", True, False)
    return convolve_im


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space.
    :param imRGB: image as a matrix with values in [0, 1].
    :return: transformed image.
    """
    yiq_matrix = np.dot(imRGB, CHANGE_MATRIX)
    return yiq_matrix


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space.
    :param imRGB: image as a matrix with values in [0, 1].
    :return: transformed image.
    """
    inverse = np.linalg.inv(CHANGE_MATRIX)  # Inverse the change basis matrix
    rgb_matrix = np.dot(imYIQ, inverse)
    return rgb_matrix


def fix_high_luminance(im_log, im_rgb, output_path=""):
    """

    :param im_log: grayscale image after log transform
    :param im_rgb: the original image in RGB
    :param filter_win_size: determine the overlap between the windows extracted after the convolution.
    the default value is quarter (3/4 overlap).
    :return:
    """
    cur_im = np.copy(im_rgb)
    window_height, window_width = get_window_size(im_log.shape[0], im_log.shape[1])
    plt.subplot(2, 3, 1)
    plt.imshow(im_log, cmap='gray')
    plt.title("im_log")
    grad = get_gradient(im_log)
    print(grad.max(), grad.min())
    plt.subplot(2, 3, 2)
    grad2 = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    plt.imshow(grad2, cmap='gray')
    plt.title("grad")
    threshold = get_derivative_threshold(grad)
    binary_im = to_binary(grad, threshold)
    plt.subplot(2, 3, 3)
    plt.imshow(binary_im, cmap='gray')
    plt.title("binary")
    convolve_img = get_convolved_im(window_height, window_width, binary_im, False)
    plt.subplot(2, 3, 4)
    plt.imshow(convolve_img, cmap='gray')
    plt.title("convolve_img")
    convolve_im_binary = conv_to_binary(convolve_img, np.mean(convolve_img))
    plt.subplot(2, 3, 5)
    plt.imshow(convolve_im_binary, cmap='gray')
    plt.title("convolve_im_binary")
    filtered_im = filter_relevant_pixels_quarter(np.copy(convolve_im_binary), window_width, window_height)
    plt.subplot(2, 3, 6)
    plt.imshow(filtered_im, cmap='gray')
    plt.title("filtered_im")
    center_windows = np.argwhere(filtered_im == 1)
    hdr_points = np.argwhere(filtered_im == 0.5)
    new_im = np.copy(im_rgb)
    plt.savefig(output_path)

    print("====== ldr windows number = ", len(center_windows))
    blue_lst = []
    for i, pos in enumerate(center_windows, 0):
        y0, y1, x0, x1 = get_window_borders_from_center(im_log.shape[1], im_log.shape[0], pos[1], window_width, pos[0],
                                                        window_height)
        cur_window = get_window(im_rgb, x0, x1, y0, y1)
        if np.nanmax(cur_window) == 0:
            print("prob")
        cur_window = cur_window / np.nanmax(cur_window)
        new_im[y0:y1, x0:x1] = cur_window
        cur_im[pos[0], pos[1], 0] = 1
        # if 162 < pos[0] < 165 and 138 < pos[1] < 142:
        #     print(pos)
        #     # new_im[pos[0], pos[1], 0] = 1
        #     # new_im[pos[0], pos[1], 1] = 0
        #     # new_im[pos[0], pos[1], 2] = 0
        #     y0, y1, x0, x1 = get_window_borders_from_center(im_log.shape[1], im_log.shape[0], pos[1], window_width * 4, pos[0],
        #                                                 window_height * 4)
        #     show_wind = get_window(new_im, x0, x1, y0, y1)
        #     print_image_details(show_wind, "wind")
        if 135 < pos[1] < 145 and 142 < pos[0] < 160:
            y0, y1, x0, x1 = get_window_borders_from_center(im_log.shape[1], im_log.shape[0], pos[1], window_width * 4,
                                                            pos[0], window_height * 4)
            show_wind = get_window(im_rgb, x0, x1, y0, y1)

            y0, y1, x0, x1 = get_window_borders_from_center(im_log.shape[1], im_log.shape[0], pos[1], window_width,
                                                            pos[0], window_height)
            true_wind = get_window(im_rgb, x0, x1, y0, y1)
            fix_wind = true_wind / np.nanmax(true_wind)
            cur_im[y0:y1, x0:x1] = fix_wind

            y0, y1, x0, x1 = get_window_borders_from_center(im_log.shape[1], im_log.shape[0], pos[1], window_width * 4,
                                                            pos[0], window_height * 4)
            show_wind_fix = get_window(cur_im, x0, x1, y0, y1)

            print_image_details(true_wind, "true wind")
            title = "max_wind = " + str(np.nanmax(true_wind)) + "\n" + "min_wind = " + str(
                np.nanmin(true_wind)) + "\n" + "mean_wind = " + str(np.nanmean(true_wind))

            # draw_results(show_wind, True, "HDR", show_wind_fix, False, "NEW", title, False, True,
            #              output_path + str(pos))

            # plot_image_coor(show_wind)

    # for pos_h in hdr_points:
    #     new_im[pos_h[0], pos_h[1], 0] = 1
    #     new_im[pos_h[0], pos_h[1], 1] = 0
    #     new_im[pos_h[0], pos_h[1], 2] = 0

    return new_im


def get_file_name(path):
    pre = os.path.splitext(path)[0]
    return str(pre)


def to_gray(im):
    return np.dot(im[..., :3], [0.299, 0.587, 0.114]).astype('float32')


def run_single(rgb_img, output_path=""):
    # print("=============== run single =============")
    yiq_img = rgb2yiq(rgb_img)
    gray_img2 = to_gray(rgb_img)
    print_image_details(gray_img2, "gray_img2")
    gray_img = np.copy(yiq_img[:, :, 0])
    print_image_details(gray_img,"yiq_img")
    img_log = log_transform(gray_img)
    print_image_details(img_log, "img_log")
    # if train:
    # fixed_rgb, center_pos, window_width, window_height = fix_high_luminance(img_log, np.copy(rgb_img), train)
    return fix_high_luminance(img_log, np.copy(rgb_img), output_path)


def plot_image_coor(img):
    fig, ax = plt.subplots()
    im = ax.imshow(img[:, :, 0], interpolation='none')
    ax.format_coord = Formatter(im)
    plt.show()


def run(input_path, output_path, train):
    print("=============== start image processing =====================")
    print("=============== train = %r  ==============================" % train)
    for img_name in os.listdir(input_path):
        im_path = input_path + "/" + img_name
        rgb_img = read_img(im_path)
        print_image_details(rgb_img, "befor : " + img_name)
        fixed_rgb = run_single(rgb_img, output_path + get_file_name(img_name))
        plt.imshow(fixed_rgb)
        plt.show()
        yiq_img = rgb2yiq(fixed_rgb)
        gray_img = np.copy(yiq_img[:, :, 0])
        plt.imshow(gray_img, cmap='gray')
        plt.show()
        # plot_image_coor(fixed_rgb)
        draw_results(rgb_img, True, "origin", fixed_rgb, False, "ldr", output_path + get_file_name(img_name) + "_comp",
                     False, True)
        save_results(fixed_rgb, output_path + get_file_name(img_name) + ".jpg")
        print_image_details(fixed_rgb, "afrter : " + img_name)
    print("=============== finish image processing =====================")


def run_single_window_tone_map(im_path, reshape=False):
    rgb_img = read_img(im_path)
    print_image_details(rgb_img, "rgb_im")
    fixed_rgb = run_single(rgb_img)
    print_image_details(fixed_rgb, "fixed_rgb")
    # plt.imshow(fixed_rgb)
    # plt.show()
    return fixed_rgb



# run("/Users/yaelvinker/PycharmProjects/lab/old_files/transfer_to_ldr/hdr_im", "/Users/yaelvinker/PycharmProjects/lab/old_files/transfer_to_ldr/res/", False)
