import numpy as np
import torch
import torch.utils.data


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


def get_window_pad(im, x0, x1, y0, y1, max_height, max_width):
    wind = im[:, y0: y1, x0: x1]
    pad_f = torch.nn.ZeroPad2d((0, int(max_width - (x1 - x0)), 0, int(max_height - (y1 - y0))))
    return pad_f(wind)


def get_window_pad_arr(im, boarders, max_height, max_width):
    y0, y1, x0, x1 = boarders[0], boarders[1], boarders[2], boarders[3]
    with torch.no_grad():
        wind = im[:, y0: y1, x0: x1]
        pad_f = torch.nn.ZeroPad2d((0, int(max_width - (x1 - x0)), 0, int(max_height - (y1 - y0))))
    return pad_f(wind)


def fix_high_luminance_2(center_windows, tensor_image, fake_img, mse_loss, window_height,
                         window_width, half_window_height, half_window_width):
    """

    :param im_log: grayscale image after log transform
    :param im_rgb: the original image in RGB
    :param filter_win_size: determine the overlap between the windows extracted after the convolution.
    the default value is quarter (3/4 overlap).
    :return:
    """
    window_borders = [get_window_borders_from_center(256, 256, pos[1], half_window_width, pos[0],
                                                     half_window_height) for pos in center_windows]
    div_factor_ = np.sum(np.array([((win_boarder[1] - win_boarder[0]) * (win_boarder[3] - win_boarder[2])) * 3
                                   for win_boarder in window_borders]))
    hdr_windows_ = torch.stack([get_window_pad_arr(tensor_image, win_border, window_height, window_width)
                                for win_border in window_borders])
    hdr_max_values_ = torch.stack([torch.max(cur_window_hdr) for cur_window_hdr in hdr_windows_])
    fake_windows_ = torch.stack([get_window_pad_arr(fake_img, win_border, window_height, window_width)
                                 for win_border in window_borders])
    hdr_windows_t = torch.stack([torch.div(x, y) for x, y in zip(hdr_windows_, hdr_max_values_)])
    loss = mse_loss(fake_windows_, hdr_windows_t)
    return loss / div_factor_


def run_all(rgb_img_tensor, fake_img, mse_loss, window_height, window_width, half_window_height, half_window_width,
            filtered_images_t):
    filtered_images = [x.detach().cpu().clone().numpy() for x in filtered_images_t]
    center_windows_ = [np.argwhere(filtered_im == 1) for filtered_im in filtered_images]
    loss_ = torch.sum(torch.stack([fix_high_luminance_2(points_, rgb_img_tensor[i], fake_img[i], mse_loss,
                                                        window_height, window_width, half_window_height,
                                                        half_window_width)
                                   for points_, i in zip(center_windows_, range(len(center_windows_)))]))
    return loss_
