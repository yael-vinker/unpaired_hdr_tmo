import torch
from utils import params
import torch.nn.functional as F
from utils import hdr_image_util

def load_data_set(data_root, dataset_properties, shuffle, hdrMode):
    from utils import ProcessedDatasetFolder
    npy_dataset = ProcessedDatasetFolder.ProcessedDatasetFolder(root=data_root,
                                                                dataset_properties=dataset_properties,
                                                                hdrMode=hdrMode)
    dataloader = torch.utils.data.DataLoader(npy_dataset, batch_size=dataset_properties["batch_size"],
                                             shuffle=shuffle, num_workers=params.workers, pin_memory=True)
    return dataloader


def load_train_data(dataset_properties, title):
    from utils import printer
    """
    :param isHdr: True if images in "dir_root" are in .hdr format, False otherwise.
    :param dir_root: path to wanted directory
    :param b_size: batch size
    :return: DataLoader object of images in "dir_root"
    """
    print("loading hdr train data from ", dataset_properties["train_root_npy"])
    train_hdr_dataloader = load_data_set(dataset_properties["train_root_npy"], dataset_properties,
                                         shuffle=True, hdrMode=True)
    train_ldr_dataloader = load_data_set(dataset_properties["train_root_ldr"], dataset_properties,
                                         shuffle=True, hdrMode=False)

    printer.print_dataset_details([train_hdr_dataloader, train_ldr_dataloader],
                                  [dataset_properties["train_root_npy"], dataset_properties["train_root_ldr"]],
                                  [title + "_hdr_dataloader", title + "_ldr_dataloader"],
                                  [True, False],
                                  [True, True])

    printer.load_data_dict_mode(train_hdr_dataloader, train_ldr_dataloader, title, images_number=2)
    return train_hdr_dataloader, train_ldr_dataloader


def load_test_data(dataset_properties, title):
    from utils import printer
    """
    :param isHdr: True if images in "dir_root" are in .hdr format, False otherwise.
    :param dir_root: path to wanted directory
    :param b_size: batch size
    :return: DataLoader object of images in "dir_root"
    """
    print("loading hdr train data from ", dataset_properties["test_dataroot_npy"])
    train_hdr_dataloader = load_data_set(dataset_properties["test_dataroot_npy"], dataset_properties,
                                         shuffle=True, hdrMode=True)
    train_ldr_dataloader = load_data_set(dataset_properties["test_dataroot_ldr"], dataset_properties,
                                         shuffle=True, hdrMode=False)

    printer.print_dataset_details([train_hdr_dataloader, train_ldr_dataloader],
                                  [dataset_properties["test_dataroot_npy"], dataset_properties["test_dataroot_ldr"]],
                                  [title + "_hdr_dataloader", title + "_ldr_dataloader"],
                                  [True, False],
                                  [True, True])

    printer.load_data_dict_mode(train_hdr_dataloader, train_ldr_dataloader, title, images_number=2)
    return train_hdr_dataloader, train_ldr_dataloader


def resize_im(im, add_frame, final_shape_addition):
    from math import floor
    print("original shape", im.shape)
    h, w = im.shape[1], im.shape[2]
    diffY, diffX = 0, 0
    h1 = (int(16 * floor(h / 16.)))
    w1 = (int(16 * floor(w / 16.)))
    diffh = h - h1
    diffw = w - w1
    im = im[:, diffh // 2:h - (diffh - diffh // 2), diffw // 2:w - (diffw - diffw // 2)]
    # if h % 2:
    #     im = im[:, diffh // 2:h - (diffh - diffh // 2), diffw // 2:w - (diffw - diffw // 2)]
    # if w % 2:
    #     im = im[:, :, 0: w - 1]
    if add_frame:
        diffY = hdr_image_util.closest_power(im.shape[1], final_shape_addition) - im.shape[1]
        diffX = hdr_image_util.closest_power(im.shape[2], final_shape_addition) - im.shape[2]
        im = add_frame_to_im(im, diffX=diffX, diffY=diffY)
    print("new shape", im.shape)
    return im, diffY, diffX


def calc_conv_params(h_in, stride, padding, dilation, kernel_size, output_padding):
    print(dilation[0] * (kernel_size[0] - 1) - padding[0])
    h_out = (h_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    print(h_out)



def crop_input_hdr_batch(input_hdr_batch, diffY, diffX):
    b, c, h, w = input_hdr_batch.shape
    th, tw = h - diffY, w - diffX
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    i, j, h, w = i, j, th, tw
    input_hdr_batch = input_hdr_batch[:, :, i:i + h, j:j + w]
    return input_hdr_batch


def add_frame_to_im(input_im, diffX, diffY):
    im = F.pad(input_im.unsqueeze(dim=0), (diffX // 2, diffX - diffX // 2,
                                           diffY // 2, diffY - diffY // 2), mode='replicate')
    im = torch.squeeze(im, dim=0)
    return im


def add_frame_to_im_batch(images_batch, diffX, diffY):
    images_batch = F.pad(images_batch, (diffX // 2, diffX - diffX // 2,
                                           diffY // 2, diffY - diffY // 2), mode='replicate')
    return images_batch




if __name__ == '__main__':
    from math import floor
    h, w = 780, 1052
    h1 = (int(16 * floor(h / 16.)))
    w1 = (int(16 * floor(w / 16.)))
    diffh = h - h1
    diffw = w - w1
    print(h1, diffh, w1, diffw)

    # im = im[:, diffh // 2:h - (diffh - diffh // 2), diffw // 2:w - (diffw - diffw // 2)]
    # calc_conv_params(h_in=20, stride=(2,2), padding=(0,1), dilation=(1,1), kernel_size=(5,5), output_padding=(0,0))
    # calc_conv_params(h_in=20, stride=(2, 2), padding=(0, 1), dilation=(1, 1), kernel_size=(4, 3), output_padding=(0, 0))