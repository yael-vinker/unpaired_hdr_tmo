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
