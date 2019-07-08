import argparse
import matplotlib.pyplot as plt
import os
import pathlib
import skimage
import imageio
import cv2
import numpy as np
import hdr_image_utils as utils

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_MAX_VALUE = 255
IMAGE_SCALE = 100


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--in_dir", type=str, default=os.path.join("hdr_images"))
    parser.add_argument("--out_dir", type=str, default="ldr_data1")
    args = parser.parse_args()

    dataroot_hdr = os.path.join(args.in_dir)
    dataroot_ldr = os.path.join(args.out_dir)

    print("=====================")
    print("DIR ROOT HDR: ", dataroot_hdr)
    print("DIR ROOT LDR: ", dataroot_ldr)
    print("=====================\n")
    return dataroot_hdr, dataroot_ldr


def create_tone_map():
    dataroot_hdr, dataroot_ldr = parse_arguments()
    for img_name in os.listdir(dataroot_hdr):
        im_path = os.path.join(dataroot_hdr, img_name)
        rgb_img = utils.read_img(im_path)
        tone_map1 = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        im1_dub = tone_map1.process(rgb_img.copy().astype(np.float32)[:, :, ::-1])
        im1 = np.clip(im1_dub, 0, 1).astype('float32')[:, :, ::-1]
        # im1 = (im1 - 0.5) / 0.5
        utils.save_results(im1, os.path.join(dataroot_ldr, utils.get_file_name(img_name) + "tonemap"))


def hdr_test(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path, format='HDR-FI')
    max_origin = np.nanmax(im_origin)
    im = (im_origin / max_origin) * IMAGE_SCALE
    log_im = np.log(im + 1)
    im_display = (((np.exp(log_im) - 1) / IMAGE_SCALE) * max_origin)
    utils.print_image_details(im_display,"im_display")
    utils.print_image_details(im_origin, "im_origin")
    # return cv2.resize(np.log(im + 1), (500, 500))


if __name__ == '__main__':
    dataroot_in_ldr, dataroot_out_ldr = parse_arguments()
    for img_name in os.listdir(dataroot_in_ldr):
        im_path = os.path.join(dataroot_in_ldr, img_name)
        hdr_test(os.path.join(dataroot_in_ldr, img_name))
        # path = pathlib.Path(im_path)
        # im_origin = imageio.imread(path)
        # print(im_origin.shape)
        # im = (im_origin / IMAGE_MAX_VALUE) * IMAGE_SCALE
        # im_log = np.log(im + 1)
        # im_display = (((np.exp(im_log) - 1) / IMAGE_SCALE) * IMAGE_MAX_VALUE).astype("uint8")
        # utils.save_results(im_log, os.path.join(dataroot_out_ldr, utils.get_file_name(img_name)))





