import torchvision.transforms as transforms
import argparse
import params
import os
import shutil
import tranforms as transforms_
import imageio
import pathlib
import numpy as np
import hdr_image_utils
import matplotlib.pyplot as plt

def hdr_log_loader(path):
    path_lib_path = pathlib.Path(path)
    file_extension = os.path.splitext(path)[1]
    if file_extension == ".hdr":
        im_origin = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
    elif file_extension == ".dng":
        im_origin = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
    else:
        raise Exception('invalid hdr file format: {}'.format(file_extension))
    im_log = np.log(im_origin + 1)
    max_log = np.max(im_log)
    im = (im_log / max_log)
    return im


def hdr_loader(path):
    path_lib_path = pathlib.Path(path)
    file_extension = os.path.splitext(path)[1]
    if file_extension == ".hdr":
        im_origin = imageio.imread(path_lib_path, format="HDR-FI").astype('float32')
    elif file_extension == ".dng":
        im_origin = imageio.imread(path_lib_path, format="RAW-FI").astype('float32')
    else:
        raise Exception('invalid hdr file format: {}'.format(file_extension))
    max_origin = np.max(im_origin)
    im = (im_origin / max_origin)
    return im

def ldr_loader(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    return im_origin

def test_result(output_dir):
    for img_name in os.listdir(output_dir):
        im_path = os.path.join(output_dir, img_name)
        data = np.load(im_path, allow_pickle=True)
        hdr_image_utils.print_image_details(data, img_name)

def create_log_npy_data(input_dir, output_dir):
    dtype = np.float32
    transform_custom = transforms.Compose([
        transforms_.Scale(params.input_size, dtype),
        transforms_.CenterCrop(params.input_size),
        transforms_.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    for img_name in os.listdir(input_dir):
        im_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        log_im = hdr_log_loader(im_path)
        transformed_im = transform_custom(log_im)
        np.save(output_path, transformed_im)
        print(output_path)

def create_npy_data(input_dir, output_dir, isLdr=False):
    dtype = np.float32
    if isLdr:
        dtype = np.uint8
    transform_custom = transforms.Compose([
        transforms_.Scale(params.input_size, dtype),
        transforms_.CenterCrop(params.input_size),
        transforms_.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    for img_name in os.listdir(input_dir):
        im_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        if isLdr:
            rgb_img = ldr_loader(im_path)
        else:
            rgb_img = hdr_loader(im_path)
        transformed_im = transform_custom(rgb_img)
        np.save(output_path, transformed_im)
        print(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--data_root_hdr", type=str, default=os.path.join("hdr_data/hdr_data"))
    parser.add_argument("--data_output", type=str, default=os.path.join("hdr_npy/hdr_npy"))
    parser.add_argument("--is_ldr", type=str, default="no")
    args = parser.parse_args()
    input_dir = os.path.join(args.data_root_hdr)
    output_dir = os.path.join(args.data_output)
    isLdr = False
    if args.is_ldr == "yes":
        isLdr = True
    print(isLdr)
    create_log_npy_data(input_dir, output_dir)
    # create_npy_data(input_dir, output_dir, isLdr)
    # test_result(output_dir)