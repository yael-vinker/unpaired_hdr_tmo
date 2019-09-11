import torchvision.transforms as transforms
import argparse
import params
import os
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

def to_gray(im):
    return np.dot(im[...,:3], [0.299, 0.587, 0.114])

def ldr_loader(path):
    path = pathlib.Path(path)
    im_origin = imageio.imread(path)
    im = im_origin / 255
    return im

def display_tensor(tensor_im, isgray):
    np_im = np.array(tensor_im.permute(1, 2, 0))
    im = (np_im - np.min(np_im)) / (np.max(np_im) - np.min(np_im))
    if isgray:
        gray = np.squeeze(im)
        plt.imshow(gray, cmap='gray')
    else:
        plt.imshow(im)
    plt.show()

def print_result(output_dir, testMode):
    for img_name in os.listdir(output_dir):
        im_path = os.path.join(output_dir, img_name)
        data = np.load(im_path, allow_pickle=True)
        if testMode:
            print("---test mode---")
            input_im = data[()]["input_image"]
            color_im = data[()]["display_image"]
            hdr_image_utils.print_tensor_details(input_im, "input_im " + img_name)
            # display_tensor(input_im, True)
            hdr_image_utils.print_tensor_details(color_im, "display_image " + img_name)
            # display_tensor(color_im, False)
        else:
            input_im = data[()]["input_image"]
            hdr_image_utils.print_tensor_details(input_im, img_name)

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

def get_transforms(images_mean_, channels_, display=False):
    if images_mean_ == 0.5:
        transform_custom_ = transforms.Compose([
            transforms_.Scale(params.input_size),
            transforms_.CenterCrop(params.input_size),
            transforms_.ToTensor(),
        ])

    elif channels_ == 3:
        if display:
            transform_custom_ = transforms.Compose([
                transforms_.Scale(params.input_size),
                transforms_.CenterCrop(params.input_size),
                transforms_.ToTensor(),
            ])
        else:
            transform_custom_ = transforms.Compose([
                transforms_.Scale(params.input_size),
                transforms_.CenterCrop(params.input_size),
                transforms_.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    else:
        transform_custom_ = transforms.Compose([
            transforms_.Scale(params.input_size),
            transforms_.CenterCrop(params.input_size),
            transforms_.ToTensor(),
            transforms_.Normalize(0.5, 0.5),
        ])
    return transform_custom_

def create_npy_data(input_dir, output_dir, isLdr, channels_, images_mean):
    from os import path
    output_transform = get_transforms(images_mean, channels_)
    for img_name, i in zip(os.listdir(input_dir), range(896)):
        im_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        if not path.exists(output_path):
            if isLdr:
                rgb_img = ldr_loader(im_path)
            else:
                rgb_img = hdr_loader(im_path)
            if channels == 1:
                output_im = to_gray(rgb_img)
            else:
                output_im = rgb_img
            transformed_output_im = output_transform(output_im)
            np.save(output_path, transformed_output_im)
            print(output_path)
        print(i)


def create_test_data(input_dir, output_dir, isLdr, channels_, images_mean):
    output_transform = get_transforms(images_mean, channels_)
    display_transform = get_transforms(images_mean, 3, True)
    for img_name in os.listdir(input_dir):
        im_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '_one_dim.npy')
        if isLdr:
            rgb_img = ldr_loader(im_path)
        else:
            rgb_img = hdr_loader(im_path)
        if channels == 1:
            output_im = to_gray(rgb_img)
        else:
            output_im = rgb_img
        # hdr_image_utils.print_image_details(np.asarray(rgb_img), "---- before "+img_name)
        transformed_output_im = output_transform(output_im)
        transformed_display_im = display_transform(rgb_img)
        data = {'input_image': transformed_output_im, 'display_image': transformed_display_im}
        np.save(output_path, data)
        print(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--root_hdr", type=str, default=os.path.join("data/hdr_data/hdr_data"))
    parser.add_argument("--root_ldr", type=str, default=os.path.join("data/ldr_data/ldr_data"))
    # parser.add_argument("--output_hdr", type=str, default=os.path.join("data/test_hdr_npy/test_hdr_npy"))
    # parser.add_argument("--output_ldr", type=str, default=os.path.join("data/test_ldr_npy/test_ldr_npy"))
    parser.add_argument("--output_hdr", type=str, default=os.path.join("data/test_hdr_npy/test_hdr_npy"))
    parser.add_argument("--output_ldr", type=str, default=os.path.join("data/test_ldr_npy/test_ldr_npy"))
    args = parser.parse_args()
    input_hdr_dir = os.path.join(args.root_hdr)
    input_ldr_dir = os.path.join(args.root_ldr)
    output_hdr_dir = os.path.join(args.output_hdr)
    output_ldr_dir = os.path.join(args.output_ldr)
    channels = 1
    images_mean = 0
    # create_log_npy_data(input_dir, output_dir)
    create_test_data(input_hdr_dir, output_hdr_dir, False, channels, images_mean)
    create_test_data(input_ldr_dir, output_ldr_dir, True, channels, images_mean)
