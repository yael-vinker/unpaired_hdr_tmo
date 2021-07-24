import inspect
import os
import sys

import activate_trained_model.pre_calc_lambdas as pre_calc_lambdas

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import tranforms
import models.unet_multi_filters.Unet as Generator
import torch.nn.functional as F
from utils import model_save_util, hdr_image_util, data_loader_util, params

extensions = [".hdr", ".dng", ".exr", ".npy"]


def run_trained_model(args):
    pre_calc_lambdas.calc_lambda(args, extensions)
    start = time.time()
    net_path = os.path.join(args.model_path, "trained_weights.pth")
    train_settings_path = os.path.join(args.model_path, "run_settings.npy")
    output_images_path = os.path.join(args.output_path)
    model_params = model_save_util.get_model_params(args.model_name, train_settings_path)
    # model_params["factor_coeff"] = 0.2
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    run_model_on_path(model_params, device, net_path, args.input_images_path,
                      output_images_path, args.f_factor_path, model_params["final_shape_addition"])
    print("tone mapping took [%.2f] seconds" % (time.time() - start))


def run_model_on_path(model_params, device, net_path, input_images_path, output_images_path,
                      f_factor_path, final_shape_addition):
    net_G = load_g_model(model_params, device, net_path)
    print("\nModel [%s] was loaded successfully\n" % model_params["model"])
    names = os.listdir(input_images_path)
    for img_name in names:
        im_path = os.path.join(input_images_path, img_name)
        print("processing [%s]" % img_name)
        if os.path.splitext(img_name)[1] in extensions:
            fast_run_model_on_single_image(net_G, im_path, device, os.path.splitext(img_name)[0],
                                           output_images_path, model_params, f_factor_path, final_shape_addition)
        # else:
        #     raise Exception('invalid hdr file format: %s' % os.path.splitext(img_name)[1])


def fast_run_model_on_single_image(G_net, im_path, device, im_name, output_path, model_params,
                                   f_factor_path, final_shape_addition):
    rgb_img, gray_im_log, f_factor = fast_load_inference(im_path, f_factor_path, model_params["factor_coeff"], device)
    rgb_img, diffY, diffX = data_loader_util.resize_im(rgb_img, model_params["add_frame"], final_shape_addition)
    gray_im_log, diffY, diffX = data_loader_util.resize_im(gray_im_log, model_params["add_frame"], final_shape_addition)
    with torch.no_grad():
        fake = G_net(gray_im_log.unsqueeze(0), apply_crop=model_params["add_frame"], diffY=diffY, diffX=diffX)
    max_p = np.percentile(fake.cpu().numpy(), 99.5)
    min_p = np.percentile(fake.cpu().numpy(), 0.5)
    # max_p = np.percentile(fake.cpu().numpy(), 100)
    # min_p = np.percentile(fake.cpu().numpy(), 0.0001)
    fake2 = fake.clamp(min_p, max_p)
    fake_im_gray_stretch = (fake2 - fake2.min()) / (fake2.max() - fake2.min())
    fake_im_color2 = hdr_image_util.back_to_color_tensor(rgb_img, fake_im_gray_stretch[0], device)
    h, w = fake_im_color2.shape[1], fake_im_color2.shape[2]
    im_max = fake_im_color2.max()
    fake_im_color2 = F.interpolate(fake_im_color2.unsqueeze(dim=0), size=(h - diffY, w - diffX),
                                   mode='bicubic',
                                   align_corners=False).squeeze(dim=0).clamp(min=0, max=im_max)
    hdr_image_util.save_gray_tensor_as_numpy_stretch(fake_im_color2, output_path,
                                                     im_name + "_fake_clamp_and_stretch")


def fast_load_inference(im_path, f_factor_path, factor_coeff, device):
    data = np.load(f_factor_path, allow_pickle=True)[()]
    f_factor = data[os.path.splitext(os.path.basename(im_path))[0]] * 255 * factor_coeff
    rgb_img = hdr_image_util.read_hdr_image(im_path)
    rgb_img = hdr_image_util.reshape_image(rgb_img, train_reshape=False)
    rgb_img = tranforms.hdr_im_transform(rgb_img).to(device)
    # TODO: check what to do with negative exr
    if rgb_img.min() < 0:
        rgb_img = rgb_img - rgb_img.min()
    gray_im = hdr_image_util.to_gray_tensor(rgb_img).to(device)
    gray_im = gray_im - gray_im.min()
    gray_im = torch.log10((gray_im / gray_im.max()) * f_factor + 1)
    gray_im = gray_im / gray_im.max()
    return rgb_img, gray_im, f_factor


def load_g_model(model_params, device, net_path):
    G_net = create_G_net(model_params, device, is_checkpoint=True, activation="relu", output_dim=1)
    checkpoint = torch.load(net_path, map_location=device)
    state_dict = checkpoint['modelG_state_dict']
    if "module" in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    G_net.load_state_dict(new_state_dict)
    G_net.to(device)
    G_net.eval()
    return G_net


def create_G_net(model_params, device_, is_checkpoint, activation, output_dim):
    layer_factor = get_layer_factor(model_params["con_operator"])
    if model_params["model"] != params.unet_network:
        assert 0, "Unsupported g model request: {}".format(model_params["model"])
    new_net = Generator.UNet(model_params["input_dim"], output_dim, model_params["last_layer"],
                             depth=model_params["depth"], layer_factor=layer_factor,
                             con_operator=model_params["con_operator"],
                             filters=model_params["filters"], bilinear=model_params["bilinear"],
                             network=model_params["model"], dilation=0,
                             to_crop=model_params["add_frame"], unet_norm=model_params["unet_norm"],
                             stretch_g=model_params["stretch_g"],
                             activation=activation, doubleConvTranspose=model_params["g_doubleConvTranspose"],
                             padding_mode=model_params["padding"],
                             convtranspose_kernel=model_params["convtranspose_kernel"],
                             up_mode=model_params["up_mode"]).to(device_)
    return set_parallel_net(new_net, device_, is_checkpoint, "Generator")


def get_layer_factor(con_operator):
    if con_operator in params.layer_factor_2_operators:
        return 2
    elif con_operator in params.layer_factor_3_operators:
        return 3
    elif con_operator in params.layer_factor_4_operators:
        return 4
    else:
        assert 0, "Unsupported con_operator request: {}".format(con_operator)


def set_parallel_net(net, device_, is_checkpoint, net_name):
    # Handle multi-gpu if desired
    if (device_.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        net = nn.DataParallel(net, list(range(torch.cuda.device_count())))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    if not is_checkpoint:
        net.apply(weights_init_xavier)
        print("Weights for " + net_name + " were initialized successfully")
    return net


def weights_init_xavier(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
        torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


default_params = {"model_path": "model_weights",
                  "model_name": "11_08_lr15D_size268_D_[1,1,1]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate__noframe__min_log_0.1hist_fit_",
                  "input_images_path": "input_images",
                  # "input_images_path": "/Users/yaelvinker/Documents/university/lab/ECCV/improve_images_for_paper/input",
                  "f_factor_path": "/Users/yaelvinker/PycharmProjects/lab/run_trained_model/lambda_data/exr_hist_dict_20_bins.npy",
                  # "output_path": "output_original_images",
                  # "output_path": "/Users/yaelvinker/Documents/university/lab/ECCV/improve_images_for_paper/double3_F/",
                  "output_path": "output",
                  "mean_hist_path": "lambda_data/ldr_avg_hist_900_images_20_bins.npy",
                  "lambda_output_path": "lambda_data",
                  "bins": 20}


def print_args(args):
    print("========== input arguments ==========")
    for arg in args.__dict__:
        print("%s [%s]" % (arg, args.__dict__[arg]))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--model_name", type=str, default=default_params["model_name"])
    parser.add_argument("--input_images_path", type=str, default=default_params["input_images_path"])
    parser.add_argument("--output_path", type=str, default=default_params["output_path"])
    parser.add_argument("--model_path", type=str, default=default_params["model_path"])
    parser.add_argument("--f_factor_path", type=str, default=default_params["f_factor_path"])

    # lambda calc params
    parser.add_argument("--mean_hist_path", type=str, default=default_params["mean_hist_path"])
    parser.add_argument("--lambda_output_path", type=str, default=default_params["lambda_output_path"])
    parser.add_argument("--bins", type=str, default=default_params["bins"])
    args = parser.parse_args()
    print_args(args)
    run_trained_model(args)
