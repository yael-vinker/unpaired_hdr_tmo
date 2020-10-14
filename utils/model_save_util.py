import inspect
import os
import sys
from utils import params
from models import Discriminator
import imageio
import numpy as np
import torch
import torch.nn as nn
import utils.hdr_image_util as hdr_image_util
import utils.data_loader_util as data_loader_util
import tranforms
import models.unet_multi_filters.Unet as Generator
import data_generator.create_dng_npy_data as create_dng_npy_data
import re
import torch.nn.functional as F
import argparse
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# ======================================
# =========== TRAIN RELATED ============
# ======================================
def weights_init(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init_xavier(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('Linear') != -1) and hasattr(m, 'weight'):
        torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def set_parallel_net(net, device_, is_checkpoint, net_name, use_xaviar=False):
    # Handle multi-gpu if desired
    if (device_.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        net = nn.DataParallel(net, list(range(torch.cuda.device_count())))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    if not is_checkpoint:
        if use_xaviar:
            net.apply(weights_init_xavier)
        else:
            net.apply(weights_init)
        print("Weights for " + net_name + " were initialized successfully")
    return net


def create_G_net(model, device_, is_checkpoint, input_dim_, last_layer, filters, con_operator, unet_depth_,
                 add_frame, unet_norm, stretch_g, activation, use_xaviar, output_dim, apply_exp,
                 g_doubleConvTranspose):
    layer_factor = get_layer_factor(con_operator)
    if model != params.unet_network:
        assert 0, "Unsupported g model request: {}".format(model)
    new_net = Generator.UNet(input_dim_, output_dim, last_layer, depth=unet_depth_,
                             layer_factor=layer_factor, con_operator=con_operator, filters=filters,
                             bilinear=False, network=model, dilation=0, to_crop=add_frame,
                             unet_norm=unet_norm, stretch_g=stretch_g,
                             activation=activation, apply_exp=apply_exp,
                             doubleConvTranspose=g_doubleConvTranspose).to(device_)
    return set_parallel_net(new_net, device_, is_checkpoint, "Generator", use_xaviar)


def create_D_net(input_dim_, down_dim, device_, is_checkpoint, norm, use_xaviar, d_model,
                 d_nlayers, last_activation, num_D, d_fully_connected, simpleD_maxpool):
    if d_model == "original":
        new_net = Discriminator.Discriminator(params.input_size, input_dim_,
                                              down_dim, norm, last_activation,
                                              d_fully_connected, d_nlayers).to(device_)
    elif d_model == "patchD":
        new_net = Discriminator.NLayerDiscriminator(input_dim_, ndf=down_dim, n_layers=d_nlayers,
                                                    norm_layer=norm, last_activation=last_activation).to(device_)
    elif "multiLayerD" in d_model:
        new_net = Discriminator.MultiscaleDiscriminator(params.input_size, d_model, input_dim_, ndf=down_dim,
                                                        n_layers=d_nlayers,
                                                        norm_layer=norm, last_activation=last_activation,
                                                        num_D=num_D, d_fully_connected=d_fully_connected,
                                                        simpleD_maxpool=simpleD_maxpool).to(device_)
    elif d_model == "simpleD":
        new_net = Discriminator.SimpleDiscriminator(params.input_size, input_dim_,
                                              down_dim, norm, last_activation, simpleD_maxpool).to(device_)
    else:
        assert 0, "Unsupported d model request: {}".format(d_model)
    return set_parallel_net(new_net, device_, is_checkpoint, "Discriminator", use_xaviar)


def save_model(path, epoch, output_dir, netG, optimizerG, netD, optimizerD):
    path = os.path.join(output_dir, path, "net_epoch_" + str(epoch) + ".pth")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'modelD_state_dict': netD.state_dict(),
        'modelG_state_dict': netG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
    }, path)


def save_discriminator_model(path, epoch, output_dir, netD, optimizerD):
    path = os.path.join(output_dir, path, "net_epoch_" + str(epoch) + ".pth")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'modelD_state_dict': netD.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }, path)


def get_layer_factor(con_operator):
    if con_operator in params.layer_factor_2_operators:
        return 2
    elif con_operator in params.layer_factor_3_operators:
        return 3
    elif con_operator in params.layer_factor_4_operators:
        return 4
    else:
        assert 0, "Unsupported con_operator request: {}".format(con_operator)


# ======================================
# ============ TEST RELATED ============
# ========= USE TRAINED MODEL ==========
# ======================================
def apply_models_from_arch_dir(input_format, device, images_source, arch_dir,
                               models_names, model_epoch, output_dir_name):
    input_images_path = get_hdr_source_path(images_source)
    for i in range(len(models_names)):
        model_name = models_names[i]
        model_params = get_model_params(model_name)
        f_factor_path = get_f_factor_path(images_source, model_params["gamma_log"], use_new_f=False)
        print("cur model = ", model_name)
        cur_model_path = os.path.join(arch_dir, model_name)
        if os.path.exists(cur_model_path):
            output_path = os.path.join(cur_model_path, input_format + output_dir_name)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            net_path = os.path.join(cur_model_path, "models")
            net_name = "net_epoch_" + str(model_epoch) + ".pth"
            cur_net_path = os.path.join(net_path, net_name)
            if os.path.exists(cur_net_path):
                cur_output_path = output_path
                if not os.path.exists(cur_output_path):
                    os.mkdir(cur_output_path)
                run_model_on_path(model_params, device, cur_net_path, input_images_path, cur_output_path,
                                  f_factor_path, None, input_images_path)
            else:
                print("model %s does not exists: " % cur_net_path)
        else:
            print("model %s does not exists" % cur_model_path)


def run_model_on_path(model_params, device, cur_net_path, input_images_path, output_images_path,
                      f_factor_path, net_G, names_path):
    import time
    a = time.time()
    if not net_G:
        net_G = load_g_model(model_params, device, cur_net_path)
    print("model " + model_params["model"] + " was loaded successfully")
    print("took %.4f seconds to load model" % (time.time() - a))
    names = os.listdir(names_path)
    ext = os.path.splitext(os.listdir(input_images_path)[0])[1]
    for img_name in names:
        im_path = os.path.join(input_images_path, os.path.splitext(img_name)[0] + ext)
        if not os.path.exists(os.path.join(output_images_path, os.path.splitext(img_name)[0] + ".png")):
            print("working on ", img_name)
            if os.path.splitext(img_name)[1] == ".hdr" or os.path.splitext(img_name)[1] == ".exr" \
                    or os.path.splitext(img_name)[1] == ".dng" or os.path.splitext(img_name)[1] == ".npy":
                run_model_on_single_image(net_G, im_path, device, os.path.splitext(img_name)[0],
                                          output_images_path, model_params, f_factor_path)
        else:
            print("%s in already exists" % (img_name))
        print("took %.4f seconds to run model" % (time.time() - a))


def load_g_model(model_params, device, net_path):
    G_net = create_G_net(model_params["model"], device, True, model_params["input_dim"], model_params["last_layer"],
                         model_params["filters"], model_params["con_operator"], model_params["depth"],
                         model_params["add_frame"], model_params["unet_norm"],  model_params["stretch_g"],
                         "relu", use_xaviar=False, output_dim=1, apply_exp=False,
                         g_doubleConvTranspose=model_params["g_doubleConvTranspose"])

    checkpoint = torch.load(net_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['modelG_state_dict']
    print("from loaded model ", list(state_dict.keys())[0])
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


def run_model_on_single_image(G_net, im_path, device, im_name, output_path, model_params, f_factor_path):
    test_mode_f_factor = model_params["test_mode_f_factor"]
    test_mode_frame = model_params["test_mode_frame"]
    rgb_img, gray_im_log, f_factor = create_dng_npy_data.hdr_preprocess(im_path,
                                                                        factor_coeff=model_params["factor_coeff"],
                                                                        train_reshape=False,
                                                                        gamma_log=model_params["gamma_log"],
                                                                        f_factor_path=f_factor_path,
                                                                        use_new_f=model_params["use_new_f"],
                                                                        data_trc=model_params["data_trc"],
                                                                        test_mode=test_mode_f_factor,
                                                                        use_contrast_ratio_f=model_params["use_contrast_ratio_f"])

    rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)
    diffY, diffX = 0, 0
    if test_mode_frame:
        print("original shape",gray_im_log.shape)
        diffY = hdr_image_util.closest_power(gray_im_log.shape[1]) - gray_im_log.shape[1]
        diffX = hdr_image_util.closest_power(gray_im_log.shape[2]) - gray_im_log.shape[2]
        gray_im_log = F.pad(gray_im_log.unsqueeze(dim=0), (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2), mode='replicate')
        gray_im_log = torch.squeeze(gray_im_log, dim=0)
        print("new shape",gray_im_log.shape)

    if model_params["add_frame"]:
        gray_im_log = data_loader_util.add_frame_to_im(gray_im_log)
    gray_im_log = gray_im_log.to(device)
    preprocessed_im_batch = gray_im_log.unsqueeze(0)
    if model_params["manualD"] == "double":
        interp_params = [0, 0.2, 0.4, 0.5, 0.8, 1]
        for a in interp_params:
            file_name = im_name + "_" + str(a)
            run_model_on_im_and_save_res(preprocessed_im_batch, G_net, rgb_img, output_path,
                                     file_name, test_mode_frame, diffY, diffX, additional_channel=a)
    elif model_params["manualD"] == "single":
        file_name = im_name + "_1"
        run_model_on_im_and_save_res(preprocessed_im_batch, G_net, rgb_img, output_path,
                                     file_name, test_mode_frame, diffY, diffX, additional_channel=1.0)
    elif model_params["manualD"] == "none":
        file_name = im_name
        run_model_on_im_and_save_res(preprocessed_im_batch, G_net, rgb_img, output_path,
                                     file_name, test_mode_frame, diffY, diffX, additional_channel=None)


def run_model_on_im_and_save_res(im_log_normalize_tensor, netG, rgb_img, out_dir, file_name,
                                 test_mode, diffY, diffX, additional_channel):
    if additional_channel is not None:
        weight_channel = torch.full(im_log_normalize_tensor.shape, additional_channel).type_as(
            im_log_normalize_tensor)
        im_log_normalize_tensor = torch.cat([im_log_normalize_tensor, weight_channel], dim=1)
    with torch.no_grad():
        print(im_log_normalize_tensor.max(), im_log_normalize_tensor.mean(), im_log_normalize_tensor.min())
        fake = netG(im_log_normalize_tensor.detach())
        print(fake.max(), fake.mean(), fake.min())
    original_im_tensor = rgb_img.unsqueeze(0)
    if test_mode:
        fake = fake[:, :, diffY // 2:fake.shape[2] - (diffY - diffY // 2),
                             diffX // 2:fake.shape[3] - (diffX - diffX // 2)]

    # fake_im_gray_stretch = (fake[0] - fake[0].min()) / (fake[0].max() - fake[0].min())

    fake_im_gray_stretch = fake[0]
    file_name_gray = file_name + "_gray"
    hdr_image_util.save_color_tensor_as_numpy(fake_im_gray_stretch, out_dir, file_name_gray)

    fake_im_color = hdr_image_util.back_to_color_batch(original_im_tensor,
                                                       fake_im_gray_stretch.unsqueeze(dim=0))
    # hdr_image_util.save_color_tensor_as_numpy(fake_im_color[0], out_dir, file_name)
    # file_name = file_name + "_color_stretch"
    hdr_image_util.save_gray_tensor_as_numpy_stretch(fake_im_color[0], out_dir, file_name)


def run_trained_model_from_path(model_name):
    net_path = os.path.join("/Users/yaelvinker/Documents/university/lab/Aug/08_27/models/", model_name,
                            "net_epoch_320.pth")
    model_params = get_model_params(model_name)
    f_factor_path = "none"
    output_images_path = os.path.join("/Users/yaelvinker/Documents/university/lab/Aug/08_27/models/", model_name,
                                      "exr_frame")
    input_images_path = os.path.join("/Users/yaelvinker/Documents/university/data/exr1")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    run_model_on_path(model_params, device, net_path, input_images_path,
                                      output_images_path, f_factor_path, None, input_images_path)


# ====== GET PARAMS ======
def get_model_params(model_name):
    model_params = {"model_name": model_name, "model": params.unet_network,
                    "filters": 32, "depth": 4,
                    "add_frame": get_frame(model_name),
                    # "add_frame": True,
                    "last_layer": get_last_layer(model_name),
                    "unet_norm": 'none',
                    "stretch_g": get_stretch_g(model_name),
                    "con_operator": get_con_operator(model_name),
                    "clip": get_clip(model_name),
                    "g_doubleConvTranspose": get_g_doubleConvTranspose(model_name),
                    "factor_coeff": get_factor_coeff(model_name),
                    "use_new_f": get_use_new_f(model_name),
                    "data_trc": get_data_trc(model_name),
                    "factorised_data": True,
                    "input_loader": None,
                    "gamma_log": 10,
                    "manualD": get_manualD(model_name),
                    "input_dim": 1,
                    "use_contrast_ratio_f": get_use_contrast_ratio_f(model_name)}
    if model_params["manualD"] != "none":
        model_params["input_dim"] = 2
    print(model_params)
    return model_params





def get_f_factor_path(name, data_gamma_log, use_new_f):
    path_dict_prev_f = {
        "test_source": {10: "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy"},
        "open_exr_exr_format": {1: "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_factors_mean150.npy",
                                2: "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_factors_mean90.npy",
                                10: "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_factors_mean160.npy"},
        "hdr_test": {10: "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy"},
        "npy_pth": {1: "/cs/snapless/raananf/yael_vinker/data/dng_data_fid.npy",
                    2: "/cs/snapless/raananf/yael_vinker/data/dng_data_fid.npy",
                    10: "/cs/snapless/raananf/yael_vinker/data/dng_data_fid.npy"}}
    # TODO : update these dirs when apply the new f on the test data
    path_dict_new = {"test_source": "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy",
                    "open_exr_exr_format": "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_newf.npy",
                    "hdr_test": "/Users/yaelvinker/PycharmProjects/lab/utils/test_factors.npy",
                    "npy_pth": "/cs/snapless/raananf/yael_vinker/data/dng_fid_newf.npy"}
    if use_new_f:
        return path_dict_new[name]
    if not data_gamma_log and name == "npy_pth":
        return path_dict_prev_f[name][1]
    return path_dict_prev_f[name][data_gamma_log]


def get_hdr_source_path(name):
    path_dict = {
        "test_source": "/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data",
        "new_source": "/Users/yaelvinker/PycharmProjects/lab/utils/temp_data",
        "open_exr_hdr_format": "/cs/snapless/raananf/yael_vinker/data/open_exr_source/open_exr_fixed_size",
        "open_exr_exr_format": "/cs/snapless/raananf/yael_vinker/data/open_exr_source/exr_format_fixed_size",
        "open_exr_exr_format_original_size": "/cs/labs/raananf/yael_vinker/data/quality_assesment/from_openEXR_data",
        "hdr_test": "/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr",
        "npy_pth": "/cs/snapless/raananf/yael_vinker/data/dng_data_fid"}
    return path_dict[name]


def get_con_operator(model_name):
    # con_op_short = {"original_unet": "ou",
    #                 "square": "s",
    #                 "square_root": "sr",
    #                 "square_and_square_root": "ssr",
    #                 "gamma": "g",
    #                 "square_and_square_root_manual_d": "ssrMD"}
    # return "square_and_square_root"
    if params.con_op_short["square_and_square_root_manual_d"] in model_name:
        return "square_and_square_root_manual_d"
    if params.con_op_short["square_and_square_root"] in model_name:
        return "square_and_square_root"
    if params.con_op_short["square_root"] in model_name:
        return "square_root"
    if params.con_op_short["original_unet"] in model_name:
        return "original_unet"
    if params.con_op_short["square"] in model_name:
        return "square"




def get_stretch_g(model_name):
    if "batchMax" in model_name:
        return "batchMax"
    elif "instanceMinMax" in model_name:
        return "instanceMinMax"
    return "none"

def get_last_layer(model_name):
    if "msig" in model_name:
        return "msig"
    else:
        return "sigmoid"

def get_frame(model_name):
    if "noframe" in model_name:
        return False
    return True

def get_clip(model_name):
    return "clip" in model_name

def get_factor_coeff(model_name):
    items = re.findall("DATA_\D*(\d+\.*\d+)", model_name)
    return float(items[0])

def get_data_trc(model_name):
    items = re.findall("DATA_(\w*)_\d+\.*\d+", model_name)
    return items[0]

def get_manualD(model_name):
    if "manualD_single" in model_name:
        return "single"
    if "manualD_double" in model_name:
        return "double"
    return "none"

def get_use_new_f(model_name):
    if "new_f" in model_name:
        return True
    return False

def get_use_contrast_ratio_f(model_name):
    if "contrast_ratio_f_" in model_name:
        return True
    return False

def get_g_doubleConvTranspose(model_name):
    if "doubleConvT" in model_name:
        return True
    return False

def get_models_names():
    models_names = [
        "add_alpha003_hdr_data_log_10_gamma_ssim_1.0_pyramid_1,2,4std_1.0_eps_0.001_6,8,1_mu_loss_2.0_1,1,1_unet_square_and_square_root_d_model_patchD"]
    return models_names


def get_apply_wind_norm(model_name):
    if "apply_wind_norm" in model_name:
        return True
    else:
        return False


def get_gamma_log(model_name):
    if "data_10" in model_name:
        return 10
    if "data_2" in model_name:
        return 2
    if "data_1" in model_name:
        return 1
    return 10


def get_std_norm_factor(model_name):
    if "apply_wind_norm" in model_name:
        if "factor_0.8" in model_name:
            return 0.8
        if "factor_0.9" in model_name:
            return 0.9
        if "factor_1" in model_name:
            return 1
    else:
        return None


def get_clip_from_name(model_name):
    if "clip_False" in model_name:
        return False
    if "clip_True" in model_name:
        return True
    else:
        assert 0, "Unsupported clip def"


def get_normalise_from_name(model_name):
    if "normalise_False" in model_name:
        return False
    if "normalise_True" in model_name:
        return True
    else:
        assert 0, "Unsupported normalise def"


def is_factorised_data(model_name):
    if "use_f_False" in model_name:
        return False
    if "use_f_True" in model_name:
        return True
    else:
        assert 0, "Unsupported factorised def"
