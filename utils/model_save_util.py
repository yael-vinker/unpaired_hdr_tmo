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
                                  input_format, f_factor_path, net_G=None, use_new_f=False)
            else:
                print("model %s does not exists: " % cur_net_path)
        else:
            print("model %s does not exists" % cur_model_path)


def run_model_on_path(model_params, device, cur_net_path, input_images_path, output_images_path, input_format,
                      f_factor_path, net_G, use_new_f):
    if not net_G:
        net_G = load_g_model(model_params, device, cur_net_path)
    print("model " + model_params["model"] + " was loaded successfully")
    for img_name in os.listdir(input_images_path):
        im_path = os.path.join(input_images_path, img_name)
        if not os.path.exists(os.path.join(output_images_path, os.path.splitext(img_name)[0] + ".png")):
            print("working on ", img_name)
            if os.path.splitext(img_name)[1] == ".hdr" or os.path.splitext(img_name)[1] == ".exr" \
                    or os.path.splitext(img_name)[1] == ".dng":
                run_model_on_single_image(net_G, im_path, device, os.path.splitext(img_name)[0],
                                          output_images_path, model_params, f_factor_path, use_new_f)
        else:
            print("%s in already exists" % (img_name))


def load_g_model(model_params, device, net_path):
    G_net = get_trained_G_net(model_params["model"], device, 1,
                              model_params["last_layer"], model_params["filters"],
                              model_params["con_operator"], model_params["depth"],
                              add_frame=True, use_pyramid_loss=True, unet_norm=model_params["unet_norm"],
                              add_clipping=model_params["clip"])

    checkpoint = torch.load(net_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['modelG_state_dict']
    print("from loaded model ", list(state_dict.keys())[0])
    if "module" in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
    # else:
    # new_state_dict = state_dict
    G_net.load_state_dict(new_state_dict)
    G_net.eval()
    return G_net


def get_trained_G_net(model, device_, input_dim_, last_layer, filters, con_operator, unet_depth_,
                      add_frame, use_pyramid_loss, unet_norm, add_clipping, output_dim=1, activation="relu"):
    layer_factor = get_layer_factor(con_operator)
    if model != params.unet_network:
        assert 0, "Unsupported g model request: {}".format(model)
    new_net = Generator.UNet(input_dim_, output_dim, last_layer, depth=unet_depth_,
                             layer_factor=layer_factor,
                             con_operator=con_operator, filters=filters, bilinear=False, network=model, dilation=0,
                             to_crop=add_frame, unet_norm=unet_norm,
                             add_clipping=add_clipping, activation=activation, apply_exp=False).to(device_)
    if (device_.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        new_net = nn.DataParallel(new_net, list(range(torch.cuda.device_count())))
    return new_net


def run_model_on_single_image(G_net, im_path, device, im_name, output_path, model_params, f_factor_path, use_new_f):
    rgb_img, gray_im_log, f_factor = create_dng_npy_data.hdr_preprocess(im_path,
                                                                        factor_coeff=1.0,
                                                                        train_reshape=False,
                                                                        gamma_log=model_params["gamma_log"],
                                                                        f_factor_path=f_factor_path,
                                                                        use_new_f=use_new_f)
    rgb_img, gray_im_log = tranforms.hdr_im_transform(rgb_img), tranforms.hdr_im_transform(gray_im_log)

    gray_im_log = data_loader_util.add_frame_to_im(gray_im_log)
    gray_im_log = gray_im_log.to(device)
    preprocessed_im_batch = gray_im_log.unsqueeze(0)
    with torch.no_grad():
        ours_tone_map_gray = G_net(preprocessed_im_batch.detach())
    fake_im_gray = torch.squeeze(ours_tone_map_gray, dim=0)

    original_im_tensor = rgb_img.unsqueeze(0)
    file_name = im_name + "_no_stretch"
    color_batch = hdr_image_util.back_to_color_batch(original_im_tensor, ours_tone_map_gray)
    hdr_image_util.save_color_tensor_as_numpy(color_batch[0], output_path, file_name)
    file_name = im_name + "_gray_no_stretch"
    hdr_image_util.save_color_tensor_as_numpy(fake_im_gray, output_path, file_name)

    file_name = im_name + "gray_stretch"
    fake_im_gray_stretch = (fake_im_gray - fake_im_gray.min()) / (fake_im_gray.max() - fake_im_gray.min())
    hdr_image_util.save_gray_tensor_as_numpy(fake_im_gray_stretch, output_path, file_name)

    file_name = im_name + "_stretch"
    color_batch_stretch = hdr_image_util.back_to_color_batch(original_im_tensor, fake_im_gray_stretch.unsqueeze(dim=0))
    hdr_image_util.save_color_tensor_as_numpy(color_batch_stretch[0], output_path, file_name)


# ====== GET PARAMS ======
def get_model_params(model_name):
    model_params = {"model_name": model_name, "model": params.unet_network, "filters": 32, "depth": 4,
                    "last_layer": get_last_layer(model_name), "unet_norm": 'none',
                    "con_operator": get_con_operator(model_name),
                    "clip": get_clip(model_name),
                    "factorised_data": True,
                    "input_loader": None,
                    "gamma_log": get_gamma_log(model_name)}
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
        "hdr_test": "/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr",
        "npy_pth": "/cs/snapless/raananf/yael_vinker/data/dng_data_fid"}
    return path_dict[name]


def get_con_operator(model_name):
    if params.original_unet in model_name:
        return params.original_unet
    if params.square_and_square_root in model_name:
        return params.square_and_square_root
    else:
        return params.original_unet


def get_last_layer(model_name):
    if "msig" in model_name:
        return "msig"
    else:
        return "sigmoid"


def get_clip(model_name):
    return "clip" in model_name


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


if __name__ == '__main__':
    output_dir = "d32_no_frame_g1e-05_d1e-05_decay50.0_stretch_1.05data10_d5.0_gamma_ssim1.0_2,4,4__unet_square_and_square_root_act_relu_d_original_nlayers4_sigmoid"
    net_path = os.path.join("/Users/yaelvinker/Documents/university/lab/Aug/summary_08_04/d32_no_frame_g1e-05_d1e-05_decay50.0_stretch_1.05data10_d5.0_gamma_ssim1.0_2,4,4__unet_square_and_square_root_act_relu_d_original_nlayers4_sigmoid/net_epoch_320.pth")
    model_params = get_model_params(output_dir)
    # f_factor_path = os.path.join("/Users/yaelvinker/Documents/university/lab/July/baseline/stretch_1.05data10_d1.0_gamma_ssim2.0_1,2,3_gamma_factor_loss_bilateral1.0_8,4,1_wind5_bmu1.0_sigr0.07_log0.8_eps1e-05_alpha0.5_mu_loss2.0_1,1,1_unet_square_and_square_root_d_model_patchD/test_factors.npy")
    f_factor_path = "none"
    output_images_path = os.path.join("/Users/yaelvinker/Documents/university/lab/Aug/summary_08_04/"
                                      "d32_no_frame_g1e-05_d1e-05_decay50.0_stretch_1.05data10_d5.0_gamma_ssim1.0_2,4,4__unet_square_and_square_root_act_relu_d_original_nlayers4_sigmoid/output_f_play/")
    input_images_path = os.path.join("/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    run_model_on_path(model_params, device, net_path, input_images_path,
                      output_images_path, "npy", f_factor_path, None, False)

    # 160986.84587859685

    # parser = argparse.ArgumentParser(description="Parser for gan network")
    # parser.add_argument("--input_format", type=str, default="npy")
    # parser.add_argument("--images_source", type=str, default="npy_pth")
    # parser.add_argument("--arch_dir", type=str, default="/cs/labs/raananf/yael_vinker/04_26/results_09_24")
    # parser.add_argument("--models_epoch", type=int, default=320)
    # parser.add_argument("--output_dir_name", type=str, default="exr_320")
    # args = parser.parse_args()
    # device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    # models_names = [
    #     "data_10_gamma_ssim_1.0_pyramid_2,2,6gamma_factor_loss_bilateral__b_sigma_r_0.071.0_eps_1e-05_6,4,1_alpha_1.0_mu_loss_1.0_1,1,3_unet_square_and_square_root_d_model_patchD"]
    # #    /cs/labs/raananf/yael_vinker/04_26/results_29_04/expdata_log_10std_5.0_eps_0.001_2,4,6_mu_loss_3.0_1,1,1_struct_factor_1.0_2,4,6_unet_square_and_square_root_d_model_patchD/
    #
    # apply_models_from_arch_dir(args.input_format, device, args.images_source, args.arch_dir,
    #                            models_names, args.models_epoch, args.output_dir_name)
    # hdr_path = "/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/3.hdr"
    # im_hdr_original = hdr_image_util.read_hdr_image(hdr_path)
    # load_model(im_hdr_original)
