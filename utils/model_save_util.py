import inspect
import os
import sys

import imageio
import numpy as np
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import params
import torch
import torch.nn as nn
import utils.image_quality_assessment_util as tmqi
import matplotlib.pyplot as plt
import utils.hdr_image_util as hdr_image_util
import utils.data_loader_util as data_loader_util
# import three_layers_unet.Unet as three_Unet
import tranforms
import models.unet_multi_filters.Unet as Generator
from models import Discriminator

# TODO ask about init BatchNorm weights
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
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)


def get_layer_factor(con_operator):
    if con_operator in params.layer_factor_2_operators:
        return 2
    elif con_operator in params.layer_factor_3_operators:
        return 3
    elif con_operator in params.layer_factor_4_operators:
        return 4
    else:
        assert 0, "Unsupported con_operator request: {}".format(con_operator)


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


def create_D_net(input_dim_, down_dim, device_, is_checkpoint, norm, use_xaviar=False):
    # Create the Discriminator
    new_net = Discriminator.Discriminator(params.input_size, input_dim_, down_dim, norm).to(device_)
    return set_parallel_net(new_net, device_, is_checkpoint, "Discriminator", use_xaviar)


def create_G_net(model, device_, is_checkpoint, input_dim_, last_layer, filters, con_operator, unet_depth_,
                 add_frame, use_pyramid_loss, unet_norm, use_xaviar=False):
    # Create the Generator (UNet architecture)
    layer_factor = get_layer_factor(con_operator)
    if model == params.unet_network:
        new_net = Generator.UNet(input_dim_, input_dim_, last_layer, depth=unet_depth_,
                                 layer_factor=layer_factor,
                                 con_operator=con_operator, filters=filters, bilinear=False, network=model, dilation=0,
                                 to_crop=add_frame, use_pyramid_loss=use_pyramid_loss, unet_norm=unet_norm).to(device_)
    elif model == params.torus_network:
        new_net = Generator.UNet(input_dim_, input_dim_, last_layer, depth=unet_depth_,
                                 layer_factor=layer_factor,
                                 con_operator=con_operator, filters=filters, bilinear=False,
                                 network=params.torus_network, dilation=2, to_crop=add_frame,
                                 use_pyramid_loss=use_pyramid_loss, unet_norm=unet_norm).to(device_)
    else:
        assert 0, "Unsupported model request: {}  (creates only G or D)".format(model)

    return set_parallel_net(new_net, device_, is_checkpoint, "Generator", use_xaviar)


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

    if epoch == 50:
        models_250_save_path = os.path.join("models_250", "models_250_net.pth")
        path_250 = os.path.join(output_dir, models_250_save_path)
        torch.save({
            'epoch': epoch,
            'modelD_state_dict': netD.state_dict(),
            'modelG_state_dict': netG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
        }, path_250)


def save_discriminator_model(path, epoch, output_dir, netD, optimizerD):
    path = os.path.join(output_dir, path, "net_epoch_" + str(epoch) + ".pth")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'modelD_state_dict': netD.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }, path)

    if epoch == 50:
        models_250_save_path = os.path.join("models_250", "models_250_net.pth")
        path_250 = os.path.join(output_dir, models_250_save_path)
        torch.save({
            'epoch': epoch,
            'modelD_state_dict': netD.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
        }, path_250)


def save_best_model(netG, output_dir, optimizerG):
    best_model_save_path = os.path.join("best_model", "best_model.pth")
    best_model_path = os.path.join(output_dir, best_model_save_path)
    torch.save({
        'modelG_state_dict': netG.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
    }, best_model_path)


def load_g_model(model, device, filters, con_operator, model_depth, net_path, pyramid_loss, last_layer, unet_norm):
    G_net = create_G_net(model, device, False, 1, last_layer,filters, con_operator, model_depth, 1, pyramid_loss , unet_norm)
    #checkpoint = torch.load(net_path, map_location=torch.device('cpu'))
    checkpoint = torch.load(net_path)
    state_dict = checkpoint['modelG_state_dict']
    # if device.type == 'cpu':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # else:
    #new_state_dict = state_dict
    G_net.load_state_dict(new_state_dict)
    G_net.eval()
    return G_net


def load_d_model(model, device, filters, con_operator, model_depth,
                 net_path="/Users/yaelvinker/PycharmProjects/lab/local_log_100_skip_connection_conv_depth_1/best_model/best_model.pth"):
    D_net = create_net("D", model, device, False, 1, 0, filters,
                       con_operator, model_depth).to(device)
    # checkpoint = torch.load(net_path, map_location=torch.device('cpu'))
    checkpoint = torch.load(net_path)
    state_dict = checkpoint['modelD_state_dict']
    # if device.type == 'cpu':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # else:
    new_state_dict = state_dict
    D_net.load_state_dict(new_state_dict)
    D_net.eval()
    return D_net


def run_model_on_single_image(G_net, original_im, device, im_name, output_path, addFrame=False, add_clipping=False,
                              apply_inverse_to_preprocess=False, normalised_data=True):
    transform_exp = tranforms.Exp(1000, add_clipping, apply_inverse_to_preprocess, normalised_data)
    b_factor_output_dir = os.path.join(
        "/cs/snapless/raananf/yael_vinker/data/open_exr_source/openexr_factors/brightness_factors.npy")
    data = np.load(b_factor_output_dir, allow_pickle=True)[()]
    preprocessed_im = tmqi.apply_preproccess_for_hdr_im_factorised(original_im, data, im_name, addFrame=True).to(device)
    #
    # preprocessed_im = tmqi.apply_preproccess_for_hdr_im(original_im).to(device)
    preprocessed_im_batch = preprocessed_im.unsqueeze(0)
    with torch.no_grad():
        ours_tone_map_gray = G_net(preprocessed_im_batch.detach())
        ours_tone_map_gray = transform_exp(ours_tone_map_gray)
        if addFrame:
            b, c, h, w = ours_tone_map_gray.shape
            th, tw = h - 2 * params.shape_addition, w - 2 * params.shape_addition
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            i, j, h, w = i, j, th, tw
            ours_tone_map_gray = ours_tone_map_gray[:, :, i: i + h, j:j + w]
    fake_im_gray = torch.squeeze(ours_tone_map_gray, dim=0)
    fake_im_gray_numpy = fake_im_gray.clone().permute(1, 2, 0).detach().cpu().numpy()
    fake_im_gray_numpy_0_1 = np.squeeze(hdr_image_util.to_0_1_range(fake_im_gray_numpy))
    # fake_im_gray_numpy_0_1 = fake_im_gray_numpy_0_1 * 1.1 - 0.05
    im = (fake_im_gray_numpy_0_1 * 255).astype('uint8')
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')

    original_im_tensor = tranforms.hdr_im_transform(original_im).unsqueeze(0)
    color_batch = hdr_image_util.back_to_color_batch(original_im_tensor, ours_tone_map_gray)
    ours_tone_map_numpy = hdr_image_util.to_0_1_range(color_batch[0].clone().permute(1, 2, 0).detach().cpu().numpy())
    im = (ours_tone_map_numpy * 255).astype('uint8')
    # ours_tone_map_numpy = (ours_tone_map_numpy * 1.1 - 0.05)
    # im = (ours_tone_map_numpy * 255)
    # im = np.clip(im, 0, 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".jpg"), im, format='JPEG-PIL')


def save_batch_images(fake_batch, hdr_origin_batch, output_path, im_name):
    new_batch = hdr_image_util.back_to_color_batch(hdr_origin_batch, fake_batch)
    for i in range(fake_batch.size(0)):
        ours_tone_map_numpy = hdr_image_util.to_0_1_range(new_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy())
        im = (ours_tone_map_numpy * 255).astype('uint8')
        imageio.imwrite(os.path.join(output_path, im_name + "_" + str(i) + ".jpg"), im, format='JPEG-PIL')


def run_model_for_path(device, train_dataroot_npy, train_dataroot_ldr, output_path, model_path, batch_size=4):
    train_data_loader_npy, train_ldr_loader = data_loader_util.load_data(train_dataroot_npy, train_dataroot_ldr,
                                                                         batch_size, testMode=False, title="train")

    G_net = load_g_model(device, model_path)
    num_iters = 0
    for data_hdr_batch in train_data_loader_npy:
        hdr_input = data_hdr_batch[params.gray_input_image_key].to(device)
        hdr_input_display = data_hdr_batch[params.color_image_key].to(device)
        with torch.no_grad():
            ours_tone_map_gray = G_net(hdr_input.detach())
            save_batch_images(ours_tone_map_gray, hdr_input_display, output_path, str(num_iters))
        num_iters += 1


def save_fake_images_for_fid():
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    train_dataroot_npy = "/cs/labs/raananf/yael_vinker/fid/inception/fake_data_input_hdr_wraper"
    train_dataroot_ldr = "/cs/labs/raananf/yael_vinker/fid/inception/fake_data_input_hdr_wraper"
    output_path = "/cs/labs/raananf/yael_vinker/fid/inception/fake_data_results"
    model_path = "/cs/labs/raananf/yael_vinker/fid/inception/best_model/best_model.pth"
    run_model_for_path(device, train_dataroot_npy, train_dataroot_ldr, output_path,
                       model_path, 16)


def save_fake_images_for_fid_hdr_input():
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    # device = torch.device("cpu")
    # input_images_path = "/cs/labs/raananf/yael_vinker/data/quality_assesment/hdr_openEXR_no_reshape"
    input_images_path = "/cs/snapless/raananf/yael_vinker/data/open_exr_source/open_exr_fixed_size"
    # input_images_path = "/cs/snapless/raananf/yael_vinker/data/open_exr_source/temp"
    # arch_dir = "/cs/snapless/raananf/yael_vinker/01_30/results"
    arch_dir = "/cs/snapless/raananf/yael_vinker/02_04/results"
    # arch_dir = "/cs/labs/raananf/yael_vinker/01_10/results_custom_ssim"
    filters = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    models = [params.unet_network,
              params.unet_network,
              params.unet_network,
              params.unet_network,
              params.unet_network,
              params.unet_network,
              params.unet_network,
              params.unet_network,
              params.unet_network,
              params.unet_network]
    con_operators = [params.original_unet,
                     params.square_and_square_root,
                     params.original_unet,
                     params.square_and_square_root,
                     params.square_and_square_root,
                     params.square_and_square_root]
    depths = [4, 4, 4, 3, 3, 4, 4, 4, 4, 4]
    # models_names = ["with_frame_32_filters_custom_ssim_window_5__log_1000_unet_square_and_square_root_depth_4"]
    # models_names = ["with_frame_32_filters_custom_ssim_window_5__log_1000_unet_square_and_square_root_depth_4"]
    # models_names = ["d_dim_32_instance_norm_pyramid_small_lr_factor01__G_unet_square_and_square_root_depth_4_filters_32_frame_1",
    #                 "d_inst_n_dim_16_inst_n_pyramid_small_lr_factor01__G_unet_square_and_square_root_depth_4_filters_32_frame_1",
    #                 "instance_norm_pyramid_small_lr_factor01__G_unet_original_unet_depth_4_filters_32_frame_1",
    #                 "d_dim_16_instance_norm_pyramid_small_lr_factor01__G_unet_original_unet_depth_4_filters_32_frame_1",
    #                 "d_inst_n_dim_16_inst_n_pyramid_small_lr_factor01__G_unet_square_and_square_root_depth_4_filters_32_frame_1"]
    # models_names = ["no_norm_pyramid_G_unet_square_and_square_root_depth_4_filters_32_frame_1",
    #       "no_norm_G_unet_square_and_square_root_depth_4_filters_32_frame_1"]
    models_names = ["new_no_tanh_clip_exp__G_unet_original_unet_depth_4_last_act_none"]
    # "eq_half_lr_lr_decay_G_unet_original_unet_depth_4_filters_32_frame_1",
    #       "half_lr_lr_decay_G_unet_original_unet_depth_4_filters_32_frame_1",
    # "half_lr_lr_decay_G_unet_original_unet_depth_4_filters_32_frame_1"]
    #      "lr_decay_G_unet_original_unet_depth_4_filters_32_frame_1",
    #     "new_data_origin_lr_ssim5pyramid_lr_decay_G_unet_square_and_square_root_depth_3_filters_32_frame_1",
    #    "origin_lr_ssim5pyramid_lr_decay_G_unet_square_and_square_root_depth_3_filters_32_frame_1",
    #   "origin_lr_ssim5pyramid_lr_decay_G_unet_square_and_square_root_depth_4_filters_32_frame_1",
    #  "pyramid_eq_half_lr_lr_decay_G_unet_original_unet_depth_4_filters_32_frame_1",
    #  "pyramid_half_lr_lr_decay_G_unet_original_unet_depth_4_filters_32_frame_1"]
    # "pyramid_half_lr_lr_decay_G_unet_square_and_square_root_depth_4_filters_32_frame_1",
    # "ssim5pyramid_lr_decay_G_unet_square_and_square_root_depth_4_filters_32_frame_1"]

    # filters = [32, 32, 32, 32,
    #            32, 32, 32, 32,
    #            32, 32, 64]
    # models = [params.torus_network, params.torus_network, params.torus_network, params.unet_network,
    #          params.unet_network, params.unet_network, params.unet_network, params.unet_network,
    #          params.unet_network, params.unet_network, params.unet_network]
    # con_operators = [params.original_unet, params.original_unet, params.square, params.original_unet,
    #                 params.original_unet, params.square_and_square_root, params.square_and_square_root, params.square,
    #                 params.square, params.square_root, params.original_unet]
    # depths = [3, 4, 3, 3,
    #           4, 3, 4, 3,
    #           4, 3, 3]

    # models_epoch = range(100, 300, 5)
    models_epoch = [135]
    print(models_epoch)

    for i in range(len(models_names)):
        # model_name = str(filters[i]) + "_filters__log_1000_" + models[i] + "_" + con_operators[i] + "_depth_" + str(depths[i])
        model_name = models_names[i]
        print("cur model = ", model_name)
        cur_model_path = os.path.join(arch_dir, model_name)
        if os.path.exists(cur_model_path):
            output_path = os.path.join(cur_model_path, "hdr_format_factorised_last_im_1")
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            net_path = os.path.join(cur_model_path, "models")
            for m in models_epoch:
                net_name = "net_epoch_" + str(m) + ".pth"
                cur_net_path = os.path.join(net_path, net_name)
                if os.path.exists(cur_net_path):
                    cur_output_path = os.path.join(output_path, str(m))
                    if not os.path.exists(cur_output_path):
                        os.mkdir(cur_output_path)
                    run_model_on_path(models[i], device, filters[i], con_operators[i], depths[i], cur_net_path,
                                      input_images_path, cur_output_path, last_layer='none', unet_norm='instance_norm')
                else:
                    print("model path does not exists: ", cur_output_path)
        else:
            print("model path does not exists")


def get_bump(im):
    import numpy as np
    tmp = im
    tmp[(tmp > 200) != 0] = 255
    tmp = tmp.astype('uint8')

    hist, bins = np.histogram(tmp, bins=255)

    a0 = np.mean(hist[0:64])
    a1 = np.mean(hist[65:200])
    return a1 / a0


def find_f(im):
    import numpy as np
    im = im / np.max(im) * 255
    big = 1.1
    f = 1.0

    for i in range(1000):
        print(i)
        print(f)
        r = get_bump(im * f)
        print(r)
        if r < big:
            f = f * 1.01
        else:
            if r > 1 / big:
                break
    return f


def run_model_on_path(model, device, filters, con_operator, model_depth, net_path, input_images_path,
                      output_images_path, last_layer, unet_norm):
    net_G = load_g_model(model, device, filters, con_operator, model_depth, net_path, 0, last_layer, unet_norm)
    print("model " + model + " was loaded successfully")
    for img_name in os.listdir(input_images_path):
        print(img_name)
        im_path = os.path.join(input_images_path, img_name)
        if os.path.splitext(img_name)[1] == ".hdr" or os.path.splitext(img_name)[1] == ".exr":
            original_im = hdr_image_util.read_hdr_image(im_path)
            # f = 10760.730115410688
            # print(f)
            # original_im = original_im * 255 * f
            run_model_on_single_image(net_G, original_im, device, os.path.splitext(img_name)[0], output_images_path)


def run_model_d_on_path(model, device, filters, con_operator, model_depth, net_path, train_data_loader_hdr,
                        train_data_loader_ldr, output_images_path):
    net_D = load_d_model(model, device, filters, con_operator, model_depth, net_path)
    print("model " + model + " was loaded successfully")

    accDreal_counter = 0
    num_iter = 0
    for data_ldr in train_data_loader_ldr:
        num_iter += 1
        real_ldr = data_ldr[params.gray_input_image_key].to(device)
        output_on_real = net_D(real_ldr).view(-1)
        # Real label = 1, so we count the samples on which D was right
        accDreal_counter += (output_on_real > 0.5).sum().item()
    accDlog_counter = 0
    num_iter = 0
    for data_hdr in train_data_loader_hdr:
        num_iter += 1
        hdr_input = data_hdr[params.gray_input_image_key].to(device)
        output_on_log = net_D(hdr_input).view(-1)
        # Real label = 1, so we count the samples on which D was right
        accDlog_counter += (output_on_log <= 0.5).sum().item()
    return accDreal_counter / (num_iter * 16), accDlog_counter / (num_iter * 16)


def compare_best_models():
    models_name = ["skip_connection"]
    models_path = [
        "/cs/labs/raananf/yael_vinker/11_20/results/log1000_with_exp_log_1000_skip_connection_conv_depth_4/best_model/best_model.pth"]
    models_depth = [4]
    input_images_path = "/cs/labs/raananf/yael_vinker/models_results/hdr_test_images"
    output_path = "/cs/labs/raananf/yael_vinker/models_results/models_results"
    for i in range(len(models_name)):
        model_path = models_path[i]
        model_name = models_name[i]
        model_depth = models_depth[i]
        model_output_path = os.path.join(output_path, model_name)
        print(model_name)
        run_model_on_path(model_path, model_name, model_depth, input_images_path, model_output_path)


def run_discriminator_on_data():
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    # device = torch.device("cpu")
    input_hdr_images_path = "/cs/labs/raananf/yael_vinker/data/train/hdrplus_dict_log1000"
    input_ldr_images_path = "/cs/labs/raananf/yael_vinker/data/train/ldr_flicker_dict"
    train_data_loader_hdr, train_data_loader_ldr = \
        data_loader_util.load_data(input_hdr_images_path, input_ldr_images_path,
                                   16, testMode=False, title="train")
    arch_dir = "/cs/labs/raananf/yael_vinker/12_25/run/results/"

    filters = [32, 32, 32, 32,
               32, 32, 32, 32,
               32, 32, 64]
    models = [params.torus_network, params.torus_network, params.torus_network, params.unet_network,
              params.unet_network, params.unet_network, params.unet_network, params.unet_network,
              params.unet_network, params.unet_network, params.unet_network]
    con_operators = [params.original_unet, params.original_unet, params.square, params.original_unet,
                     params.original_unet, params.square_and_square_root, params.square_and_square_root, params.square,
                     params.square, params.square_root, params.original_unet]
    depths = [3, 4, 3, 3,
              4, 3, 4, 3,
              4, 3, 3]

    models_epoch = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    # models_epoch = [0]

    # filters = [32]
    # models = [params.unet_network]
    # con_operators = [params.square]
    # depths = [3]

    acc_ldr_list = []
    acc_log_list = []
    for i in range(len(filters)):
        model_name = str(filters[i]) + "_filters__log_1000_" + models[i] + "_" + con_operators[i] + "_depth_" + str(
            depths[i])
        print("cur model = ", model_name)
        cur_model_path = os.path.join(arch_dir, model_name)
        if os.path.exists(cur_model_path):
            output_path = os.path.join(cur_model_path, "evaluate_d")
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            net_path = os.path.join(cur_model_path, "models")
            for m in models_epoch:
                net_name = "net_epoch_" + str(m) + ".pth"
                cur_net_path = os.path.join(net_path, net_name)
                if os.path.exists(cur_net_path):
                    cur_output_path = os.path.join(output_path, str(m))
                    if not os.path.exists(cur_output_path):
                        os.mkdir(cur_output_path)
                    acc_ldr, acc_log = run_model_d_on_path(models[i], device, filters[i], con_operators[i], depths[i],
                                                           cur_net_path,
                                                           train_data_loader_hdr,
                                                           train_data_loader_ldr, cur_output_path)
                    acc_ldr_list.append(acc_ldr)
                    acc_log_list.append(acc_log)
                else:
                    print("model path does not exists: ", cur_net_path)

            plt.figure()
            plt.plot(models_epoch, acc_ldr_list, 'o', label='acc D LDR')
            plt.plot(models_epoch, acc_log_list, 'o', label='acc D LOG')
            # plt.plot(range(iters_n), acc_G, '-g', label='acc G')

            plt.xlabel(models_epoch)
            plt.legend(loc='upper left')
            plt.title(model_name)

            # save image
            plt.savefig(os.path.join(output_path, "evaluate_d.png"))  # should before show method
            plt.close()
        else:
            print("model path does not exists")


if __name__ == '__main__':
    # save_fake_images_for_fid_hdr_input()
    save_fake_images_for_fid_hdr_input()

    # hdr_path = "/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/S0020.hdr"
    # im_hdr_original = hdr_image_util.read_hdr_image(hdr_path)
    # load_model(im_hdr_original)
