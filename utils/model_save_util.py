from utils import params
from models import Discriminator
import imageio
import numpy as np
import torch
import torch.nn as nn
import utils.image_quality_assessment_util as tmqi
import matplotlib.pyplot as plt
import utils.hdr_image_util as hdr_image_util
import utils.data_loader_util as data_loader_util
import tranforms
import models.unet_multi_filters.Unet as Generator
import sys
import inspect
import os
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


# ====== TRAIN RELATED ======
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
        # todo check the gain parameter
        # gain = np.sqrt(2.0)
        torch.nn.init.xavier_normal_(m.weight, gain = np.sqrt(2.0))
        if m.bias is not None:
            nn.init.constant(m.bias, 0)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.xavier_normal_(m.weight, gain = np.sqrt(2.0))
        if m.bias is not None:
            nn.init.constant(m.bias, 0)


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
                 add_frame, use_pyramid_loss, unet_norm, add_clipping, normalization, use_xaviar):
    # Create the Generator (UNet architecture)
    layer_factor = get_layer_factor(con_operator)
    if model == params.unet_network:
        new_net = Generator.UNet(input_dim_, input_dim_, last_layer, depth=unet_depth_,
                                 layer_factor=layer_factor,
                                 con_operator=con_operator, filters=filters, bilinear=False, network=model, dilation=0,
                                 to_crop=add_frame, use_pyramid_loss=use_pyramid_loss, unet_norm=unet_norm,
                                 add_clipping=add_clipping, normalization=normalization).to(device_)
    elif model == params.torus_network:
        new_net = Generator.UNet(input_dim_, input_dim_, last_layer, depth=unet_depth_,
                                 layer_factor=layer_factor,
                                 con_operator=con_operator, filters=filters, bilinear=False,
                                 network=params.torus_network, dilation=2, to_crop=add_frame,
                                 use_pyramid_loss=use_pyramid_loss, unet_norm=unet_norm,
                                 add_clipping=add_clipping, normalization=normalization).to(device_)
    else:
        assert 0, "Unsupported g model request: {}".format(model)

    return set_parallel_net(new_net, device_, is_checkpoint, "Generator", use_xaviar)


def create_D_net(input_dim_, down_dim, device_, is_checkpoint, norm, use_xaviar, d_model):
    # Create the Discriminator
    if d_model == "original":
        new_net = Discriminator.Discriminator(params.input_size, input_dim_, down_dim, norm).to(device_)
    elif d_model == "patchD":
        new_net = Discriminator.NLayerDiscriminator(input_dim_).to(device_)
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


# ====== TEST RELATED ======
def get_layer_factor(con_operator):
    if con_operator in params.layer_factor_2_operators:
        return 2
    elif con_operator in params.layer_factor_3_operators:
        return 3
    elif con_operator in params.layer_factor_4_operators:
        return 4
    else:
        assert 0, "Unsupported con_operator request: {}".format(con_operator)


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


def save_batch_images(fake_batch, hdr_origin_batch, output_path, im_name):
    new_batch = hdr_image_util.back_to_color_batch(hdr_origin_batch, fake_batch)
    for i in range(fake_batch.size(0)):
        ours_tone_map_numpy = hdr_image_util.to_0_1_range(new_batch[i].clone().permute(1, 2, 0).detach().cpu().numpy())
        im = (ours_tone_map_numpy * 255).astype('uint8')
        imageio.imwrite(os.path.join(output_path, im_name + "_" + str(i) + ".jpg"), im, format='JPEG-PIL')


# ====== USE TRAINED MODEL ======
def save_fake_images_for_fid_hdr_input(factor_coeff, input_format):
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    input_images_path = get_hdr_source_path("test_source")
    arch_dir = "/Users/yaelvinker/PycharmProjects/lab"
    models_names = get_models_names()
    models_epoch = [0]
    print(models_epoch)

    for i in range(len(models_names)):
        model_name = models_names[i]
        model_params = get_model_params(model_name)
        print("cur model = ", model_name)
        cur_model_path = os.path.join(arch_dir, model_name)
        if os.path.exists(cur_model_path):
            output_path = os.path.join(cur_model_path, input_format + "_format_factorised_" + str(factor_coeff))
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
                    run_model_on_path(model_params, device, cur_net_path, input_images_path, cur_output_path)
                else:
                    print("model path does not exists: ", cur_output_path)
        else:
            print("model path does not exists")


def run_model_on_path(model_params, device, cur_net_path, input_images_path, output_images_path):
    net_G = load_g_model(model_params, device, cur_net_path)
    print("model " + model_params["model"] + " was loaded successfully")
    for img_name in os.listdir(input_images_path):
        print(img_name)
        im_path = os.path.join(input_images_path, img_name)
        if os.path.splitext(img_name)[1] == ".hdr" or os.path.splitext(img_name)[1] == ".exr":
            original_im = hdr_image_util.read_hdr_image(im_path)
            print("original shape", original_im.shape)
            # f = 10760.730115410688
            # print(f)
            # original_im = original_im * 255 * f
            run_model_on_single_image(net_G, original_im, device, os.path.splitext(img_name)[0],
                                      output_images_path, model_params)


def load_g_model(model_params, device, net_path):
    G_net = get_trained_G_net(model_params["model"], device, 1,
                              model_params["last_layer"], model_params["filters"],
                              model_params["con_operator"], model_params["depth"],
                              True, True, model_params["unet_norm"], model_params["clip"])
    checkpoint = torch.load(net_path)
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
    G_net.eval()
    return G_net



def get_trained_G_net(model, device_, input_dim_, last_layer, filters, con_operator, unet_depth_,
                 add_frame, use_pyramid_loss, unet_norm, add_clipping):
    layer_factor = get_layer_factor(con_operator)
    if model == params.unet_network:
        new_net = Generator.UNet(input_dim_, input_dim_, last_layer, depth=unet_depth_,
                                 layer_factor=layer_factor,
                                 con_operator=con_operator, filters=filters, bilinear=False, network=model, dilation=0,
                                 to_crop=add_frame, use_pyramid_loss=use_pyramid_loss, unet_norm=unet_norm,
                                 add_clipping=add_clipping, normalization='max_normalization').to(device_)
    elif model == params.torus_network:
        new_net = Generator.UNet(input_dim_, input_dim_, last_layer, depth=unet_depth_,
                                 layer_factor=layer_factor,
                                 con_operator=con_operator, filters=filters, bilinear=False,
                                 network=params.torus_network, dilation=2, to_crop=add_frame,
                                 use_pyramid_loss=use_pyramid_loss, unet_norm=unet_norm,
                                 add_clipping=add_clipping, normalization='max_normalization').to(device_)
    else:
        assert 0, "Unsupported model request: {}  (creates only G or D)".format(model)

    if (device_.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        new_net = nn.DataParallel(new_net, list(range(torch.cuda.device_count())))
    return new_net


def run_model_on_single_image(G_net, original_im, device, im_name, output_path, model_params, addFrame=True):
    transform_exp = tranforms.Exp(1000, model_params["clip"], False, model_params["normalise"],
                                  model_params["factorised_data"])
    preprocessed_im = tmqi.apply_preproccess_for_hdr_im_factorised(original_im, addFrame=True).to(device)
    save_gray_tensor_as_numpy(preprocessed_im, output_path, im_name + "_input")

    preprocessed_im_batch = preprocessed_im.unsqueeze(0)
    exp_input = torch.squeeze(transform_exp(preprocessed_im_batch), dim=0)
    save_gray_tensor_as_numpy(exp_input, output_path, im_name + "_input_exp")

    with torch.no_grad():
        ours_tone_map_gray = G_net(preprocessed_im_batch.detach())
        ours_tone_map_gray = transform_exp(ours_tone_map_gray)
    fake_im_gray = torch.squeeze(ours_tone_map_gray, dim=0)
    save_gray_tensor_as_numpy(fake_im_gray, output_path, im_name)

    original_im_tensor = tranforms.hdr_im_transform(original_im).unsqueeze(0)
    color_batch = hdr_image_util.back_to_color_batch(original_im_tensor, ours_tone_map_gray)
    save_color_tensor_as_numpy(color_batch[0], output_path, im_name)


def save_fake_images_for_fid():
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    train_dataroot_npy = "/cs/labs/raananf/yael_vinker/fid/inception/fake_data_input_hdr_wraper"
    train_dataroot_ldr = "/cs/labs/raananf/yael_vinker/fid/inception/fake_data_input_hdr_wraper"
    output_path = "/cs/labs/raananf/yael_vinker/fid/inception/fake_data_results"
    model_path = "/cs/labs/raananf/yael_vinker/fid/inception/best_model/best_model.pth"
    run_model_for_path(device, train_dataroot_npy, train_dataroot_ldr, output_path,
                       model_path, 16)


# ====== GET PARAMS ======
def get_model_params(model_name):
    model_params = {"model_name": model_name, "model": params.unet_network, "filters": 32, "depth": 4,
                    "last_layer": 'none', "unet_norm": 'none', "con_operator": get_con_operator(model_name),
                    "clip": get_clip_from_name(model_name), "normalise": get_normalise_from_name(model_name),
                    "factorised_data": is_factorised_data(model_name)}
    return model_params


def get_hdr_source_path(name):
    path_dict = {"test_source": "/Users/yaelvinker/PycharmProjects/lab/utils/hdr_data",
        "open_exr_hdr_format": "/cs/snapless/raananf/yael_vinker/data/open_exr_source/open_exr_fixed_size"}
    return path_dict[name]


def get_con_operator(model_name):
    if params.original_unet in model_name:
        return params.original_unet
    if params.square_and_square_root in model_name:
        return params.square_and_square_root
    else:
        assert 0, "Unsupported con_operator"


def get_models_names():
    models_names = ["unet_square_and_square_root_last_act_none_norm_g_none_use_f_True_coeff_0.1_clip_False_normalise_False"]
    return models_names


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


# ====== SAVE IMAGES ======
def save_gray_tensor_as_numpy(tensor, output_path, im_name):
    tensor = tensor.clone().permute(1, 2, 0).detach().cpu().numpy()
    tensor_0_1 = np.squeeze(hdr_image_util.to_0_1_range(tensor))
    im = (tensor_0_1 * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".png"), im, format='PNG-FI')


def save_color_tensor_as_numpy(tensor, output_path, im_name):
    tensor = hdr_image_util.to_0_1_range(tensor.clone().permute(1, 2, 0).detach().cpu().numpy())
    im = (tensor * 255).astype('uint8')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    imageio.imwrite(os.path.join(output_path, im_name + ".jpg"), im, format='JPEG-PIL')


# ====== DISCRIMINATOR TRAINED MODEL ======
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
    save_fake_images_for_fid_hdr_input(0.1, "hdr")

    # hdr_path = "/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/3.hdr"
    # im_hdr_original = hdr_image_util.read_hdr_image(hdr_path)
    # load_model(im_hdr_original)
