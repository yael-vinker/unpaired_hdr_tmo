import os
import sys
# Add the ptdraft folder path to the sys.path list
import skimage

sys.path.append('/cs/labs/raananf/yael_vinker/11_03/lab')
import imageio
import torch
import params
import torch
import torus.Unet as TorusUnet
import torch.nn as nn
import unet.Unet as Unet
import utils.image_quality_assessment_util as tmqi
import matplotlib.pyplot as plt
import utils.hdr_image_util as hdr_image_util
import utils.data_loader_util as data_loader_util
import three_layers_unet.Unet as three_Unet
import tranforms


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

def save_best_model(netG, output_dir, optimizerG):
    best_model_save_path = os.path.join("best_model", "best_model.pth")
    best_model_path = os.path.join(output_dir, best_model_save_path)
    torch.save({
        'modelG_state_dict': netG.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
    }, best_model_path)

def load_g_model(device, model_name, model_depth, net_path="/Users/yaelvinker/PycharmProjects/lab/local_log_100_skip_connection_conv_depth_1/best_model/best_model.pth"):
    if model_name == "skip_connection":
        G_net = Unet.UNet(1, 1, 0, bilinear=False, depth=model_depth).to(device)
    elif model_name == "torus":
        G_net = TorusUnet.UNet(1, 1, 0, bilinear=False, depth=model_depth).to(device)
    elif model_name == "unet3_layer":
        G_net = three_Unet.UNet(1, 1, 0, bilinear=False).to(device)
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Using [%d] GPUs" % torch.cuda.device_count())
        G_net = nn.DataParallel(G_net, list(range(torch.cuda.device_count())))
    checkpoint = torch.load(net_path)
    state_dict = checkpoint['modelG_state_dict']
    # G_net.load_state_dict(checkpoint['modelG_state_dict'])

    # if device.type == 'cpu':
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

def run_model_on_single_image(G_net, original_im, device, im_name, output_path):
    preprocessed_im = tmqi.apply_preproccess_for_hdr_im(original_im).to(device)
    preprocessed_im_batch = preprocessed_im.unsqueeze(0)
    with torch.no_grad():
        ours_tone_map_gray = G_net(preprocessed_im_batch.detach())
    original_im_tensor = tranforms.hdr_im_transform(original_im).unsqueeze(0)
    color_batch = hdr_image_util.back_to_color_batch(original_im_tensor, ours_tone_map_gray)
    ours_tone_map_numpy = hdr_image_util.to_0_1_range(color_batch[0].clone().permute(1, 2, 0).detach().cpu().numpy())
    im = (ours_tone_map_numpy * 255).astype('uint8')
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



def run_model_on_path(model_path, model_name, model_depth, input_images_path, output_images_path):
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    net_G = load_g_model(device, model_name, model_depth, model_path)
    print("model " + model_name + " was loaded successfully")
    for img_name in os.listdir(input_images_path):
        im_path = os.path.join(input_images_path, img_name)
        original_im = hdr_image_util.reshape_image(hdr_image_util.read_hdr_image(im_path))
        run_model_on_single_image(net_G, original_im, device, img_name, output_images_path)

def compare_best_models():
    models_name = ["skip_connection"]
    models_path = ["/cs/labs/raananf/yael_vinker/11_20/results/log1000_with_exp_log_1000_skip_connection_conv_depth_4/best_model/best_model.pth"]
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


if __name__ == '__main__':
    compare_best_models()


    # hdr_path = "/Users/yaelvinker/PycharmProjects/lab/data/hdr_data/hdr_data/S0020.hdr"
    # im_hdr_original = hdr_image_util.read_hdr_image(hdr_path)
    # load_model(im_hdr_original)