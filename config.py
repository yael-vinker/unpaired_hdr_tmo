import argparse
import os

import torch

import gan_trainer_utils as g_t_utils
import params


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=params.num_epochs)
    parser.add_argument("--model", type=str, default=params.unet_network)  # up sampling is the default
    parser.add_argument("--con_operator", type=str, default=params.original_unet)
    parser.add_argument("--filters", type=int, default=params.filters)
    parser.add_argument("--unet_depth", type=int, default=4)
    parser.add_argument("--add_frame", type=int, default=1)  # int(False) = 0
    parser.add_argument("--G_lr", type=float, default=params.lr)
    parser.add_argument("--D_lr", type=float, default=params.lr)
    parser.add_argument("--data_root_npy", type=str, default=params.train_dataroot_hdr)
    parser.add_argument("--data_root_ldr", type=str, default=params.train_dataroot_ldr)
    parser.add_argument("--checkpoint", type=str, default="no")
    parser.add_argument("--test_dataroot_npy", type=str, default=params.test_dataroot_hdr)
    parser.add_argument("--test_dataroot_original_hdr", type=str, default=params.test_dataroot_original_hdr)

    parser.add_argument("--test_dataroot_ldr", type=str, default=params.test_dataroot_ldr)
    parser.add_argument("--result_dir_prefix", type=str, default="local")
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--loss_g_d_factor", type=float, default=1)
    parser.add_argument("--ssim_loss_factor", type=float, default=1)
    parser.add_argument("--ssim_loss", type=str, default=params.ssim_custom)
    parser.add_argument("--ssim_window_size", type=int, default=11)
    # if 0, images are in [-1, 1] range, if 0.5 then [0,1]
    parser.add_argument("--input_images_mean", type=float, default=0)
    parser.add_argument("--log_factor", type=int, default=1000)
    parser.add_argument("--use_transform_exp", type=int, default=1)  # int(False) = 0
    parser.add_argument("--epoch_to_save", type=int, default=2)
    parser.add_argument("--decay_epoch", type=int, default=5, help="epoch from which to start lr decay")
    parser.add_argument("--d_down_dim", type=int, default=params.dim)
    parser.add_argument("--d_norm", type=str, default='instance_norm')
    parser.add_argument("--pyramid_loss", type=int, default=1)
    parser.add_argument('--pyramid_weight_list', help='delimited list input', type=str, default="0.7,0.5,0.2")
    args = parser.parse_args()
    return args


def get_opt():
    opt = parse_arguments()
    opt.data_root_npy = os.path.join(opt.data_root_npy)
    opt.data_root_ldr = os.path.join(opt.data_root_ldr)
    opt.test_data_root_npy = os.path.join(opt.test_dataroot_npy)
    opt.test_data_root_ldr = os.path.join(opt.test_dataroot_ldr)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    # device = torch.device("cpu")
    opt.device = device
    isCheckpoint = True
    if opt.checkpoint == 'no':
        isCheckpoint = False
    opt.isCheckpoint = isCheckpoint
    opt.output_dir = g_t_utils.create_dir(opt.result_dir_prefix + "_log_" + str(opt.log_factor), opt.model,
                                          opt.con_operator,
                                          params.models_save_path,
                                          params.loss_path, params.results_path, opt.unet_depth)
    opt.pyramid_weight_list = [float(item) for item in opt.pyramid_weight_list.split(',')]

    return opt
