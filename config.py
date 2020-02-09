import argparse
import os
import torch
from utils import params
import random


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    # ====== GENERAL SETTINGS ======
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--change_random_seed", type=int, default=1)

    # ====== TRAINING ======
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=params.num_epochs)
    parser.add_argument("--G_lr", type=float, default=params.lr)
    parser.add_argument("--D_lr", type=float, default=params.lr)
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--milestones", type=str, default='100', help="epoch from which to start lr decay")

    # ====== ARCHITECTURES ======
    parser.add_argument("--model", type=str, default=params.unet_network)  # up sampling is the default
    parser.add_argument("--filters", type=int, default=params.filters)
    parser.add_argument("--unet_depth", type=int, default=4)
    parser.add_argument("--con_operator", type=str, default=params.square_and_square_root)
    parser.add_argument('--unet_norm', type=str, default='none', help="none/instance_norm/batch_norm")
    parser.add_argument("--d_down_dim", type=int, default=params.dim_d)
    parser.add_argument("--d_norm", type=str, default='none')
    parser.add_argument('--last_layer', type=str, default='sigmoid', help="none/tanh")
    parser.add_argument('--use_xaviar', type=int, default=1)

    # ====== LOSS ======
    parser.add_argument("--loss_g_d_factor", type=float, default=1)
    parser.add_argument("--ssim_loss_factor", type=float, default=1)
    parser.add_argument("--ssim_loss", type=str, default=params.ssim_custom)
    parser.add_argument("--ssim_window_size", type=int, default=5)
    parser.add_argument("--pyramid_loss", type=int, default=1)
    parser.add_argument('--pyramid_weight_list', help='delimited list input', type=str, default="0.7,0.5,0.2")

    # ====== DATASET ======
    parser.add_argument("--data_root_npy", type=str, default=params.train_dataroot_hdr)
    parser.add_argument("--data_root_ldr", type=str, default=params.train_dataroot_ldr)
    parser.add_argument("--test_dataroot_npy", type=str, default=params.test_dataroot_hdr)
    parser.add_argument("--test_dataroot_original_hdr", type=str, default=params.test_dataroot_original_hdr)
    parser.add_argument("--test_dataroot_ldr", type=str, default=params.test_dataroot_ldr)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--input_images_mean", type=float, default=0)
    parser.add_argument('--use_factorise_data', type=int, default=1)
    parser.add_argument('--factor_coeff', type=float, default=0.1)

    # ====== POST PROCESS ======
    parser.add_argument("--add_frame", type=int, default=1)  # int(False) = 0
    parser.add_argument("--add_clipping", type=int, default=1)  # int(False) = 0
    parser.add_argument('--use_normalization', type=int, default=1)
    parser.add_argument("--log_factor", type=float, default=1000)
    parser.add_argument("--normalization", type=str, default='min_max_normalization', help='max/min_max')

    # ====== SAVE RESULTS ======
    parser.add_argument("--epoch_to_save", type=int, default=5)
    parser.add_argument("--result_dir_prefix", type=str, default="")

    args = parser.parse_args()
    return args


def get_opt():
    import numpy as np
    opt = parse_arguments()
    if opt.change_random_seed:
        manualSeed = random.randint(1, 10000)
    else:
        manualSeed = params.manualSeed
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    opt.manual_seed = manualSeed
    opt.data_root_npy = os.path.join(opt.data_root_npy)
    opt.data_root_ldr = os.path.join(opt.data_root_ldr)
    opt.test_data_root_npy = os.path.join(opt.test_dataroot_npy)
    opt.test_data_root_ldr = os.path.join(opt.test_dataroot_ldr)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    # device = torch.device("cpu")
    opt.device = device

    opt.output_dir = create_dir(opt)
    opt.pyramid_weight_list = [float(item) for item in opt.pyramid_weight_list.split(',')]
    opt.milestones = [int(item) for item in opt.milestones.split(',')]
    return opt


def create_dir(opt):
    result_dir_pref, model_name, con_operator, model_depth, filters, add_frame = opt.result_dir_prefix, opt.model, opt.con_operator, opt.unet_depth, opt.filters, opt.add_frame
    output_dir = result_dir_pref + "_random_seed_" + str(bool(opt.change_random_seed)) + "_" + model_name + "_" + con_operator + "_last_act_" + opt.last_layer \
                 + "_norm_g_" + opt.unet_norm + "_use_f_" + str(bool(opt.use_factorise_data)) + "_coeff_" \
                 + str(opt.factor_coeff) + "_clip_" + str(bool(opt.add_clipping)) + "_normalise_" + opt.normalization
    model_path = params.models_save_path
    loss_graph_path = params.loss_path
    result_path = params.results_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Directory ", output_dir, " created")

    best_acc_path = os.path.join(output_dir, params.best_acc_images_path)
    models_images = os.path.join(output_dir, params.models_images)
    model_path = os.path.join(output_dir, model_path)
    models_250_save_path = os.path.join("models_250", "models_250_net.pth")
    model_path_250 = os.path.join(output_dir, models_250_save_path)
    best_model_save_path = os.path.join("best_model", "best_model.pth")
    best_model_path = os.path.join(output_dir, best_model_save_path)
    loss_graph_path = os.path.join(output_dir, loss_graph_path)
    result_path = os.path.join(output_dir, result_path)
    acc_path = os.path.join(output_dir, "accuracy")
    tmqi_path = os.path.join(output_dir, "tmqi")
    gradient_flow_path = os.path.join(output_dir, params.gradient_flow_path, "g")

    if not os.path.exists(models_images):
        os.mkdir(models_images)
        print("Directory ", models_images, " created")

    if not os.path.exists(best_acc_path):
        os.mkdir(best_acc_path)
        print("Directory ", best_acc_path, " created")

    if not os.path.exists(os.path.dirname(gradient_flow_path)):
        os.makedirs(os.path.dirname(gradient_flow_path))
        print("Directory ", gradient_flow_path, " created")

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print("Directory ", model_path, " created")

    if not os.path.exists(os.path.dirname(model_path_250)):
        os.makedirs(os.path.dirname(model_path_250))
        print("Directory ", model_path_250, " created")

    if not os.path.exists(os.path.dirname(best_model_path)):
        os.makedirs(os.path.dirname(best_model_path))
        print("Directory ", best_model_path, " created")

    if not os.path.exists(loss_graph_path):
        os.mkdir(loss_graph_path)

        print("Directory ", loss_graph_path, " created")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print("Directory ", result_path, " created")
    if not os.path.exists(acc_path):
        os.mkdir(acc_path)
        print("Directory ", acc_path, " created")

    if not os.path.exists(tmqi_path):
        os.mkdir(tmqi_path)
        print("Directory ", tmqi_path, " created")
    return output_dir