import argparse
import os
import torch
from utils import params
import random


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser for gan network")
    # ====== GENERAL SETTINGS ======
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--change_random_seed", type=int, default=0)

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
    parser.add_argument("--con_operator", type=str, default=params.gamma)
    parser.add_argument('--unet_norm', type=str, default='none', help="none/instance_norm/batch_norm")
    parser.add_argument("--d_down_dim", type=int, default=params.dim_d)
    parser.add_argument("--d_norm", type=str, default='batch_norm')
    parser.add_argument('--last_layer', type=str, default='sigmoid', help="none/tanh")
    parser.add_argument('--use_xaviar', type=int, default=1)
    parser.add_argument('--d_model', type=str, default='patchD', help="original/patchD")

    # ====== LOSS ======
    parser.add_argument("--loss_g_d_factor", type=float, default=1)
    parser.add_argument("--ssim_loss_factor", type=float, default=5)
    parser.add_argument("--ssim_loss", type=str, default=params.ssim_custom)
    parser.add_argument("--ssim_window_size", type=int, default=5)
    parser.add_argument("--pyramid_loss", type=int, default=0)
    parser.add_argument('--pyramid_weight_list', help='delimited list input', type=str, default="1,1,1,1,1")
    parser.add_argument('--pyramid_pow', type=int, default=0)
    parser.add_argument('--ssim_compare_to', type=str, default="original")
    parser.add_argument('--use_sigma_loss', type=int, default=0)
    parser.add_argument('--use_c3_in_ssim', type=int, default=1)
    parser.add_argument('--apply_sig_mu_ssim', type=int, default=0)
    parser.add_argument('--train_with_D', type=int, default=1)
    parser.add_argument('--struct_methods', type=str, default="struct_loss_a")

    # ====== DATASET ======
    parser.add_argument("--data_root_npy", type=str, default=params.train_dataroot_hdr)
    parser.add_argument("--data_root_ldr", type=str, default=params.train_dataroot_ldr)
    parser.add_argument("--test_dataroot_npy", type=str, default=params.test_dataroot_hdr)
    parser.add_argument("--test_dataroot_original_hdr", type=str, default=params.test_dataroot_original_hdr)
    parser.add_argument("--test_dataroot_ldr", type=str, default=params.test_dataroot_ldr)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--input_images_mean", type=float, default=0)
    parser.add_argument('--use_factorise_data', type=int, default=1)
    parser.add_argument('--use_factorise_gamma_data', type=int, default=1)
    parser.add_argument('--factor_coeff', type=float, default=1)
    parser.add_argument('--window_tm_data', type=int, default=0)
    parser.add_argument('--apply_wind_norm', type=int, default=0)
    parser.add_argument('--std_norm_factor', type=float, default=0.8)
    parser.add_argument('--wind_norm_option', type=str, default="a")

    # ====== POST PROCESS ======
    parser.add_argument("--add_frame", type=int, default=1)  # int(False) = 0
    parser.add_argument("--add_clipping", type=int, default=0)  # int(False) = 0
    parser.add_argument('--use_normalization', type=int, default=0)
    parser.add_argument("--log_factor", type=float, default=1000)
    parser.add_argument("--normalization", type=str, default='bugy_max_normalization', help='max/min_max')

    # ====== SAVE RESULTS ======
    parser.add_argument("--epoch_to_save", type=int, default=5)
    parser.add_argument("--result_dir_prefix", type=str, default="")

    args = parser.parse_args()
    return args


def get_opt():
    opt = parse_arguments()
    if opt.change_random_seed:
        manualSeed = random.randint(1, 10000)
    else:
        manualSeed = params.manualSeed
    torch.manual_seed(manualSeed)
    opt.manual_seed = manualSeed
    opt = define_dirs(opt)
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    opt.device = device
    opt.pyramid_weight_list = torch.FloatTensor([float(item) for item in opt.pyramid_weight_list.split(',')]).to(device)
    opt.milestones = [int(item) for item in opt.milestones.split(',')]
    opt.dataset_properties = get_dataset_properties(opt)
    return opt


def define_dirs(opt):
    opt.data_root_npy = os.path.join(opt.data_root_npy)
    opt.data_root_ldr = os.path.join(opt.data_root_ldr)
    opt.test_data_root_npy = os.path.join(opt.test_dataroot_npy)
    opt.test_data_root_ldr = os.path.join(opt.test_dataroot_ldr)
    opt.output_dir = create_dir(opt)
    return opt


def get_dataset_properties(opt):
    dataset_properties = {"train_root_npy": opt.data_root_npy,
                          "train_root_ldr": opt.data_root_ldr,
                          "test_dataroot_npy": opt.test_dataroot_npy,
                          "test_dataroot_original_hdr": opt.test_dataroot_original_hdr,
                          "test_dataroot_ldr": opt.test_dataroot_ldr,
                          "input_dim": opt.input_dim,
                          "output_dim": opt.output_dim,
                          "input_images_mean": opt.input_images_mean,
                          "use_factorise_data": opt.use_factorise_data,
                          "use_factorise_gamma_data": opt.use_factorise_gamma_data,
                          "factor_coeff": opt.factor_coeff,
                          "window_tm_data": opt.window_tm_data,
                          "apply_wind_norm": opt.apply_wind_norm,
                          "std_norm_factor": opt.std_norm_factor,
                          "add_frame": opt.add_frame,
                          "batch_size": opt.batch_size,
                          "normalization": opt.normalization,
                          "use_c3": opt.use_c3_in_ssim,
                          "wind_norm_option": opt.wind_norm_option}
    return dataset_properties


def create_dir(opt):
    result_dir_pref, model_name, con_operator, model_depth, filters, add_frame = opt.result_dir_prefix, opt.model, \
                                                                                 opt.con_operator, opt.unet_depth, \
                                                                                 opt.filters, opt.add_frame
    if not opt.train_with_D:
        result_dir_pref = result_dir_pref + "no_D_"
    if opt.change_random_seed:
        result_dir_pref = result_dir_pref + "_rseed_" + str(bool(opt.change_random_seed))
    if opt.apply_wind_norm:
        result_dir_pref = result_dir_pref + "apply_wind_norm_" + opt.wind_norm_option + "_factor_" + str(opt.std_norm_factor)
    if opt.apply_sig_mu_ssim:
        result_dir_pref = result_dir_pref + "apply_sig_mu_ssim"
    output_dir = result_dir_pref \
                 + "_" + model_name + "_" + con_operator \
                 + "_d_model_" + opt.d_model \
                 + "_" + opt.struct_methods + "_" + str(opt.ssim_loss_factor)
    if opt.pyramid_loss:
        output_dir = output_dir + "_pyramid_" + opt.pyramid_weight_list
    output_dir = output_dir + "_sigloss_" + str(opt.use_sigma_loss)
    if opt.window_tm_data:
        output_dir = output_dir + "_window_tm_data"
    else:
        output_dir = output_dir + "_use_f_" + str(bool(opt.use_factorise_data)) \
                 + "_coeff_" + str(opt.factor_coeff) \
                 + "_gamma_input_" + str(bool(opt.use_factorise_gamma_data))



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