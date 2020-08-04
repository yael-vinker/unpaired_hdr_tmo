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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--G_lr", type=float, default=params.lr)
    parser.add_argument("--D_lr", type=float, default=params.lr)
    parser.add_argument("--lr_decay_step", type=float, default=30)
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--milestones", type=str, default='100', help="epoch from which to start lr decay")
    parser.add_argument("--d_pretrain_epochs", type=int, default=5)

    # ====== ARCHITECTURES ======
    parser.add_argument("--model", type=str, default=params.unet_network)  # up sampling is the default
    parser.add_argument("--filters", type=int, default=params.filters)
    parser.add_argument("--unet_depth", type=int, default=4)
    parser.add_argument("--con_operator", type=str, default=params.square_and_square_root)
    parser.add_argument('--unet_norm', type=str, default='none', help="none/instance_norm/batch_norm")
    parser.add_argument('--g_activation', type=str, default='relu', help="none/relu/leakyrelu")
    parser.add_argument("--d_down_dim", type=int, default=64)
    parser.add_argument("--d_nlayers", type=int, default=5)
    parser.add_argument("--d_norm", type=str, default='none')
    parser.add_argument('--last_layer', type=str, default='sigmoid', help="none/tanh")
    parser.add_argument('--custom_sig_factor', type=float, default=3)
    parser.add_argument('--use_xaviar', type=int, default=1)
    parser.add_argument('--d_model', type=str, default='patchD', help="original/patchD")
    parser.add_argument('--d_last_activation', type=str, default='sigmoid', help="sigmoid/none")
    parser.add_argument('--apply_exp', type=int, default=0)

    # ====== LOSS ======
    parser.add_argument('--train_with_D', type=int, default=1)
    parser.add_argument("--loss_g_d_factor", type=float, default=1)
    parser.add_argument("--multi_scale_D", type=int, default=0)
    parser.add_argument('--struct_method', type=str, default="gamma_ssim") # hdr_ssim, gamma_ssim, div_ssim, laplace_ssim
    parser.add_argument("--ssim_loss_factor", type=float, default=1)
    parser.add_argument("--ssim_window_size", type=int, default=5)
    parser.add_argument('--pyramid_weight_list', help='delimited list input', type=str, default="2,2,6")

    parser.add_argument('--apply_intensity_loss', type=float, default=0)
    parser.add_argument('--std_method', type=str, default="gamma_factor_loss_bilateral")
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--apply_intensity_loss_laplacian_weights', type=int, default=1)
    parser.add_argument('--intensity_epsilon', type=float, default=0.00001)
    parser.add_argument('--std_pyramid_weight_list', help='delimited list input', type=str, default="5,5,1")

    parser.add_argument('--mu_loss_factor', type=float, default=0)
    parser.add_argument('--mu_pyramid_weight_list', help='delimited list input', type=str, default="1,1,1")

    parser.add_argument('--apply_sig_mu_ssim', type=int, default=0)
    parser.add_argument('--bilateral_sigma_r', type=float, default=0.05)
    parser.add_argument('--bilateral_mu', type=float, default=1)
    parser.add_argument('--std_mul_max', type=int, default=0)
    parser.add_argument('--blf_input', type=str, default="log",
                        help="can be 'log' for log(hdr/hdr.max * brightness)**alpha or 'gamma' for gamma.")
    parser.add_argument('--blf_alpha', type=float, default=0.8,
                        help="if blf_input is log than specify alpha")

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
    parser.add_argument('--gamma_log', type=int, default=10)
    # parser.add_argument('--f_factor_path', type=str, default=params.f_factor_path) #52180538.8149
    parser.add_argument('--f_factor_path', type=str, default="none")  # 52180538.8149
    parser.add_argument('--use_new_f', type=int, default=1)
    parser.add_argument('--max_stretch', type=float, default=1)
    parser.add_argument('--min_stretch', type=float, default=0)

    # ====== POST PROCESS ======
    parser.add_argument("--add_frame", type=int, default=0)  # int(False) = 0
    parser.add_argument("--add_clipping", type=int, default=0)  # int(False) = 0
    parser.add_argument('--use_normalization', type=int, default=0)
    parser.add_argument("--log_factor", type=float, default=1000)
    parser.add_argument("--normalization", type=str, default='bugy_max_normalization', help='max/min_max')

    # ====== SAVE RESULTS ======
    parser.add_argument("--epoch_to_save", type=int, default=2)
    parser.add_argument("--result_dir_prefix", type=str, default="")
    parser.add_argument("--final_epoch", type=int, default=0)

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
    opt.std_pyramid_weight_list = torch.FloatTensor([float(item) for item in opt.std_pyramid_weight_list.split(',')]).to(device)
    opt.mu_pyramid_weight_list = torch.FloatTensor([float(item) for item in opt.mu_pyramid_weight_list.split(',')]).to(device)
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
                          "use_c3": False,
                          "wind_norm_option": opt.wind_norm_option,
                          "max_stretch": opt.max_stretch,
                          "min_stretch": opt.min_stretch}
    return dataset_properties


def create_dir(opt):
    result_dir_pref, model_name, con_operator, model_depth, filters, add_frame = opt.result_dir_prefix, opt.model, \
                                                                                 opt.con_operator, opt.unet_depth, \
                                                                                 opt.filters, opt.add_frame
    if opt.d_pretrain_epochs:
        result_dir_pref = result_dir_pref + "pretrain" + str(opt.d_pretrain_epochs) + "_"
    if not opt.add_frame:
        result_dir_pref = result_dir_pref + "no_frame_"
    result_dir_pref = result_dir_pref + "g" + str(opt.G_lr) + "_d" + \
                      str(opt.D_lr) + "_decay" + str(opt.lr_decay_step) + "_"
    if opt.normalization == "stretch":
        result_dir_pref = result_dir_pref + "stretch_" + str(opt.max_stretch)
    if opt.add_clipping:
        result_dir_pref = result_dir_pref + "clip_"
    if opt.apply_exp:
        result_dir_pref = result_dir_pref + "exp_"
    if opt.use_new_f:
        result_dir_pref = result_dir_pref + "new_f_"
    else:
        result_dir_pref = result_dir_pref + "data" + str(opt.gamma_log) + "_"
    if not opt.train_with_D:
        result_dir_pref = result_dir_pref + "no_D_"
    if opt.change_random_seed:
        result_dir_pref = result_dir_pref + "_rseed_" + str(bool(opt.change_random_seed))
    if opt.apply_wind_norm:
        result_dir_pref = result_dir_pref + "apply_wind_norm_" + opt.wind_norm_option + "_factor_" + str(opt.std_norm_factor)
    if opt.apply_sig_mu_ssim:
        result_dir_pref = result_dir_pref + "apply_sig_mu_ssim"
    result_dir_pref = result_dir_pref + "d" + str(opt.loss_g_d_factor)
    if opt.ssim_loss_factor:
        result_dir_pref = result_dir_pref + "_" + opt.struct_method + str(opt.ssim_loss_factor) + "_" + opt.pyramid_weight_list + "_"
    if opt.apply_intensity_loss:
        s = opt.std_method + str(opt.apply_intensity_loss) + "_" + opt.std_pyramid_weight_list + "_wind" + \
            str(opt.ssim_window_size) + "_"
        if opt.std_mul_max:
            s += "std_mul_max_"
        if opt.apply_intensity_loss_laplacian_weights:
            s = s + "bmu" + str(opt.bilateral_mu) + "_sigr" + str(opt.bilateral_sigma_r) + "_"
            if opt.blf_input == "log":
                s = s + str(opt.blf_input) + str(opt.blf_alpha) + "_"
        result_dir_pref = result_dir_pref + s + "eps" + str(opt.intensity_epsilon)
        if opt.std_method not in ["std", "std_bilateral"]:
            result_dir_pref = result_dir_pref + "_alpha" + str(opt.alpha)
    if opt.mu_loss_factor:
        result_dir_pref = result_dir_pref + "_mu_loss" + str(opt.mu_loss_factor) + "_" + opt.mu_pyramid_weight_list
    if opt.last_layer == "msig":
        result_dir_pref = result_dir_pref + "_msig_" + str(opt.custom_sig_factor)
    output_dir = result_dir_pref \
                 + "_" + model_name + "_" + con_operator + "_act_" + opt.g_activation \
                 + "_d_" + opt.d_model + "_nlayers" + str(opt.d_nlayers) + "_" + opt.d_last_activation
    if opt.multi_scale_D:
        output_dir += "_2scale"
    model_path = params.models_save_path
    loss_graph_path = params.loss_path
    result_path = params.results_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Directory ", output_dir, " created")

    models_images = os.path.join(output_dir, params.models_images)
    model_path = os.path.join(output_dir, model_path)
    best_model_save_path = os.path.join("best_model", "best_model.pth")
    best_model_path = os.path.join(output_dir, best_model_save_path)
    loss_graph_path = os.path.join(output_dir, loss_graph_path)
    result_path = os.path.join(output_dir, result_path)
    acc_path = os.path.join(output_dir, "accuracy")

    if not os.path.exists(models_images):
        os.mkdir(models_images)
        print("Directory ", models_images, " created")

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        print("Directory ", model_path, " created")

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
    return output_dir
