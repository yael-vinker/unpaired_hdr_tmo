#!/bin/bash
change_random_seed=0
batch_size=16
num_epochs=325
G_lr=0.00001
D_lr=0.00001
lr_decay_step=50
model="unet"
con_operator="square_and_square_root"
unet_norm="none"
use_xaviar=1
g_activation="relu"
d_pretrain_epochs=50

# ====== DATASET ======
data_root_npy="/cs/snapless/raananf/yael_vinker/data/04_26_new_data/hdrplus_gamma_log_10_with_gamma_factor_train"
#data_root_npy="/cs/snapless/raananf/yael_vinker/data/new_data/train/train_hdrplus_new_f_1"
data_root_ldr="/cs/snapless/raananf/yael_vinker/data/div2k_large/train_half2"
test_dataroot_npy="/cs/snapless/raananf/yael_vinker/data/04_26_new_data/hdrplus_gamma_log_10_with_gamma_factor_train_test"
#test_dataroot_npy="/cs/snapless/raananf/yael_vinker/data/new_data/test/test_hdrplus_new_f_1"
test_dataroot_original_hdr="/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr"
test_dataroot_ldr="/cs/snapless/raananf/yael_vinker/data/div2k_large/test_half"
f_factor_path="/cs/labs/raananf/yael_vinker/data/test/test_factors.npy"
#f_factor_path="none"
gamma_log=10
use_new_f=0

add_frame=0
input_dim=1
add_clipping=0
apply_exp=0

use_factorise_data=1
factor_coeff=1
use_normalization=0
last_layer="sigmoid"
custom_sig_factor=3
d_model="original"
num_D=0
d_last_activation="sigmoid"
d_down_dim=16
d_norm="none"
milestones="200"
epoch_to_save=40
final_epoch=320
d_nlayers=3

# =================== LOSS ==================

std_method="gamma_factor_loss_bilateral"

intensity_epsilon=0.00001
apply_intensity_loss_laplacian_weights=0

loss_g_d_factor=5
train_with_D=1
multi_scale_D=0

ssim_window_size=5
struct_method="gamma_ssim"
ssim_loss_factor=1
pyramid_weight_list="2,4,4"

alpha=0.5
bilateral_sigma_r=0.07
apply_intensity_loss=0
std_pyramid_weight_list="0"
std_mul_max=0

mu_loss_factor=0
mu_pyramid_weight_list="0"
normalization="stretch"
max_stretch=1.05
min_stretch=0.025
bilateral_mu=1
blf_input="log"
blf_alpha=0.8

enhance_detail=0
stretch_g="none"
g_doubleConvTranspose=1
d_fully_connected=0
simpleD_maxpool=1
data_trc="gamma"

result_dir_prefix="/cs/labs/raananf/yael_vinker/Aug/01_18/results_08_18/general_test/"
echo "========================= 1 ==========================="
sbatch --mem=8000m -c2 --gres=gpu:2 --time=2-0 train.sh \
  $change_random_seed $batch_size $num_epochs \
  $G_lr $D_lr $model $con_operator $use_xaviar \
  $loss_g_d_factor $train_with_D $ssim_loss_factor $pyramid_weight_list $apply_intensity_loss \
  $intensity_epsilon $std_pyramid_weight_list $mu_loss_factor $mu_pyramid_weight_list \
  $data_root_npy $data_root_ldr $test_dataroot_npy $test_dataroot_original_hdr $test_dataroot_ldr \
  $result_dir_prefix $use_factorise_data $factor_coeff $add_clipping $use_normalization \
  $normalization $last_layer $d_model $d_down_dim $d_norm $milestones $add_frame $input_dim \
  $apply_intensity_loss_laplacian_weights $std_method $alpha $struct_method \
  $bilateral_sigma_r $apply_exp $f_factor_path $gamma_log $custom_sig_factor \
  $epoch_to_save $final_epoch $bilateral_mu $max_stretch $min_stretch $ssim_window_size \
  $use_new_f $blf_input $blf_alpha $std_mul_max $multi_scale_D $g_activation $d_last_activation \
  $lr_decay_step $d_nlayers $d_pretrain_epochs $num_D $unet_norm $enhance_detail \
  $stretch_g $g_doubleConvTranspose $d_fully_connected $simpleD_maxpool $data_trc