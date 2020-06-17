#!/bin/bash
change_random_seed=0
batch_size=16
num_epochs=325
G_lr=0.00001
D_lr=0.000005
model="unet"
con_operator="square_and_square_root"
use_xaviar=1

# ====== DATASET ======
data_root_npy="/cs/snapless/raananf/yael_vinker/data/04_26_new_data/hdrplus_gamma_log_10_with_gamma_factor_train"
data_root_ldr="/cs/snapless/raananf/yael_vinker/data/div2k_large/train_half2"
test_dataroot_npy="/cs/snapless/raananf/yael_vinker/data/04_26_new_data/hdrplus_gamma_log_10_with_gamma_factor_train_test"
test_dataroot_original_hdr="/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr"
test_dataroot_ldr="/cs/snapless/raananf/yael_vinker/data/div2k_large/test_half"
result_dir_prefix="/cs/labs/raananf/yael_vinker/06_21/results/"
f_factor_path="/cs/labs/raananf/yael_vinker/data/test/test_factors.npy"
gamma_log=10

add_frame=1
input_dim=1
add_clipping=0
apply_exp=0

use_factorise_data=1
factor_coeff=1
use_normalization=0
last_layer="sigmoid"
custom_sig_factor=3
d_model="patchD"
d_down_dim=64
d_norm="none"
milestones="200"
epoch_to_save=40
final_epoch=320

std_method="gamma_factor_loss_bilateral"
intensity_epsilon=0.00001
apply_intensity_loss_laplacian_weights=1

# ====== LOSS ======
loss_g_d_factor=1
train_with_D=1

struct_method="gamma_ssim"
ssim_loss_factor=1
pyramid_weight_list="2,4,6"

alpha=0.5
bilateral_sigma_r=0.1
apply_intensity_loss=1
std_pyramid_weight_list="8,4,1"

mu_loss_factor=2
mu_pyramid_weight_list="1,1,1"
normalization="stretch"
max_stretch=1.05
min_stretch=0.025
bilateral_mu=1

echo "========================= 1 ==========================="
sbatch --mem=4000m -c2 --gres=gpu:2 --time=2-0 train.sh \
  $change_random_seed $batch_size $num_epochs \
  $G_lr $D_lr $model $con_operator $use_xaviar \
  $loss_g_d_factor $train_with_D $ssim_loss_factor $pyramid_weight_list $apply_intensity_loss \
  $intensity_epsilon $std_pyramid_weight_list $mu_loss_factor $mu_pyramid_weight_list \
  $data_root_npy $data_root_ldr $test_dataroot_npy $test_dataroot_original_hdr $test_dataroot_ldr \
  $result_dir_prefix $use_factorise_data $factor_coeff $add_clipping $use_normalization \
  $normalization $last_layer $d_model $d_down_dim $d_norm $milestones $add_frame $input_dim \
  $apply_intensity_loss_laplacian_weights $std_method $alpha $struct_method \
  $bilateral_sigma_r $apply_exp $f_factor_path $gamma_log $custom_sig_factor \
  $epoch_to_save $final_epoch $bilateral_mu $max_stretch $min_stretch