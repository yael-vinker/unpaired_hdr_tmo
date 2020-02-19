#!/bin/bash

change_random_seed=0
batch_size=16
num_epochs=500
G_lr=0.00001
D_lr=0.000005
model="unet"
con_operator="square_and_square_root"
use_xaviar=1
ssim_loss_factor=5
pyramid_weight_list="1,1,1"
pyramid_pow=0
data_root_npy="/cs/snapless/raananf/yael_vinker/data/new_data/train/hdrplus_gamma_use_factorise_data_1_factor_coeff_1.0_use_normalization_0"
data_root_ldr="/cs/snapless/raananf/yael_vinker/data/new_data/train/flicker_use_factorise_data_1_factor_coeff_1.0_use_normalization_0"
test_dataroot_npy="/cs/snapless/raananf/yael_vinker/data/new_data/test/hdrplus_gamma_use_factorise_data_1_factor_coeff_1.0_use_normalization_0"
test_dataroot_original_hdr="/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr"
test_dataroot_ldr="/cs/snapless/raananf/yael_vinker/data/new_data/test/flicker_use_factorise_data_1_factor_coeff_1.0_use_normalization_0"
result_dir_prefix="/cs/snapless/raananf/yael_vinker/02_18/results/gamma_"
use_factorise_data=1
factor_coeff=1
add_clipping=0
use_normalization=0
normalization="bugy_max_normalization"
last_layer="sigmoid"
d_model="patchD"
d_down_dim=64
pyramid_loss=0
d_norm="none"
use_sigma_loss=5
use_c3_in_ssim=0

sbatch --mem=8000m -c2 --gres=gpu:2 --time=2-0 train.sh $change_random_seed $batch_size $num_epochs \
  $G_lr $D_lr $model $con_operator $use_xaviar $ssim_loss_factor $pyramid_weight_list $data_root_npy \
  $data_root_ldr $test_dataroot_npy $test_dataroot_original_hdr $test_dataroot_ldr $result_dir_prefix \
  $use_factorise_data $factor_coeff $add_clipping $use_normalization $normalization $last_layer \
  $pyramid_pow $d_model $d_down_dim $pyramid_loss $d_norm $use_sigma_loss $use_c3_in_ssim


change_random_seed=0
batch_size=16
num_epochs=500
G_lr=0.00001
D_lr=0.000005
model="unet"
con_operator="square_and_square_root"
use_xaviar=1
ssim_loss_factor=5
pyramid_weight_list="1,1,1"
pyramid_pow=0
data_root_npy="/cs/snapless/raananf/yael_vinker/data/new_data/train/hdrplus_gamma_use_factorise_data_1_factor_coeff_1.0_use_normalization_0"
data_root_ldr="/cs/snapless/raananf/yael_vinker/data/new_data/train/flicker_use_factorise_data_1_factor_coeff_1.0_use_normalization_0"
test_dataroot_npy="/cs/snapless/raananf/yael_vinker/data/new_data/test/hdrplus_gamma_use_factorise_data_1_factor_coeff_1.0_use_normalization_0"
test_dataroot_original_hdr="/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr"
test_dataroot_ldr="/cs/snapless/raananf/yael_vinker/data/new_data/test/flicker_use_factorise_data_1_factor_coeff_1.0_use_normalization_0"
result_dir_prefix="/cs/snapless/raananf/yael_vinker/02_18/results/gamma_"
use_factorise_data=1
factor_coeff=1
add_clipping=1
use_normalization=0
normalization="bugy_max_normalization"
last_layer="sigmoid"
d_model="original"
d_down_dim=64
pyramid_loss=0
d_norm="none"
use_sigma_loss=5
use_c3_in_ssim=0

sbatch --mem=8000m -c2 --gres=gpu:2 --time=2-0 train.sh $change_random_seed $batch_size $num_epochs \
  $G_lr $D_lr $model $con_operator $use_xaviar $ssim_loss_factor $pyramid_weight_list $data_root_npy \
  $data_root_ldr $test_dataroot_npy $test_dataroot_original_hdr $test_dataroot_ldr $result_dir_prefix \
  $use_factorise_data $factor_coeff $add_clipping $use_normalization $normalization $last_layer \
  $pyramid_pow $d_model $d_down_dim $pyramid_loss $d_norm $use_sigma_loss $use_c3_in_ssim
