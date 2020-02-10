#!/bin/bash

change_random_seed=0
batch_size=16
num_epochs=500
G_lr=0.00001
D_lr=0.000005
model="unet"
con_operator="original_unet"
use_xaviar=1
ssim_loss_factor=2
pyramid_weight_list="1,1,1"
data_root_npy="/cs/snapless/raananf/yael_vinker/data/new_data/train/hdrplus_use_factorise_data_1_factor_coeff_01_use_normalization_0"
data_root_ldr="/cs/snapless/raananf/yael_vinker/data/new_data/train/flicker_use_factorise_data_1_factor_coeff_01_use_normalization_0"
test_dataroot_npy="/cs/snapless/raananf/yael_vinker/data/new_data/test/hdrplus_use_factorise_data_1_factor_coeff_01_use_normalization_0"
test_dataroot_original_hdr="/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr"
test_dataroot_ldr="/cs/snapless/raananf/yael_vinker/data/new_data/test/flicker_use_factorise_data_1_factor_coeff_01_use_normalization_0"
result_dir_prefix="/cs/snapless/raananf/yael_vinker/02_09/vanilla/_"
use_factorise_data=1
factor_coeff=0.1
add_clipping=1
use_normalization=0
normalization="max_normalization"
last_layer="none"

sbatch --mem=4000m -c2 --gres=gpu:2 --time=2-0 train1.sh $change_random_seed $batch_size $num_epochs \
  $G_lr $D_lr $model $con_operator $use_xaviar $ssim_loss_factor $pyramid_weight_list $data_root_npy \
  $data_root_ldr $test_dataroot_npy $test_dataroot_original_hdr $test_dataroot_ldr $result_dir_prefix \
  $use_factorise_data $factor_coeff $add_clipping $use_normalization $normalization $last_layer

change_random_seed=0
batch_size=16
num_epochs=500
G_lr=0.00001
D_lr=0.000005
model="unet"
con_operator="original_unet"
use_xaviar=1
ssim_loss_factor=2
pyramid_weight_list="1,1,1"
data_root_npy="/cs/snapless/raananf/yael_vinker/data/new_data/train/hdrplus_use_factorise_data_1_factor_coeff_01_use_normalization_0"
data_root_ldr="/cs/snapless/raananf/yael_vinker/data/new_data/train/flicker_use_factorise_data_1_factor_coeff_01_use_normalization_0"
test_dataroot_npy="/cs/snapless/raananf/yael_vinker/data/new_data/test/hdrplus_use_factorise_data_1_factor_coeff_01_use_normalization_0"
test_dataroot_original_hdr="/cs/labs/raananf/yael_vinker/data/test/tmqi_test_hdr"
test_dataroot_ldr="/cs/snapless/raananf/yael_vinker/data/new_data/test/flicker_use_factorise_data_1_factor_coeff_01_use_normalization_0"
result_dir_prefix="/cs/snapless/raananf/yael_vinker/02_09/vanilla/_"
use_factorise_data=1
factor_coeff=0.1
add_clipping=1
use_normalization=0
normalization="min_max_normalization"
last_layer="none"

sbatch --mem=4000m -c2 --gres=gpu:2 --time=2-0 train1.sh $change_random_seed $batch_size $num_epochs \
  $G_lr $D_lr $model $con_operator $use_xaviar $ssim_loss_factor $pyramid_weight_list $data_root_npy \
  $data_root_ldr $test_dataroot_npy $test_dataroot_original_hdr $test_dataroot_ldr $result_dir_prefix \
  $use_factorise_data $factor_coeff $add_clipping $use_normalization $normalization $last_layer