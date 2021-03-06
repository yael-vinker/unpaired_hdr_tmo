#!/bin/bash

# ====== GENERAL SETTINGS ======
checkpoint=0
change_random_seed=0

# ====== TRAINING ======
batch_size=16
num_epochs=325
G_lr=0.00001
D_lr=0.000015
lr_decay_step=50
d_pretrain_epochs=50
use_xaviar=1

# ====== SLIDER_MODE ======
manual_d_training=0
d_weight_mul_mode="none"
strong_details_D_weights="1,1,1"
basic_details_D_weights="0.8,0.5,0"

# ====== ARCHITECTURES ======
model="unet"
filters=32
unet_depth=4
con_operator="square_root"
unet_norm="none"
g_activation="relu"

d_down_dim=16
d_nlayers=3
d_norm="none"
last_layer="sigmoid"
d_model="multiLayerD_simpleD"
num_D=3
d_last_activation="sigmoid"
stretch_g="none"
g_doubleConvTranspose=1
d_fully_connected=0
simpleD_maxpool=0
bilinear=0
padding="replicate"
d_padding=0
convtranspose_kernel=2
final_shape_addition=0
up_mode=0
input_dim=1
output_dim=1

# ====== LOSS ======
train_with_D=1
loss_g_d_factor=1
adv_weight_list="1,1,0"
struct_method="gamma_struct"
ssim_loss_factor=1
ssim_window_size=5
pyramid_weight_list="1,1,1"

# ====== DATASET ======
data_root_npy="/cs/snapless/raananf/yael_vinker/data/new_data_crop_fix/train/"
data_root_ldr="/cs/snapless/raananf/yael_vinker/data/div2k_large/train_half2"
test_dataroot_npy="/cs/snapless/raananf/yael_vinker/data/new_data_crop_fix/test"
test_dataroot_original_hdr="/cs/labs/raananf/yael_vinker/data/test/new_test_hdr"
test_dataroot_ldr="/cs/snapless/raananf/yael_vinker/data/div2k_large/test_half"
use_factorise_data=1
factor_coeff=0.1
gamma_log=10
f_factor_path="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/train_valid/valid_hist_dict_20_bins.npy"
use_new_f=0
use_contrast_ratio_f=0
use_hist_fit=1
f_train_dict_path="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/fix_lum_hist/dng_hist_20_bins_all_fix.npy"
data_trc="min_log"
max_stretch=1
min_stretch=0
add_frame=0
normalization="bugy_max_normalization"

# ====== SAVE RESULTS ======
epoch_to_save=40
#result_dir_prefix="/cs/labs/raananf/yael_vinker/Oct/10_26/results_10_26/push_test/"
result_dir_prefix="/cs/labs/raananf/yael_vinker/2021/ICCV_rebuttal/results_06_14/"
final_epoch=320
fid_real_path="/cs/snapless/raananf/yael_vinker/data/div2k_large/test_half2"
#fid_res_path="/cs/labs/raananf/yael_vinker/Oct/10_26/fid_res/"
fid_res_path="/cs/labs/raananf/yael_vinker/2021/ICCV_rebuttal/results/fid_res/"


test_names=("test_git")

for ((i = 0; i < ${#test_names[@]}; ++i)); do
  test_name="${test_names[i]}"

  echo "======================================================"
  echo "tests_name $test_name"

  sbatch -J "vanilla_110" -o "slurm-vanilla_110.out" --mem=8000m -c2 --gres=gpu:1 --time=2-0 train.sh \
    $checkpoint \
    $change_random_seed \
    $batch_size \
    $num_epochs \
    $G_lr \
    $D_lr \
    $lr_decay_step \
    $d_pretrain_epochs \
    $use_xaviar \
    $manual_d_training \
    $d_weight_mul_mode \
    $strong_details_D_weights \
    $basic_details_D_weights \
    $model \
    $filters \
    $unet_depth \
    $con_operator \
    $unet_norm \
    $g_activation \
    $d_down_dim \
    $d_nlayers \
    $d_norm \
    $last_layer \
    $d_model \
    $num_D \
    $d_last_activation \
    $stretch_g \
    $g_doubleConvTranspose \
    $d_fully_connected \
    $simpleD_maxpool \
    $bilinear \
    $padding \
    $d_padding \
    $convtranspose_kernel \
    $final_shape_addition \
    $up_mode \
    $input_dim \
    $output_dim \
    $train_with_D \
    $loss_g_d_factor \
    $adv_weight_list \
    $struct_method \
    $ssim_loss_factor \
    $ssim_window_size \
    $pyramid_weight_list \
    $data_root_npy \
    $data_root_ldr \
    $test_dataroot_npy \
    $test_dataroot_original_hdr \
    $test_dataroot_ldr \
    $use_factorise_data \
    $factor_coeff \
    $gamma_log \
    $f_factor_path \
    $use_new_f \
    $use_contrast_ratio_f \
    $use_hist_fit \
    $f_train_dict_path \
    $data_trc \
    $max_stretch \
    $min_stretch \
    $add_frame \
    $normalization \
    $epoch_to_save \
    $result_dir_prefix \
    $final_epoch \
    $fid_real_path \
    $fid_res_path
  echo "======================================================"
done