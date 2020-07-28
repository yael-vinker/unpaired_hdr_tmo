#!/bin/bash

im_path=""
full_size_im=0
num_steps=1001
output_path=""

# ==== LOSS ====
wind_size=5
pyramid_weight_list="2,4,6"
apply_sig_mu_ssim=0
struct_method="gamma_ssim"
std_norm_factor=0.8

std_pyramid_weight_list="4,1,1"
intensity_epsilon=0.00001
alpha=0.5
std_method="gamma_factor_loss_bilateral"
blf_alpha=0.8
mu_pyramid_weight_list="2,2,2"


sbatch --mem=4000m -c1 --gres=gpu:1 --time=2-0 train.sh \
  $im_path $full_size_im $num_steps $output_path \
  $wind_size $pyramid_weight_list $apply_sig_mu_ssim $struct_method $std_norm_factor \
  $std_pyramid_weight_list $intensity_epsilon $alpha $std_method $blf_alpha \
  $mu_pyramid_weight_list