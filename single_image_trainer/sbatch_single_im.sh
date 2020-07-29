#!/bin/bash

im_path="/cs/labs/raananf/yael_vinker/July/07_28/lab/single_image_trainer/input/OtterPoint.exr"
full_size_im=1
num_steps=2001
output_path="/cs/labs/raananf/yael_vinker/July/07_28/lab/single_image_trainer/output_07_29/"

# ==== LOSS ====
wind_size=5
apply_sig_mu_ssim=0
struct_method="gamma_ssim"
std_norm_factor=0.8
blf_alpha=0.8

use_struct_loss=0
pyramid_weight_list="0"

use_contrast_loss=1
std_pyramid_weight_list="1,1,1"
std_method="gamma_factor_loss_bilateral"
intensity_epsilon=0.00001
alpha=0.1

use_cmprs_loss=1
mu_pyramid_weight_list="2,2,2"


sbatch --mem=4000m -c1 --gres=gpu:1 --time=1-0 train.sh \
  $im_path $full_size_im $num_steps $output_path \
  $wind_size $pyramid_weight_list $apply_sig_mu_ssim $struct_method $std_norm_factor \
  $std_pyramid_weight_list $intensity_epsilon $alpha $std_method $blf_alpha \
  $mu_pyramid_weight_list $use_struct_loss $use_contrast_loss $use_cmprs_loss


use_struct_loss=1
pyramid_weight_list="2,4,6"

use_contrast_loss=0
std_pyramid_weight_list="0"
std_method="gamma_factor_loss_bilateral"
intensity_epsilon=0.00001
alpha=0.1

use_cmprs_loss=1
mu_pyramid_weight_list="2,2,2"


sbatch --mem=4000m -c1 --gres=gpu:1 --time=1-0 train.sh \
  $im_path $full_size_im $num_steps $output_path \
  $wind_size $pyramid_weight_list $apply_sig_mu_ssim $struct_method $std_norm_factor \
  $std_pyramid_weight_list $intensity_epsilon $alpha $std_method $blf_alpha \
  $mu_pyramid_weight_list $use_struct_loss $use_contrast_loss $use_cmprs_loss

use_struct_loss=1
pyramid_weight_list="2,4,6"

use_contrast_loss=1
std_pyramid_weight_list="1,1,1"
std_method="gamma_factor_loss_bilateral"
intensity_epsilon=0.00001
alpha=0.1

use_cmprs_loss=0
mu_pyramid_weight_list="0"


sbatch --mem=4000m -c1 --gres=gpu:1 --time=1-0 train.sh \
  $im_path $full_size_im $num_steps $output_path \
  $wind_size $pyramid_weight_list $apply_sig_mu_ssim $struct_method $std_norm_factor \
  $std_pyramid_weight_list $intensity_epsilon $alpha $std_method $blf_alpha \
  $mu_pyramid_weight_list $use_struct_loss $use_contrast_loss $use_cmprs_loss

use_struct_loss=1
pyramid_weight_list="2,4,6"

use_contrast_loss=1
std_pyramid_weight_list="1,1,1"
std_method="gamma_factor_loss"
intensity_epsilon=0.00001
alpha=0.1

use_cmprs_loss=1
mu_pyramid_weight_list="2,2,2"


sbatch --mem=4000m -c1 --gres=gpu:1 --time=1-0 train.sh \
  $im_path $full_size_im $num_steps $output_path \
  $wind_size $pyramid_weight_list $apply_sig_mu_ssim $struct_method $std_norm_factor \
  $std_pyramid_weight_list $intensity_epsilon $alpha $std_method $blf_alpha \
  $mu_pyramid_weight_list $use_struct_loss $use_contrast_loss $use_cmprs_loss