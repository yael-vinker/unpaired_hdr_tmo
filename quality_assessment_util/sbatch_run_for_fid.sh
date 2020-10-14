#!/bin/bash

#dir="/cs/snapless/raananf/yael_vinker/data/dng_data_npy/"
dir="/cs/snapless/raananf/yael_vinker/data/dng_data_npy/set14_b/"
sub_dirs=("set1" "set2" "set3" "set4" "set5" "set6" "set7" "set8" "set9" "set10" "set11" "set12" "set13" "set14" "set15" "set16" "set17" "set18" "set19" "set20" "set21" "set22")

func_to_run="run_trained_model"
model_name="1_D_multiLayerD_simpleD__num_D3_1,1,1_ch16_3layers_sigmoid__G_unet_ssr_relu_doubleConvT___rseed_Truepretrain50_lr_g1e-05_d1e-05_decay50_noframe__LOSS_d1.0_gamma_ssim1.0_1,1,1__DATA_min_log_0.1hist_fit_"
output_name="fid_color_stretch_fix"
model_path="/Users/yaelvinker/Documents/university/lab/Oct/10_13/10_13_summary"
f_factor_path="/Users/yaelvinker/PycharmProjects/lab/data_generator/hist_fit/temp_data_test20_bins.npy"
test_mode_f_factor=0
test_mode_frame=0

for ((i = 0; i < ${#sub_dirs[@]}; ++i)); do
	cur_sub="${sub_dirs[i]}"
	input_path="$dir/$cur_sub/"
	echo "$input_path"
	sbatch --mem=4000m -c1 --gres=gpu:1 --time=2-0 run_trained_model.sh $func_to_run $model_name \
  $input_path $output_name $model_path \
  $f_factor_path $test_mode_f_factor $test_mode_frame
done