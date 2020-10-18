#!/bin/bash

#dir="/cs/snapless/raananf/yael_vinker/data/dng_data_npy/"
dir="/cs/snapless/raananf/yael_vinker/data/dng_data_npy/set14_b/"
sub_dirs=("set1" "set2" "set3" "set4" "set5" "set6" "set7" "set8" "set9" "set10" "set11" "set12" "set13" "set14" "set15" "set16" "set17" "set18" "set19" "set20" "set21" "set22")

func_to_run="run_trained_model"
model_name="D_[0.8,0.5,0]__G_ssr_d1.0_struct_1.0[0.4,0.8,0.8]___noframe__min_log_0.5hist_fit_"
output_name="fid_color_stretch_fix"
model_path="/cs/labs/raananf/yael_vinker/Oct/10_15/results_10_16/fix_train_dataset/"
f_factor_path="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/dng_hist_20_bins_all_fix.npy"
test_mode_f_factor=0
test_mode_frame=1
input_images_names_path=""

for ((i = 0; i < ${#sub_dirs[@]}; ++i)); do
	cur_sub="${sub_dirs[i]}"
	input_path="$dir/$cur_sub/"
	input_images_names_path="$dir/$cur_sub/"
	echo "$input_path"
	sbatch --mem=4000m -c1 --gres=gpu:1 --time=2-0 run_trained_model.sh $func_to_run $model_name \
    		$input_path $output_name $model_path \
    		$f_factor_path $test_mode_f_factor $test_mode_frame $input_images_names_path
done