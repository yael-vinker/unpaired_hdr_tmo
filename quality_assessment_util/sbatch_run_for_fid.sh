#!/bin/bash

dir="/cs/snapless/raananf/yael_vinker/data/oct_fid_npy_split/"
sub_dirs=("set_0" "set_50" "set_100" "set_150" "set_200" "set_250" "set_300" "set_350" "set_400" "set_450" "set_500" "set_550" "set_600" "set_650" "set_700" "set_750" "set_800" "set_850" "set_900" "set_950" "set_1000")

func_to_run="run_trained_model"
model_name="D_[0.8,0.8,0.8]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate_lr_g1e-05_d1e-05_decay50__noframe__min_log_0.1hist_fit_"
output_name="fid"
model_path="/cs/labs/raananf/yael_vinker/Nov/11_03/results_11_03/tmqi_fid/"
f_factor_path="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/fix_lum_hist/dng_hist_20_bins_all_fix.npy"
test_mode_f_factor=0
test_mode_frame=0
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
