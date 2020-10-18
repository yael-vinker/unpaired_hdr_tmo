#!/bin/bash

func_to_run="run_trained_model"
model_names=("D_[0.8,0.5,0]__G_ssr_d1.0_struct_1.0[0.4,0.8,0.8]___noframe__min_log_0.5hist_fit_" \
            "D_[0.8,0.5,0]__G_ssr_d1.0_struct_1.0[0.4,0.8,0.8]___noframe__min_log_0.8hist_fit_" \
            "D_[0.8,0.5,0]__G_ssr_d1.0_struct_1.0[0.4,0.8,0.8]___noframe__min_log_1.0hist_fit_" \
            "D_[0.8,0.5,0]__G_ssr_d1.0_struct_1.0[0.4,0.8,0.8]__rseed7414_noframe__min_log_0.5hist_fit_" \
            "D_[0.8,0.5,0]__G_ssr_d1.0_struct_1.0[0.4,0.8,0.8]__rseed8467_noframe__min_log_0.5hist_fit_" \
            "D_[1,1,0]__G_ssr_d1.0_struct_1.0[0.4,0.8,0.8]___noframe__min_log_0.5hist_fit_" \
            "D_[1,1,1]__G_ssr_d1.0_struct_1.0[0.5,1,1]___noframe__min_log_0.1hist_fit_")
output_name="fid_new_subset"
model_path="/cs/labs/raananf/yael_vinker/Oct/10_15/results_10_16/fix_train_dataset/"
f_factor_path="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/dng_hist_20_bins_all_fix.npy"
test_mode_f_factor=0
test_mode_frame=1
input_path="/cs/labs/raananf/yael_vinker/dng_collection"
input_images_names_path="/cs/snapless/raananf/yael_vinker/data/new_data_crop_fix/test_fid"

for ((i = 0; i < ${#model_names[@]}; ++i)); do
  model_name="${model_names[i]}"
  echo "model_name $model_name"
  sbatch --mem=4000m -c1 --gres=gpu:1 --time=2-0 run_trained_model.sh $func_to_run $model_name \
    $input_path $output_name $model_path \
    $f_factor_path $test_mode_f_factor $test_mode_frame $input_images_names_path
done