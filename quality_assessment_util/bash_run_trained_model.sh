#!/bin/bash

func_to_run="run_trained_model"

model_name="D_[1,1,0]__G_ssr_d1.0_struct_1.0[0.4,0.8,0.8]___noframe__min_log_0.1hist_fit_/"
#input_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
input_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
output_name="script_test"
model_path="/Users/yaelvinker/Documents/university/lab/Oct/10_15/summary_10_16/fix_train_dataset"
f_factor_path="/Users/yaelvinker/PycharmProjects/lab/data_generator/hist_fit/temp_data_test20_bins.npy"
test_mode_f_factor=0
test_mode_frame=0
input_images_names_path=""

python3.6 image_quality_assessment_util.py --func_to_run $func_to_run --model_name $model_name \
  --input_path $input_path --output_name $output_name --model_path $model_path \
  --f_factor_path $f_factor_path --test_mode_f_factor $test_mode_f_factor --test_mode_frame $test_mode_frame $input_images_names_path