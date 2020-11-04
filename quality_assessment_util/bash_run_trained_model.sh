#!/bin/bash

func_to_run="run_trained_model"

model_name="11_03"
input_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data/"
#input_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
output_name="script_test"
model_path="/Users/yaelvinker/Documents/university/lab/Nov/same_run_fid_check/"
f_factor_path="/Users/yaelvinker/Documents/university/lab/lum_hist_re/valid_hist_dict_20_bins.npy"
test_mode_f_factor=0
test_mode_frame=0
#input_images_names_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
input_images_names_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data/"

python3.6 image_quality_assessment_util.py --func_to_run $func_to_run --model_name $model_name \
  --input_path $input_path --output_name $output_name --model_path $model_path \
  --f_factor_path $f_factor_path --test_mode_f_factor $test_mode_f_factor --test_mode_frame $test_mode_frame \
  --input_images_names_path $input_images_names_path