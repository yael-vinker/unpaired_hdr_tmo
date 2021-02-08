#!/bin/bash

func_to_run="run_trained_model"

#model_name="11_04_D_[1,1,1]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate__noframe__min_log_0.1hist_fit_"
#input_path="/Users/yaelvinker/Documents/university/data/hdr_gallery_3/"
model_name="11_08_lr15D_size268_D_[1,1,1]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate__noframe__min_log_0.1hist_fit_"

input_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
output_name="gray_0.01_color_stretch"
model_path="/Users/yaelvinker/Documents/university/lab/Nov/good_models"
f_factor_path="/Users/yaelvinker/Documents/university/lab/lum_hist_re/exr_hist_dict_20_bins.npy"
#f_factor_path="/Users/yaelvinker/Documents/university/lab/lum_hist_re/hdr_gallery20_bins.npy"
test_mode_f_factor=0
test_mode_frame=0
input_images_names_path="/Users/yaelvinker/PycharmProjects/lab/utils/folders/temp_data"
#input_images_names_path="/Users/yaelvinker/Documents/university/data/hdr_gallery_3/"

python3.6 image_quality_assessment_util.py --func_to_run $func_to_run --model_name $model_name \
  --input_path $input_path --output_name $output_name --model_path $model_path \
  --f_factor_path $f_factor_path --test_mode_f_factor $test_mode_f_factor --test_mode_frame $test_mode_frame \
  --input_images_names_path $input_images_names_path