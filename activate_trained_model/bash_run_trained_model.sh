#!/bin/bash

model_path="/Users/yaelvinker/PycharmProjects/lab/run_trained_model/model_weights"
model_name="11_08_lr15D_size268_D_[1,1,1]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate__noframe__min_log_0.1hist_fit_"

input_images_path="/Users/yaelvinker/PycharmProjects/lab/run_trained_model/input_images"
#f_factor_path="/Users/yaelvinker/Documents/university/lab/lum_hist_re/exr_hist_dict_20_bins.npy"
f_factor_path="/Users/yaelvinker/PycharmProjects/lab/run_trained_model/lambda_data/exr_hist_dict_20_bins.npy"
output_path="/Users/yaelvinker/PycharmProjects/lab/run_trained_model/output/"

# lambda calc params
mean_hist_path="/Users/yaelvinker/PycharmProjects/lab/run_trained_model/lambda_data/ldr_avg_hist_900_images_20_bins.npy"
lambda_output_path="/Users/yaelvinker/PycharmProjects/lab/run_trained_model/lambda_data"
bins=20
#'belgium': 6539.507650375366
python3.6 run_trained_model.py --model_name $model_name \
  --input_images_path $input_images_path --output_path $output_path --model_path $model_path \
  --f_factor_path $f_factor_path \
  --mean_hist_path $mean_hist_path --lambda_output_path $lambda_output_path --bins $bins