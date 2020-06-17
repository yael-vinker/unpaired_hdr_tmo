#!/bin/bash

#input_format="exr"
input_format="npy"
#images_source="open_exr_exr_format"
images_source="npy_pth"
arch_dir="/cs/labs/raananf/yael_vinker/05_15/results_05_15"
models_epoch=320
output_dir_name="_320_dng"

sbatch --mem=8000m -c1 --gres=gpu:1 --time=1-0 run_model.sh \
  $input_format $images_source $arch_dir $models_epoch $output_dir_name
