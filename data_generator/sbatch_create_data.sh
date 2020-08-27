#!/bin/bash
input_dir_hdrplus="/cs/labs/raananf/yael_vinker/dng_collection"
output_dir_pref="/cs/snapless/raananf/yael_vinker/data/new_data"
isLdr=0
number_of_images=1000
use_factorise_data=1
factor_coeff=1
use_new_f=1
data_trc="min_log"

sbatch --mem=8000m -c1 --time=2-0 --gres=gpu:2 create_data.sh $input_dir_hdrplus \
  $output_dir_pref $isLdr $number_of_images $use_factorise_data $factor_coeff \
  $use_normalization $use_new_f $data_trc
