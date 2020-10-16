#!/bin/bash
input_dir_hdrplus="/cs/labs/raananf/yael_vinker/dng_collection"
output_dir_pref="/cs/snapless/raananf/yael_vinker/data/new_data"
isLdr=0
number_of_images=1000
use_factorise_data=1
factor_coeff=1
use_new_f=1
data_trc="min_log"
crop_data=0
inpue_names_path_dir="/cs/labs/raananf/yael_vinker/data/new_lum_est_hist/dng_name_split"
names=("dng[0_100]")
for ((i = 0; i < ${#names[@]}; ++i)); do
  cur_name="${names[i]}"
	input_dir_names="$inpue_names_path_dir/$cur_name.npy"
	echo "$inpue_names_path"
	sbatch --mem=8000m -c1 --time=2-0 --gres=gpu:1 create_data.sh $input_dir_hdrplus \
    $output_dir_pref $isLdr $number_of_images $use_factorise_data $factor_coeff \
    $use_normalization $use_new_f $data_trc $crop_data $input_dir_names
done


