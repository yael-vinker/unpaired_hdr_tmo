#!/bin/bash
input_dir_hdrplus="/cs/labs/raananf/yael_vinker/dng_collection"
output_dir_pref="/cs/snapless/raananf/yael_vinker/data/new_data"
isLdr=0
number_of_images=1000
use_factorise_data=1
factor_coeff=1

sbatch --mem=8000m -c1 --time=2-0 --gres=gpu:2 create_data.sh $input_dir_hdrplus \
  $output_dir_pref $isLdr $number_of_images $use_factorise_data $factor_coeff \
  $use_normalization
#
#input_dir_ldr="/cs/dataset/flickr30k/images"
#output_dir_pref="/cs/snapless/raananf/yael_vinker/data/new_data"
#isLdr=1
#number_of_images=1000
#use_factorise_data=1
#factor_coeff=1
#use_normalization=0
#echo "input_dir $input_dir_ldr"
#echo "output_dir_pref $output_dir_pref"
#echo "isLdr $isLdr"
#echo "number_of_images $number_of_images"
#echo "use_factorise_datasoph $use_factorise_data"
#echo "factor_coeff $factor_coeff"
#echo "use_normalization $use_normalization"
#sbatch --mem=4000m -c1 --time=2-0 --gres=gpu:1 create_data.sh $input_dir_hdrplus \
#  $output_dir_pref $isLdr $number_of_images $use_factorise_data $factor_coeff \
#  $use_normalization
#
##!/bin/bash
#input_dir_hdrplus="/cs/labs/raananf/yael_vinker/dng_collection"
#output_dir_pref="/cs/snapless/raananf/yael_vinker/data/new_data"
#isLdr=0
#number_of_images=1000
#use_factorise_data=0
#factor_coeff=1
#use_normalization=0
#echo "input_dir $input_dir_hdrplus"
#echo "output_dir_pref $output_dir_pref"
#echo "isLdr $isLdr"
#echo "number_of_images $number_of_images"
#echo "use_factorise_datasoph $use_factorise_data"
#echo "factor_coeff $factor_coeff"
#echo "use_normalization $use_normalization"
#sbatch --mem=4000m -c1 --time=2-0 --gres=gpu:1 create_data.sh $input_dir_hdrplus \
#  $output_dir_pref $isLdr $number_of_images $use_factorise_data $factor_coeff \
#  $use_normalization
#
#input_dir_ldr="/cs/dataset/flickr30k/images"
#output_dir_pref="/cs/snapless/raananf/yael_vinker/data/new_data"
#isLdr=1
#number_of_images=1000
#use_factorise_data=0
#factor_coeff=1
#use_normalization=0
#echo "input_dir $input_dir_ldr"
#echo "output_dir_pref $output_dir_pref"
#echo "isLdr $isLdr"
#echo "number_of_images $number_of_images"
#echo "use_factorise_datasoph $use_factorise_data"
#echo "factor_coeff $factor_coeff"
#echo "use_normalization $use_normalization"
#sbatch --mem=4000m -c1 --time=2-0 --gres=gpu:1 create_data.sh $input_dir_hdrplus \
#  $output_dir_pref $isLdr $number_of_images $use_factorise_data $factor_coeff \
#  $use_normalization