#!/bin/csh

source /cs/labs/raananf/yael_vinker/python7-venv/bin/activate.csh

echo "im_path $1"
echo "full_size_im $2"
echo "num_steps $3"
echo "output_path $4"
echo "wind_size $5"
echo "pyramid_weight_list $6"
echo "apply_sig_mu_ssim $7"
echo "struct_method $8"
echo "std_norm_factor $9"
echo "std_pyramid_weight_list ${10}"
echo "intensity_epsilon ${11}"
echo "alpha ${12}"
echo "std_method ${13}"
echo "blf_alpha ${14}"
echo "mu_pyramid_weight_list ${15}"


python3.7 -W ignore -u single_image_trainer.py \
  --im_path $1 \
  --full_size_im $2 \
  --num_steps $3 \
  --output_path $4 \
  --wind_size $5 \
  --pyramid_weight_list $6 \
  --apply_sig_mu_ssim $7 \
  --struct_method $8 \
  --std_norm_factor $9 \
  --std_pyramid_weight_list ${10} \
  --intensity_epsilon ${11} \
  --alpha ${12} \
  --std_method ${13} \
  --blf_alpha ${14} \
  --mu_pyramid_weight_list ${15}