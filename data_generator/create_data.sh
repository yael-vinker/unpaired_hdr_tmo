#!/bin/csh

source /cs/labs/raananf/yael_vinker/my_venv/bin/activate.csh

echo $1 $2 $3 $4 $5 $6 $7
python3.6 create_dng_npy_data.py \
  --input_dir $1 \
  --output_dir_pref $2 \
  --isLdr $3 \
  --number_of_images $4 \
  --use_factorise_data $5 \
  --factor_coeff $6 \
  --use_normalization $7
