#!/bin/csh

source /cs/labs/raananf/yael_vinker/my_venv/bin/activate.csh

echo "input_dir $1"
echo "output_dir_pref $2"
echo "isLdr $3"
echo "number_of_images $4"
echo "use_factorise_datasoph $5"
echo "factor_coeff $6"
echo "use_new_f $7"
echo "data_trc $8"

python3.6 create_dng_npy_data.py \
  --input_dir $1 \
  --output_dir_pref $2 \
  --isLdr $3 \
  --number_of_images $4 \
  --use_factorise_data $5 \
  --factor_coeff $6 \
  --use_new_f $7 \
  --data_trc $8
