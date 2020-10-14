#!/bin/csh

source /cs/labs/raananf/yael_vinker/python7-venv/bin/activate.csh

echo "func_to_run $1"
echo "model_name $2"
echo "input_path $3"
echo "output_name $4"
echo "model_path $5"
echo "f_factor_path $6"
echo "test_mode_f_factor $7"
echo "test_mode_frame $8"
echo "input_images_names_path $9"

python3.7 -W ignore -u image_quality_assessment_util.py \
  --func_to_run $1 \
  --model_name $2 \
  --input_path $3 \
  --output_name $4 \
  --model_path $5 \
  --f_factor_path $6 \
  --test_mode_f_factor $7 \
  --test_mode_frame $8 \
  --input_images_names_path $9