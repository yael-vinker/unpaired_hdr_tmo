#!/bin/csh

source /cs/labs/raananf/yael_vinker/my_venv/bin/activate.csh

echo "input_format $1"
echo "images_source $2"
echo "arch_dir $3"
echo "models_epoch $4"
echo "output_dir_name $5"

python3.6 -W ignore -u model_save_util.py \
  --input_format $1 --images_source $2 --arch_dir $3 --models_epoch $4 --output_dir_name $5
