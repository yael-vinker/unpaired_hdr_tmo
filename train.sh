#!/bin/csh

source /cs/labs/raananf/yael_vinker/my_venv/bin/activate.csh

echo "change_random_seed $1"
echo "batch_size $2"
echo "num_epochs $3"
echo "G_lr $4"
echo "D_lr $5"
echo "model $6"
echo "con_operator $7"
echo "use_xaviar $8"
echo "ssim_loss_factor $9"
echo "pyramid_weight_list $10"
echo "pyramid_pow $23"
echo "data_root_npy $11"
echo "data_root_ldr $12"
echo "test_dataroot_npy $13"
echo "test_dataroot_original_hdr $14"
echo "test_dataroot_ldr $15"
echo "result_dir_prefix $16"
echo "use_factorise_data $17"
echo "factor_coeff $18"
echo "add_clipping $19"
echo "use_normalization $20"
echo "normalization $21"
echo "last_layer $22"
echo "d_model $24"
echo "d_down_dim $25"
echo "pyramid_loss $26"
echo "d_norm $27"
echo "use_sigma_loss $28"
echo "use_c3_in_ssim $29"


python3.6 -W ignore -u main_train.py \
  --change_random_seed $1 \
  --batch_size $2 \
  --num_epochs $3 \
  --G_lr $4 \
  --D_lr $5 \
  --model $6 \
  --con_operator $7 \
  --use_xaviar $8 \
  --ssim_loss_factor $9 \
  --pyramid_weight_list $10 \
  --data_root_npy $11 \
  --data_root_ldr $12 \
  --test_dataroot_npy $13 \
  --test_dataroot_original_hdr $14 \
  --test_dataroot_ldr $15 \
  --result_dir_prefix $16 \
  --use_factorise_data $17 \
  --factor_coeff $18 \
  --add_clipping $19 \
  --use_normalization $20 \
  --normalization $21 \
  --last_layer $22 \
  --pyramid_pow $23 \
  --d_model $24 \
  --d_down_dim $25 \
  --pyramid_loss $26 \
  --d_norm $27 \
  --use_sigma_loss $28 \
  --use_c3_in_ssim $29