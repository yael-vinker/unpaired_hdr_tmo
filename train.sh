#!/bin/csh

source /cs/labs/raananf/yael_vinker/my_venv/bin/activate.csh

python3.6 -W ignore -u main_train.py --change_random_seed $1 \
  --batch_size $2 --num_epochs $3 \
  --G_lr $4 --D_lr $5 --model $6 \
  --con_operator $7 --unet_norm $8 --last_layer $9 \
  --use_xaviar $10 --ssim_loss_factor $11 --pyramid_loss $12 \
  --pyramid_weight_list $13 \
  --data_root_npy $14 --data_root_ldr $15 \
  --test_dataroot_npy $16 --test_dataroot_original_hdr $17 \
  --test_dataroot_ldr $18 --result_dir_prefix $19 --use_factorise_data $20 \
  --add_clipping $21 --use_normalization $22 --factor_coeff $23
