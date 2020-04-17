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

echo "====== LOSS ======"
echo "loss_g_d_factor $9"
echo "train_with_D ${10}"
echo "ssim_loss_factor ${11}"
echo "pyramid_weight_list ${12}"
echo "apply_intensity_loss ${13}"
echo "intensity_epsilon ${14}"
echo "std_pyramid_weight_list ${15}"
echo "mu_loss_factor ${16}"
echo "mu_pyramid_weight_list ${17}"

echo "====== DATASET ======"
echo "data_root_npy ${18}"
echo "data_root_ldr ${19}"
echo "test_dataroot_npy ${20}"
echo "test_dataroot_original_hdr ${21}"
echo "test_dataroot_ldr ${22}"
echo "result_dir_prefix ${23}"
echo "use_factorise_data ${24}"
echo "factor_coeff ${25}"
echo "add_clipping ${26}"
echo "use_normalization ${27}"
echo "normalization ${28}"
echo "last_layer ${29}"
echo "d_model ${30}"
echo "d_down_dim ${31}"
echo "d_norm ${32}"
echo "milestones ${33}"
echo "add_frame ${34}"
echo "input_dim ${35}"


python3.6 -W ignore -u main_train.py \
  --change_random_seed $1 \
  --batch_size $2 \
  --num_epochs $3 \
  --G_lr $4 \
  --D_lr $5 \
  --model $6 \
  --con_operator $7 \
  --use_xaviar $8 \
  --loss_g_d_factor $9 \
  --train_with_D ${10} \
  --ssim_loss_factor ${11} \
  --pyramid_weight_list ${12} \
  --apply_intensity_loss ${13} \
  --intensity_epsilon ${14} \
  --std_pyramid_weight_list ${15} \
  --mu_loss_factor ${16} \
  --mu_pyramid_weight_list ${17} \
  --data_root_npy ${18} \
  --data_root_ldr ${19} \
  --test_dataroot_npy ${20} \
  --test_dataroot_original_hdr ${21} \
  --test_dataroot_ldr ${22} \
  --result_dir_prefix ${23} \
  --use_factorise_data ${24} \
  --factor_coeff ${25} \
  --add_clipping ${26} \
  --use_normalization ${27} \
  --normalization ${28} \
  --last_layer ${29} \
  --d_model ${30} \
  --d_down_dim ${31} \
  --d_norm ${32} \
  --milestones ${33} \
  --add_frame ${34} \
  --input_dim ${35}

