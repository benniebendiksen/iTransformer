#!/bin/bash

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate gpu_env_copy

# Debugging: Print the active environment and Python path
echo "Activated Conda environment: $(conda info --envs | grep \*)"
echo "Python path: $(which python)"
python -c "import sys; print('Python version:', sys.version)"
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import torch; print('Torch CUDA Available:', torch.cuda.is_available())"

model_name=iTransformer

# baseline re-testing
#data_path="btcusd_pca_components_lightboost_12h_4h_reduced_60_7_5_1_2_1_old.csv"
#data_path="btcusd_pca_components_lightboost_12h_4h_reduced_70_7_5_1_2_1_old.csv"
#data_path="btcusd_pca_components_lightboost_12h_4h_reduced_reattempt_70_7_5_1_2_1_old.csv"
#data_path="btcusd_pca_components_lightboost_12h_4h_reduced_70_7_5_1_2_1_old_baseline.csv"
#data_path="pca_components_btcusdt_70_april_15_reduced.csv"
#data_path="pca_components_btcusdt_70_april_15_reduced_extended_14_fixed_sizes.csv"
#data_path="pca_components_btcusdt_68_april_15_reduced_extended_28_fixed_sizes.csv"
#data_path="pca_components_btcusdt_70_april_15_reduced_extended_14_double_fixed_sizes.csv"
# data_path="pca_components_btcusdt_70_april_15_to_date_fixed_train_val_size.csv"

#data_path="btcusd_pca_components_lightboost_12h_4h_reduced_70_7_5_1_2_1_old_reordered.csv"
#data_path="pca_components_btcusdt_40_reattempt_12h_80_top.csv"
#data_path="pca_components_btcusdt_42_12h_full_binance_reduced.csv"
# data_path="btcusdt_12h_historical_reduced_python_processed_1_2_1_old_reattempt_corr_removed.csv"
#data_path="pca_components_btcusdt_12h_56_reduced.csv"
#data_path="pca_components_btcusdt_12h_55_reduced_attempt_2.csv"
#data_path="pca_components_btcusdt_12h_45_reduced_lance_seed_april_15.csv"
#data_path="pca_components_btcusdt_12h_44_reduced_lance_seed_april_15_baseline_set_sizes.csv"
#data_path="pca_components_btcusdt_12h_48_07_05_reduced_lance_seed_april_15.csv"
#data_path="pca_components_btcusdt_12h_46_07_05_lance_seed_april_15.csv"
#data_path="pca_components_btcusdt_12h_45_reduced_lance_seed_2.csv"
#data_path="pca_components_btcusdt_4h_43_07_05_lance_seed_march_9_2020.csv"
data_path="pca_components_btcusdt_4h_48_lance_seed_march_9_2020.csv"

seq_len=96
pred_len=1
# enc_in=73
enc_in=53
d_model=512
data_file=$(basename "$data_path" .csv)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/logits/ \
  --data_path $data_path \
  --model_id "${data_file}_${seq_len}_${pred_len}_${enc_in}" \
  --model $model_name \
  --data logits \
  --features MS \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 4 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out 1 \
  --des 'Logits' \
  --d_model $d_model \
  --d_ff $d_model \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 5 \
  --train_epochs 50 \
  --patience 7 \
  --exp_name logits \
  --target close \
  --is_shorting 1 \
  --precision_factor 2.0 \
  --auto_weight 1 \
  --freq 12h \

  exit 0

data_path="btcusd_pca_components_lightboost_12h_4h_reduced_60_7_5_1_2_1_old.csv"
seq_len=96
pred_len=1
enc_in=61
d_model=512
data_file=$(basename "$data_path" .csv)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/logits/ \
  --data_path $data_path \
  --model_id "1_${data_file}_${seq_len}_${pred_len}_${enc_in}" \
  --model $model_name \
  --data logits \
  --features MS \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 4 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out 1 \
  --des 'Logits' \
  --d_model $d_model \
  --d_ff $d_model \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 5 \
  --train_epochs 50 \
  --patience 7 \
  --exp_name logits \
  --target close \
  --is_shorting 1 \
  --precision_factor 2.0 \
  --auto_weight 1 \
  --freq 12h \

#
#seq_len=96
#pred_len=1
#enc_in=45
#d_model=512
#data_file=$(basename "$data_path" .csv)
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/logits/ \
#  --data_path $data_path \
#  --model_id "1_${data_file}_${seq_len}_${pred_len}_${enc_in}" \
#  --model $model_name \
#  --data logits \
#  --features MS \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --e_layers 4 \
#  --enc_in $enc_in \
#  --dec_in $enc_in \
#  --c_out 1 \
#  --des 'Logits' \
#  --d_model $d_model \
#  --d_ff $d_model \
#  --batch_size 32 \
#  --learning_rate 0.001 \
#  --itr 5 \
#  --train_epochs 50 \
#  --patience 7 \
#  --exp_name logits \
#  --target close \
#  --is_shorting 1 \
#  --precision_factor 2.0 \
#  --auto_weight 1 \
#  --freq 12h \
#  --dropout 0.15 \
#
#
#seq_len=96
#pred_len=1
#enc_in=45
#d_model=512
#data_file=$(basename "$data_path" .csv)
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/logits/ \
#  --data_path $data_path \
#  --model_id "1_${data_file}_${seq_len}_${pred_len}_${enc_in}" \
#  --model $model_name \
#  --data logits \
#  --features MS \
#  --seq_len $seq_len \
#  --pred_len $pred_len \
#  --e_layers 4 \
#  --enc_in $enc_in \
#  --dec_in $enc_in \
#  --c_out 1 \
#  --des 'Logits' \
#  --d_model $d_model \
#  --d_ff $d_model \
#  --batch_size 32 \
#  --learning_rate 0.001 \
#  --itr 5 \
#  --train_epochs 50 \
#  --patience 7 \
#  --exp_name logits \
#  --target close \
#  --is_shorting 1 \
#  --precision_factor 2.0 \
#  --auto_weight 1 \
#  --freq 12h \
#  --dropout 0.2 \
