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

# baseline testing
#data_path="btcusdt_pca_components_12h_4h_72_07_05_baseline.csv"
# data_path="btcusdt_pca_components_12h_4h_40_3558_282_no_l_juice_testing.csv"
#data_path="btcusdt_pca_components_12h_4h_70_3558_282_04_04.csv"
#data_path="btcusdt_pca_components_12h_4h_50_7_5_bitstamp.csv"
#data_path="btcusd_pca_components_12h_1d_70_7_5_bitsap.csv"
#data_path="btcusd_pca_components_12h_70_7_5_bitsap_1_4_2.csv"
#data_path="btcusd_pca_components_12h_reduced_4h_53_7_5_1_2_1.csv"
#data_path="btcusdt_pca_components_12h_60_07_05.csv"
#data_path="btcusd_pca_components_lightboost_12h_4h_reduced_60_7_5_1_2_1_old.csv"
data_path="btcusd_pca_components_lightboost_12h_4h_reduced_70_7_5_1_2_1_old.csv"
seq_len=96
pred_len=1
#enc_in=65
enc_in=75
d_model=512
data_file=$(basename "$data_path" .csv)
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/logits/ \
  --data_path $data_path \
  --model_id "2_${data_file}_${seq_len}_${pred_len}_${enc_in}" \
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
