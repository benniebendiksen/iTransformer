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



# data_path="btcusdt_pca_components_12h_1d_53_07_05.csv"
# juice-testing
#data_path="btcusdt_pca_components_12h_4h_72_3558_282_juice_testing.csv"
#data_path="btcusdt_pca_components_12h_4h_50_3558_282_no_l_juice_testing.csv"
#data_path="btcusdt_pca_components_12h_4h_70_3558_282_04_04.csv_dud"

data_path="btcusdt_pca_components_12h_4h_57_7_5_bitstamp.csv"
seq_len=96
pred_len=1
enc_in=62
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
#data_path="btcusdt_pca_components_12h_4h_50_3558_282_04_04_two.csv"
#seq_len=96
#pred_len=1
#enc_in=55
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
#  --freq 12h
#
#seq_len=96
#pred_len=1
#enc_in=55
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
#seq_len=96
#pred_len=1
#enc_in=55
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
