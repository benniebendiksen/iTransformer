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

# Configuration for price change forecasting with trading strategy:
# - features=MS (multivariate input, univariate output)
# - seq_len=96 (24 hours with 15-minute intervals)
# - pred_len=4 (predict 4 timesteps ahead, 1 hour)
# - exp_name=crypto (use our custom experiment class)
# - is_shorting=0 (do not use shorting strategy, only go long)
# - precision_factor=2.0 (focus on precision for long-only strategy)
#   When precision_factor > 1.0:
#     False positives (incorrectly predicting price increases) are penalized precision_factor times more heavily
#     False negatives (missing actual price increases) are penalized precision_factor times less
# - auto_weight=1 (automatically adjust weights based on class distribution)
#   When auto_weight=1 (enabled):
#     The loss function calculates the ratio of negative:positive examples in each batch
#     It applies a ratio as a weight to positive examples
#     For example, if there are 24 decreases and 8 increases in a batch, each increase gets weighted 3x more

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/logits/ \
  --data_path btcusdc_pca_components_1h_41.csv \
  --model_id crypto_96_4_noshort \
  --model $model_name \
  --data crypto \
  --features MS \
  --seq_len 24 \
  --label_len 48 \
  --pred_len 4 \
  --e_layers 4 \
  --enc_in 44 \
  --dec_in 44 \
  --c_out 1 \
  --des 'Crypto' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 5 \
  --train_epochs 50 \
  --patience 5 \
  --exp_name crypto \
  --target close \
  --is_shorting 1 \
  --precision_factor 2.0 \
  --auto_weight 1

# Alternative configuration with shorting enabled
python -u run.py \
   --is_training 1 \
   --root_path ./dataset/logits/ \
   --data_path btcusdc_pca_components_1h_52.csv \
   --model_id crypto_96_4_short \
   --model $model_name \
   --data crypto \
   --features MS \
   --seq_len 96 \
   --label_len 48 \
   --pred_len 4 \
   --e_layers 4 \
   --enc_in 44 \
   --dec_in 44 \
   --c_out 1 \
   --des 'Crypto' \
   --d_model 512 \
   --d_ff 512 \
   --batch_size 32 \
   --learning_rate 0.001 \
   --itr 5 \
   --train_epochs 15 \
   --patience 5 \
   --exp_name crypto \
   --target close \
   --is_shorting 1 \
   --auto_weight 1

python -u run.py \
   --is_training 1 \
   --root_path ./dataset/logits/ \
   --data_path btcusdc_pca_components_1h_53.csv \
   --model_id crypto_96_4_short \
   --model $model_name \
   --data crypto \
   --features MS \
   --seq_len 96 \
   --label_len 48 \
   --pred_len 4 \
   --e_layers 4 \
   --enc_in 44 \
   --dec_in 44 \
   --c_out 1 \
   --des 'Crypto' \
   --d_model 512 \
   --d_ff 512 \
   --batch_size 32 \
   --learning_rate 0.001 \
   --itr 5 \
   --train_epochs 15 \
   --patience 5 \
   --exp_name crypto \
   --target close \
   --is_shorting 1 \
   --auto_weight 1

python -u run.py \
   --is_training 1 \
   --root_path ./dataset/logits/ \
   --data_path btcusdc_pca_components_1h_53.csv \
   --model_id crypto_96_4_short \
   --model $model_name \
   --data crypto \
   --features MS \
   --seq_len 48 \
   --label_len 48 \
   --pred_len 4 \
   --e_layers 4 \
   --enc_in 44 \
   --dec_in 44 \
   --c_out 1 \
   --des 'Crypto' \
   --d_model 512 \
   --d_ff 512 \
   --batch_size 32 \
   --learning_rate 0.001 \
   --itr 5 \
   --train_epochs 15 \
   --patience 5 \
   --exp_name crypto \
   --target close \
   --is_shorting 1 \
   --auto_weight 1

python -u run.py \
   --is_training 1 \
   --root_path ./dataset/logits/ \
   --data_path btcusdc_1h_historical.csv \
   --model_id crypto_96_4_short \
   --model $model_name \
   --data crypto \
   --features MS \
   --seq_len 48 \
   --label_len 48 \
   --pred_len 4 \
   --e_layers 4 \
   --enc_in 44 \
   --dec_in 44 \
   --c_out 1 \
   --des 'Crypto' \
   --d_model 512 \
   --d_ff 512 \
   --batch_size 32 \
   --learning_rate 0.001 \
   --itr 5 \
   --train_epochs 15 \
   --patience 5 \
   --exp_name crypto \
   --target close \
   --is_shorting 1 \
   --auto_weight 1