model_name=ZWF_final_v1

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name="5-15-v1"
data_name=ETTh1

seq_len=96
for pred_len in 96 192 336 720 12 24 36 48 60
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --features M \
      --cycle 24 \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --num_seasonal_components 2 \
      --max_season_length 12 \
      --trend_length 36 \
      --hidden_dim 64 \
      --hidden_dim_mask 64 \
      --d_model 256 \
      --itr 1 --batch_size 2 --learning_rate 0.0005 --random_seed $random_seed
done
done


