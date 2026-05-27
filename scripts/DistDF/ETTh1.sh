

seq_len=96

for pred_len in 96 192 336 720 12 24 36 48 60
do
for random_seed in 2024
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 24 \
      --d_ff 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --patch_len 4 \
      --stride 4 \
      --learning_rate 0.0001 \
      --lradj type3 \
      --train_epochs 30 \
      --patience 5 \
      --batch_size 128 \
      --itr 1 \
      --random_seed $random_seed \
      --rec_lambda 0.9 \
      --auxi_lambda 0.1 \
      --cf_dim 128 \
      --cf_depth 2 \
      --cf_heads 8 \
      --cf_mlp 96 \
      --cf_head_dim 32 \
      --use_nys 0 \
      --individual 0 \
      --var_weight 0.002
done
done