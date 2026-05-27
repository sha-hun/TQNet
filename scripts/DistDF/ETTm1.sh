
seq_len=96

for pred_len in 96 192 336 720
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
      --e_layers 3 \
      --n_heads 16 \
      --d_model 24 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 4 \
      --stride 4 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --learning_rate 0.0001 \
      --lradj TST \
      --pct_start 0.4 \
      --train_epochs 100 \
      --patience 10 \
      --batch_size 128 \
      --itr 1 \
      --random_seed $random_seed \
      --rec_lambda 0.5 \
      --auxi_lambda 0.5 \
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