
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
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --factor 3 \
      --learning_rate 0.0005 \
      --lradj type1 \
      --train_epochs 100 \
      --patience 5 \
      --batch_size 8 \
      --itr 1 \
      --random_seed $random_seed \
      --rec_lambda 0.995 \
      --auxi_lambda 0.005 \
      --joint_forecast 1 \
      --ot_type upper_bound \
      --normalize 1 \
      --distance wasserstein_empirical_per_dim \
      --mask_factor 0.0 \
      --reg_sk 0.005 \
      --eps 1e-9 \
      --alpha 0.35 \
      --d_model 512 \
      --d_ff 512 \
      --ca_layers 3 \
      --pd_layers 1 \
      --ia_layers 1 \
      --num_p 8 \
      --n_heads 64 \
      --period 24 \
      --attn_dropout 0.15 \
      --stable_len 2 \
      --dropout 0.0 \
      --var_weight 0.002
done
done