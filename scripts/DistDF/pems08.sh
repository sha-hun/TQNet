model_name=YourModelName

root_path_name=./dataset/PEMS/
data_path_name=PEMS08.npz
model_id_name="PEMS08_compare"
data_name=PEMS

seq_len=96

# 注意：PEMS数据集的预测长度是 12 24 36 48
for pred_len in 12 24 36 48
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
      --enc_in 170 \
      --dec_in 170 \
      --c_out 170 \
      --factor 3 \
      --learning_rate 0.002 \
      --lradj type1 \
      --train_epochs 100 \
      --patience 15 \
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
      --alpha 0.05 \
      --d_model 128 \
      --d_ff 128 \
      --ca_layers 1 \
      --pd_layers 1 \
      --ia_layers 1 \
      --num_p 12 \
      --n_heads 8 \
      --period 48 \
      --attn_dropout 0.15 \
      --stable_len 6 \
      --dropout 0.0 \
      --var_weight 0.01
done
done