seq_len=96
for pred_len in 96 192 336 720 12 24 36 48 60
do
for random_seed in 2024
do
    python -u run.py \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.5 \
      --fc_dropout 0.25 \
      --patch_len 48 32 16 6 3 \
      --des Exp \
      --train_epochs 50 \
      --patience 5 \
      --top_k 5 \
done
done




# ==============================
# Loop over sequence & pred lengths
# ==============================
for seq_len in "${seq_len_list[@]}"; do

  # Create folder for this sequence length
  seq_log_dir="./logs/LongForecasting/${data_name}/${seq_len}"
  mkdir -p "$seq_log_dir"

  for pred_len in "${pred_len_list[@]}"; do

    echo "Running ${data_name} with seq_len=${seq_len}, pred_len=${pred_len} ..."

    # Run experiment and save log inside seq_len folder
    python -u run.py \
      --random_seed $random_seed \
      --task_name $task_name \
      --model_id ${model_name}_${data_name}_seq${seq_len}_pred${pred_len} \
      --is_training 1 \
      --model $model_name \
      --data $data_name \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \

      --batch_size 16 \
      --learning_rate $learning_rate \
      --itr 1 \
      > "${seq_log_dir}/${model_name}_${data_name}_seq${seq_len}_pred${pred_len}.log"

    wait
  done
done
