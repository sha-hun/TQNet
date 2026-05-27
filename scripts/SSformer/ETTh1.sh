seq_len=96
for pred_len in 96 192 336 720  12 24 36 48 60
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \

      --label_len 96 \
      --pred_len 96 \
      --e_layers 2 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model 16 \
      --d_ff 32 \
      --topk 5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 
done
done