seq_len=96
for pred_len in 96 192 336 720 12 24 36 48 60
do
for random_seed in 2024
do
    python -u run.py \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 1024 \
      --d_ff 2024 \
      --dropout 0.5 \
      --fc_dropout 0.25 \
      --patch_len 48 32 16 6 3 \
      --des Exp \
      --train_epochs 50 \
      --patience 5 \
      --top_k 5 \
      --batch_size 16 \
  done
done


