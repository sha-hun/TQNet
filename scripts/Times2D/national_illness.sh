seq_len=96
for pred_len in 96 192 336 720 12 24 36 48 60
do
for random_seed in 2024
do
    python -u run.py \
      --enc_in 7 \
      --e_layers 2 \
      --n_heads 4 \
      --d_model 64 \
      --d_ff 64 \
      --dropout 0.5 \
      --fc_dropout 0.25 \
      --patch_len 48 32 16 6 3 \
      --des Exp \
      --train_epochs 50 \
      --patience 5 \
      --top_k 5 \
  done
done


