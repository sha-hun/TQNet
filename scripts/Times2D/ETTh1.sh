for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 64 \
    --d_ff 64 \
    --dropout 0.5 \
    --fc_dropout 0.25 \
    --patch_len 48 32 16 6 3 \
    --des Exp \
    --top_k 5 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0001 \
  done
done