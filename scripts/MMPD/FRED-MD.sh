mkdir -p ./out/logs

backbone='Decoder'
loss='MMPD'
data='FRED-MD'
patch_size=12

for pred_len in 24  36  48  60; do
    log_time=$(date +"%y-%m-%d-%H-%M-%S")

    python -u main_mmpd.py  --data $data --backbone $backbone --loss_func $loss --seq_len 336 --pred_len $pred_len \
        --training True \
        --patch_size $patch_size --d_layers 2 --d_model 256 --d_ff 512 --n_heads 4 \
        --weighted True --point_weight 0.01 \
        --d_diffusion 256 --diffusion_layers 1 --radius 3 --max_diffusion_steps 1000 --beta_schedule linear \
        --batch_size 32 --learning_rate 1e-4 --lradj cosine --train_epochs 20 --patience 5 --gpu 0 \
        --testing False \
        > ./out/logs/Time${log_time}_Train_data${data}_il336_ol${pred_len}_backbone${backbone}_loss${loss}.log 2>&1
done