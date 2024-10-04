if [ ! -d "./res" ]; then
    mkdir ./res/
fi

if [ ! -d "./res/longforecasting" ]; then
    mkdir ./res/longforecasting/
fi

model_name=SPACE
seq_len=96

for pred_len in 96 192 336 720
do
    python3 -u run_longExp.py \
        --task_name long_term_forecast \
        --model $model_name \
        --is_training 1 \
        --root_path ./data/ETT/ \
        --data_path ETTm1.csv \
        --data ETTm1 \
        --features M \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 512 \
        --d_mutual 256 \
        --lradj 'TST' \
        --criterion 'mae' \
        --batch_size 128 \
        --train_epochs 30 \
        --learning_rate 0.0005 \
        --patience 5 \
        --itr 1 > res/longforecasting/$model_name'_ETTm1_'$pred_len'pl'.log
done
