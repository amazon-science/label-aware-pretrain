#!/bin/bash

python ../models/run_cpt.py \
    --model_name_or_path 't5-base' \
    --do_train \
    --do_eval \
    --task translation_src_to_tgt \
    --train_file $1 \
    --validation_file /efs/aarmuell/LexLabSemPretrain/data/top/topv2.sub.tok.json \
    --output_dir $2	 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --overwrite_output_dir \
    --learning_rate 5e-4 \
    --logging_steps 250 \
    --evaluation_strategy steps \
    --num_train_epochs 4 \
    --save_total_limit 6 \
    --save_strategy epoch