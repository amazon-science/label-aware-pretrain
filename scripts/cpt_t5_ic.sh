#!/bin/bash

#SBATCH --job-name=T5-base-cpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path 't5-base' \
    --do_train \
    --do_eval \
    --task translation_src_to_tgt \
    --train_file ../data/cs_twitter/internal_wikihow_twcs.json \
    --validation_file ../data/top/topv2.sub.json \
    --output_dir ../output/t5-ic-smallnoisy/	 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --overwrite_output_dir \
    --learning_rate 5e-4 \
    --logging_steps 4000 \
    --evaluation_strategy steps \
    --num_train_epochs 10 \
    --save_total_limit 40 \
    --save_strategy epoch
