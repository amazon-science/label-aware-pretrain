#!/bin/bash

#SBATCH --job-name=MT5-base-finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

conda activate py38

python ../models/run_classification.py \
    --model_name_or_path 'roberta-base' \
    --do_train \
    --do_eval \
    --train_file ../data/classifier/concat_train.json \
    --validation_file ../data/classifier/concat_dev.json \
    --output_dir ../output/roberta-dialogue-act	 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --logging_steps 593 \
    --evaluation_strategy steps \
    --save_strategy epoch \
    --num_train_epochs 8