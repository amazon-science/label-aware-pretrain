#!/bin/bash

#SBATCH --job-name=MT5-base-finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

conda activate py38

python ../models/run_classification.py \
    --model_name_or_path '../output/roberta-dialogue-act/checkpoint-17790/' \
    --do_predict \
    --train_file ../data/c4/c4_train_filter.json \
    --validation_file ../data/c4/c4_train_filter.json \
    --test_file ../data/c4/c4_train_filter.json \
    --output_dir ../output/roberta-dialogue-act	 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --logging_steps 593