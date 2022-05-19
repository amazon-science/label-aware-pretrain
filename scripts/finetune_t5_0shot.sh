#!/bin/bash

#SBATCH --job-name=MT5-base-finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path "../output/t5-multichoice-gold/epoch4/" \
    --do_eval \
    --task translation_src_to_tgt \
    --train_file ../data/snips/snips_dev.json \
    --validation_file ../data/top/low_resource_splits/weather_valid_10spis_ft.multichoice.json \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=32 \
    --output_dir "../output/t5-multichoice-gold/" \
    --overwrite_output_dir \
    --predict_with_generate \
    --separate_preds_by_train \
    --save_steps 900000 \
    --num_train_epochs 1