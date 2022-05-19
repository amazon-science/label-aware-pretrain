#!/bin/bash

#SBATCH --job-name=MT5-base-finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path ../output/t5-cpt-gold/epoch1 \
    --do_train \
    --do_eval \
    --task translation_src_to_tgt \
    --train_file ../data/snips/fewshot_ft/train_32_examples.json \
    --validation_file ../data/snips/snips_dev.json \
    --output_dir ../output/t5-cpt-gold/epoch1	 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --predict_with_generate \
    --learning_rate 5e-4 \
    --ignore_data_skip \
    --no_save \
    --separate_preds_by_train \
    --save_steps 900000 \
    --num_train_epochs 2