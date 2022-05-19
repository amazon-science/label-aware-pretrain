#!/bin/bash

#SBATCH --job-name=T5-base-cpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

conda activate py38

python ../models/run_seq2seq.py \
    --model_name_or_path '../output/t5-ic-goldsilver/epoch2/' \
    --do_eval \
    --task translation_src_to_tgt \
    --train_file ../data/c4/c4_train_filter.intgen.json \
    --validation_file ../data/c4/c4_train_filter.intgen.json \
    --output_dir ../output/t5-ic-goldsilver/epoch2/	 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --predict_with_generate \
    --get_confidence \
    --overwrite_output_dir