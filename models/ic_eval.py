import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dev_file', type=str, help="Data on which we evaluate.")
parser.add_argument('model_dir', type=str, help="Directory containing eval results.")

args = parser.parse_args()

random_seeds = (42,)

scores_across_seeds = []
for seed in random_seeds:
    scores_file = os.path.basename(args.dev_file).replace(".json", "") + \
        "_seed" + str(seed) + \
        ".eval_results_seq2seq.txt"
    scores_file = os.path.join(args.model_dir, scores_file)

with open(scores_file, 'r') as scores:
    exact_match_str = scores.readline()
    exact_match_score = float(exact_match_str.split(":")[1].strip())
    scores_across_seeds.append(exact_match_score)