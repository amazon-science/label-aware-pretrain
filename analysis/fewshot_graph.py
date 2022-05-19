import argparse
import math
import os
import itertools
import numpy as np
import matplotlib.ticker as mticker

from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind_from_stats as ttest
from prettytable import PrettyTable
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('fewshot_dir', type=str, help="Directory containing "
                                                  "fewshot training .json files.")
parser.add_argument('dev_file', type=str, help="Data on which we evaluate.")
parser.add_argument("--ft", default=False, action='store_true', help="Whether to fine-tune models.")
parser.add_argument("--rerun", default=False, action='store_true', help="If a dev scores file already exists, "
                                                    "do not rerun fine-tuning.")

args = parser.parse_args()

# number of fine-tuning epochs on the largest few-shot fine-tuning set that gets us the best dev performance
EPOCHS_FT = 2
num_shots = defaultdict(list)   # key: model name. value: list of graph values
ic_acc = defaultdict(list)
ic_stddev = defaultdict(list)

random_seeds = (12, 24, 36, 48, 60)     # for comparing across models

# huggingface model name OR directory containing trained model
models = ("t5-base", "../output/t5-cpt-gold-fulllab/epoch3/", "../output/t5-cpt-goldsilver-fulllab/epoch4/",
          "../output/t5-cpt-mednoisy-probfilter/epoch1/", "../output/t5-cpt-largenoisy/epoch0.5/")

for model in models:
    output_dir = "../output/t5-fewshot" if model == "t5-base" else model
    for filename in os.listdir(args.fewshot_dir):
        num_examples_per_intent = int(filename.split("_")[1])
        # same number of fine-tuning iterations regardless of size of fine-tuning set
        num_epochs = EPOCHS_FT * (2**(math.log2(32) - math.log2(num_examples_per_intent)))

        scores_across_seeds = []
        for seed in random_seeds:
            scores_file = os.path.basename(args.dev_file).replace(".json", "") + \
                "_" + filename.replace(".json", "") + "_seed" + str(seed) + \
                ".eval_results_seq2seq.txt"
            scores_file = os.path.join(output_dir, scores_file)

            # run fine-tuning (if needed)
            if args.ft and (args.rerun or not os.path.exists(scores_file)):
                ft_command = '../scripts/finetune_t5_config.sh {} {} {} {} {} {}'.format(
                    model, os.path.join(args.fewshot_dir, filename), args.dev_file,
                    output_dir, str(num_epochs), str(seed)
                )
                print(f"Running {model} for {num_epochs} epochs w/ {num_examples_per_intent} examples "
                      f"per intent (seed: {seed})")
                os.system(ft_command)   # run fine-tuning script and wait until completion

            with open(scores_file, 'r') as scores:
                exact_match_str = scores.readline()
                exact_match_score = float(exact_match_str.split(":")[1].strip())
                scores_across_seeds.append(exact_match_score)

        # calculate mean score and std. dev of score
        num_shots[model].append(num_examples_per_intent)
        ic_acc[model].append(np.mean(scores_across_seeds))
        ic_stddev[model].append(np.std(scores_across_seeds))


# graph IC accuracies and std. devs by model and num_examples_per_intent
id_to_name = {
    't5-base': 'T5',
    '../output/t5-cpt-gold-pack/': 'T5 (CPT, label semantics, seq pack)',
    '../output/t5-cpt-gold-nolabsem/epoch3/': 'T5 (CPT, 15% noising, no label semantics)',
    "../output/t5-cpt-gold/epoch3/": "T5 (CPT, 15% noising)",
    "../output/t5-cpt-gold-fulllab/epoch3/": 'T5 (CPT, label noising) +gold',
    "../output/t5-cpt-gold-fulllab-halfdata/epoch1/": "T5 (CPT, label noising, half data)",
    "../output/t5-ic-gold/epoch3/": "T5 (IC)",
    "../output/t5-cpt-goldsilver-fulllab/epoch4/": "T5 (CPT, label noising) +gold+WikiHow",
    "../output/t5-ic-goldsilver/epoch1/": "T5 (IC w/ wikiHow) epoch1",
    "../output/t5-ic-goldsilver/epoch2/": "+gold+WikiHow (IC)",
    "../output/t5-cpt-smallnoisy/epoch2/": "+gold+WikiHow+CSTwitter",
    "../output/t5-ic-smallnoisy/epoch4/": "+gold+WikiHow+CSTwitter (IC)",
    "../output/t5-multichoice-gold/epoch4/": "T5 (multi-choice)",
    "../output/t5-cpt-mednoisy/epoch1/": "+gold+silver+CSTwitter+Reddit",
    "../output/t5-cpt-mednoisy-probfilter/epoch1/": "+gold+silver+CSTwitter+Reddit (filter)",
    "../output/t5-cpt-largenoisy/epoch2/": "+C4 (2x) epoch2"
}

palette = itertools.cycle(sns.color_palette("Set2"))
sns.set_style('whitegrid')
fig, ax = plt.subplots()
for model in ic_acc.keys():
    # sort by x values
    sorted_num_shots, sorted_ic_acc = zip(*sorted(zip(num_shots[model], ic_acc[model])))
    _, sorted_ic_stddev = zip(*sorted(zip(num_shots[model], ic_stddev[model])))
    color = next(palette)
    plt.plot(sorted_num_shots, sorted_ic_acc, label=id_to_name[model], marker='.', color=color)
    plt.fill_between(sorted_num_shots, np.array(sorted_ic_acc) - np.array(sorted_ic_stddev),
                     np.array(sorted_ic_acc) + np.array(sorted_ic_stddev), facecolor=color, alpha=0.3)

    print(model, np.mean(sorted_ic_acc))
    # print(model, np.mean(sorted_ic_stddev))
    for shots, acc, stddev in zip(sorted_num_shots, sorted_ic_acc, sorted_ic_stddev):
        print(f"\t{shots}: {acc} ({stddev})")

plt.legend()
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.set_ylim([0.6, 1.0])
if "reminder" in args.dev_file or "weather" in args.dev_file:   # different y-range for TOPv2 eval
    ax.set_ylim([0.5, 1.0])
    domain = "weather" if "weather" in args.dev_file else "reminder"
    plt.title(f"TOP ({domain}) few-shot IC accuracy")
elif "snips" in args.dev_file:
    plt.title("SNIPS few-shot IC accuracy")
elif "atis" in args.dev_file:
    plt.title("ATIS few-shot IC accuracy")
plt.xlabel('Examples per-intent')
plt.ylabel('IC Accuracy')
plt.savefig('fewshot/t5_med_vs_large.png', format='png', bbox_inches='tight')

# calculate significances for each model pair and each num_shots
table = PrettyTable()
table.field_names = ["Model 1", "Model 2", 1, 2, 4, 8, 16, 32]
for model_1 in ic_acc.keys():
    for model_2 in ic_acc.keys():
        if model_1 == model_2:
            continue
        sorted_num_shots_1, sorted_ic_acc_1 = zip(*sorted(zip(num_shots[model_1], ic_acc[model_1])))
        _, sorted_ic_stddev_1 = zip(*sorted(zip(num_shots[model_1], ic_stddev[model_1])))
        sorted_num_shots_2, sorted_ic_acc_2 = zip(*sorted(zip(num_shots[model_2], ic_acc[model_2])))
        _, sorted_ic_stddev_2 = zip(*sorted(zip(num_shots[model_2], ic_stddev[model_2])))
        table_row = [id_to_name[model_1], id_to_name[model_2]]
        for idx, _ in enumerate(sorted_num_shots_1):
            p_val = ttest(sorted_ic_acc_1[idx], sorted_ic_stddev_1[idx], 5,
                          sorted_ic_acc_2[idx], sorted_ic_stddev_2[idx], 5)[1]
            if p_val < .001:
                sig_symbol = "***"
            elif p_val < .01:
                sig_symbol = "**"
            elif p_val < .05:
                sig_symbol = "*"
            else:
                sig_symbol = ""
            table_row.append(sig_symbol)
        table.add_row(table_row)
print(table)
