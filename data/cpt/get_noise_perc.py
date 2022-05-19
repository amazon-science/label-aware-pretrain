import json
import sys
import numpy as np
import seaborn as sns
from transformers import T5Tokenizer
from matplotlib import pyplot as plt

tokenizer = T5Tokenizer.from_pretrained('t5-base')

noise_percs = []
with open(sys.argv[1], 'r') as in_data:
    for data_string in in_data:
        data = json.loads(data_string)
        noised_len = len(tokenizer.encode(data["targets"])) - 1
        src_len = len(tokenizer.encode(data["inputs"])) + noised_len - 2
        noise_percs.append((noised_len / src_len) * 100)

avg_noise_perc = np.mean(noise_percs)
var_noise_perc = np.std(noise_percs)

# plot histogram of noise density per-example
sns.set()
sns.set_style('whitegrid')
hist, bin_edges = np.histogram(noise_percs, bins=100)
plt.bar(bin_edges[:-1], hist)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('% tokens noised')
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80], [0, 10, 20, 30, 40, 50, 60, 70, 80])
plt.ylabel('Frequency')
plt.savefig('noise_percs.png', format='png', bbox_inches='tight')

print(f"Average noising percentage: {avg_noise_perc}")
print(f"Std. dev. of noising percentage: {var_noise_perc}")