from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import defaultdict
import os
import json
import sys
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_intents_json(intents_file):
    intents = []
    with open(intents_file, 'r') as examples:
        for example in examples:
            data = json.loads(example)
            intent = data["translation"]["tgt"]
            intents.append(intent)
    return intents

def load_intents_list(intents_file):
    intents = []
    with open(intents_file, 'r') as intent_preds:
        for intent in intent_preds:
            intents.append(intent.strip())
    return intents

if len(sys.argv) < 3:
    raise Exception("Usage: python conf_mat.py <true_intents.json> <pred_intents.txt>")

confmat_name = ""
if "snips" in sys.argv[1]:
    confmat_name = "snips"
elif "atis" in sys.argv[1]:
    confmat_name = "atis"
else:
    confmat_name = "other"

true_intents = load_intents_json(sys.argv[1])
pred_intents = load_intents_list(sys.argv[2])


# replace all bad generated intents with BAD GENERATION
intent_set = set(true_intents)
for idx, pred in enumerate(pred_intents):
    if pred not in intent_set:
        pred_intents[idx] = "ε"

# FOR DEBUGGING
print(set(true_intents), set(pred_intents))
for intent in set(true_intents):
    if intent not in set(pred_intents):
        pred_intents.append(intent)
        true_intents.append("ε")

if "ε" not in set(true_intents):
    true_intents.append("ε")
    pred_intents.append("ε")

precisions = defaultdict(list)
recalls = defaultdict(list)
f1s = defaultdict(list)
precs, recs, f1s, supports = precision_recall_fscore_support(true_intents, pred_intents)

print("Precisions: ", precs)
print("Recalls: ", recs)
print("F1 scores: ", f1s)

# cmap = sns.color_palette("Reds", 256)
cmap = sns.color_palette("Blues", 256)

out_path = "confmat_figures"
if not os.path.exists(out_path):
    os.makedirs(out_path)

#sns.set_context("paper", rc={"font.size":18,"axes.labelsize":18})
sns.set(font_scale = 1.0)

data = confusion_matrix(true_intents, pred_intents)
print(confusion_matrix)
df_cm = pd.DataFrame(data, columns=np.unique(pred_intents), index=np.unique(true_intents))
df_cm.index.name = "True Intent"
df_cm.columns.name = "Predicted Intent"

# non-normalized figure
ax = sns.heatmap(df_cm, cmap=cmap, annot=False)
# ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
# ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
# ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
# ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
plt.xticks(rotation=45, ha='left')
plt.savefig(os.path.join(out_path, f"conf_mat_{confmat_name}_1shot.png"), format="png", \
    bbox_inches='tight', pad_inches=0.01)
plt.cla(); plt.clf()

# normalized figure
df_cmn = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]
df_cmn.index.name = "True Intent"
df_cmn.columns.name = "Predicted Intent"
ax = sns.heatmap(df_cmn, cmap=cmap, annot=False, vmax=1.0)
# ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
# ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
# ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0, 100, 10)))
# ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(left=True)
plt.xticks(rotation=45, ha='left')
plt.savefig(os.path.join(out_path, f"conf_mat_{confmat_name}_1shot_norm.png"), format="png", \
    bbox_inches='tight', pad_inches=0.01)
plt.cla(); plt.clf()
