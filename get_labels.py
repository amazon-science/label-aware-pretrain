import json
import argparse
import pickle
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help="File containing "
                                                  "utterances and intents.")
parser.add_argument("--write", default=False, action='store_true', help="Create label-to-label fine-tuning set.")
parser.add_argument("--save-label-set", default=False, action='store_true', help="Save set of labels to file.")

args = parser.parse_args()

intent_list = []
intent_set = set()
with open(args.dataset, "r") as in_file:
    for line in in_file:
        if args.dataset.endswith(".json"):
            data = json.loads(line)
            intent_name = data["translation"]["tgt"].strip()
            # intent_name = data["label"]
            intent_list.append(intent_name)
            intent_set.update([intent_name])
            continue
        intent_name = line.split(",")[0]
        intent_set.update([intent_name])
        intent_list.append(intent_name)

intent_counts = Counter(intent_list)

print(intent_counts, len(intent_set))

if args.write:
    out_file = ".".join(args.dataset.split(".")[:-1]) + "_labels.json"
    with open(out_file, 'w') as label_dataset:
        for intent in intent_set:
            data = json.dumps({'translation': {
                "src": intent, "tgt": intent, 'prefix': 'intent classification: '}
            })
            label_dataset.write(data + '\n')

if args.save_label_set:
    pickle.dump(intent_set, open("label_set.pkl", "wb"))