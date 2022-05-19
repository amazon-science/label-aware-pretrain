import sys
import random
import json
import os
from collections import defaultdict

# get list of training examples by intent
intent_to_example = defaultdict(list)
with open(sys.argv[1], "r") as in_file:
    for line in in_file:
        data = json.loads(line)
        intent_name = data["translation"]["tgt"]
        utterance = data["translation"]["src"]
        intent_to_example[intent_name].append(utterance)

# sample `NUM_SAMPLES` training examples per intent *without repetition*
NUM_SAMPLES = 8    # == 2**3
intent_samples = defaultdict(list)
for intent in intent_to_example.keys():
    intent_samples[intent] = random.sample(intent_to_example[intent], NUM_SAMPLES)

if not os.path.exists("fewshot_ft"):
    os.mkdir("fewshot_ft")

# create few-shot fine-tuning data
# smaller datasets will be subsets of larger ones
for exp in range(4):
    num_per_intent = 2**exp
    with open(f"fewshot_ft/train_{num_per_intent}_examples.json", "w") as data_out:
        for utterance_id in range(num_per_intent):
            for intent in intent_samples.keys():
                data = {"translation":
                            {"src": intent_samples[intent][utterance_id],
                             "tgt": intent,
                             "prefix": "intent classification: "}
                       }
                json_obj = json.dumps(data)
                data_out.write(json_obj + "\n")