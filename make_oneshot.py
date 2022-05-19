import sys
import random
import json
from collections import defaultdict

# get list of training examples by intent
intent_to_example = defaultdict(list)
with open(sys.argv[1], "r") as in_file:
    for line in in_file:
        intent_name, utterance = line.split(",")
        utterance = utterance.strip()
        intent_name = " ".join(intent_name.split("_")[1:])
        intent_to_example[intent_name].append(utterance)

# sample 10 training examples per intent *without repetition*
NUM_SAMPLES = 10
intent_samples = defaultdict(list)
for intent in intent_to_example.keys():
    intent_samples[intent] = random.sample(intent_to_example[intent], NUM_SAMPLES)

# create NUM_SAMPLES fine-tuning sets
for ft_num in range(NUM_SAMPLES):
    with open(f"oneshot_ft/train_{ft_num}.json", "w") as data_out:
        for intent in intent_samples.keys():
            data = {"translation":
                        {"src": intent_samples[intent][ft_num], "tgt": intent, "prefix": "intent classification: "}
                   }
            json_obj = json.dumps(data)
            data_out.write(json_obj + "\n")
