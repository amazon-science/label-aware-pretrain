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

# create .json file
with open(sys.argv[2], "w") as data_out:
    for intent in intent_to_example.keys():
        for utterance in intent_to_example[intent]:
            data = {"translation":
                        {"src": utterance, "tgt": intent, "prefix": "intent classification: "}
                   }
            json_obj = json.dumps(data)
            data_out.write(json_obj + "\n")
