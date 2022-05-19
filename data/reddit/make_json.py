import json
import sys
import random
import ast

in_file = sys.argv[1]
out_file = in_file.replace(".json", ".utt.json")
with open(sys.argv[1], 'r') as in_data, open(out_file, 'w') as out_data:
    for line in in_data:
        try:
            data = json.loads(line)
        except:
            continue
        utterance = data["body"].strip()
        if len(utterance.split()) < 2:
            continue
        label = random.choice(["yes", "no"])
        try:
            json_obj = json.dumps({"utterance": utterance, "label": label})
        except:
            continue
        out_data.write(json_obj + "\n")