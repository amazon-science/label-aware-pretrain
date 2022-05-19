import json
import os
import sys

data_dir = sys.argv[1]
for filename in os.listdir(data_dir):
    if filename.endswith(".proc.json"):
        continue
    if not filename.endswith(".json"):
        continue

    split = filename.split(".json")[0]
    with open(os.path.join(data_dir, filename), 'r') as in_data, \
            open(os.path.join(data_dir, split+".proc.json"), 'w') as out_data:
        for line in in_data:
            data = json.loads(line.strip())
            if len(data["original_intents"]) > 0 and data["original_intents"][0].lower() == "outofdomain":
                continue
            if len(data["utterance"].strip().split()) < 2:
                continue

            label_transform = {'NotIntent': 'no', 'Intent': 'yes'}
            data['label'] = label_transform[data['label']]
            out_data.write(json.dumps(data) + "\n")