import csv
import json
import sys
import random

in_file = sys.argv[1]
out_file = in_file.replace(".csv", ".json")
# set random seed for reproducible behavior
random.seed(12)
inbound_count, outbound_count = 0, 0
with open(sys.argv[1], 'r') as in_data, open(out_file, 'w') as out_data:
    in_data.readline()  # skip header
    reader = csv.reader(in_data)
    for row in reader:
        utterance = row[4]
        label = random.choice(["yes", "no"])
        if row[2] == "False":   # is an outbound utterance
            outbound_count += 1
        else:
            inbound_count += 1
        json_obj = json.dumps({"utterance": utterance, "label": label, "inbound": row[2]})
        out_data.write(json_obj + "\n")

print(f"Inbound count: {inbound_count} || Outbound count: {outbound_count}")