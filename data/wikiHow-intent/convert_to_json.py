import sys
import csv
import json

with open(sys.argv[1], 'r') as in_data, open("wikihow.json", 'w') as out_data:
    in_data.readline()  # skip header
    reader = csv.reader(in_data, quotechar='"')
    for row in reader:
        utterance = row[5].strip()
        correct_label_idx = int(row[-1])
        how_to_phrase = row[7 + correct_label_idx]
        intent = " ".join(how_to_phrase.strip().split()[2:])

        json_obj = json.dumps({"translation":
            {"src": utterance, "tgt": intent, "prefix": "intent classification: "}
        })
        out_data.write(json_obj + '\n')