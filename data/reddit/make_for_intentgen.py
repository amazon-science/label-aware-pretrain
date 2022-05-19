import json
import sys
import random

in_file = sys.argv[1]
out_file = in_file.replace(".makegen.json", ".intgen.json")
with open(sys.argv[1], 'r') as in_data, open(out_file, 'w') as out_data:
    for example in in_data:
        data = json.loads(example)
        utterance = data["utterance"]
        intent = "s"    # doesn't matter. model won't see this but dataloader expects this field
        json_obj = json.dumps({"translation": {"src": utterance, "tgt": intent,
                                               "prefix": "intent classification: "}})
        out_data.write(json_obj + "\n")