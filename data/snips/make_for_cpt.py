import sys
import json
from gluonnlp.data import SNIPSDataset

idx_to_label = {
        0: 'add to playlist',
        1: 'book restaurant',
        2: 'get weather',
        3: 'play music',
        4: 'rate book',
        5: 'search creative work',
        6: 'search screening event'
        }

segment = sys.argv[1]
assert segment in ("train", "dev", "test")

out_file = sys.argv[2]
snips_test = SNIPSDataset(root=".", segment=sys.argv[1])
with open(out_file, "w") as data_out:
    for example in snips_test:
        source = " ".join(example[0])
        target = idx_to_label[example[2][0]]
        data = {"translation": 
                {"src": source, "tgt": target, "prefix": "intent classification: "}
            }
        json_obj = json.dumps(data)
        data_out.write(json_obj + "\n")
