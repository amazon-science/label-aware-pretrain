import json
import sys

def process_example(input):
    utterance, intent = input
    intent = intent.replace("_", " ")
    data = {"translation": {"src": utterance, "tgt": intent, "prefix": "intent classification: "}}
    return data

in_file = sys.argv[1]
out_basename = in_file.split(".json")[0] + "_proc"
with open(sys.argv[1], 'r') as in_data, open(out_basename+"_train.json", 'w') as train_data, \
        open(out_basename+'_test.json', 'w') as test_data:
    one_str = in_data.read().replace("\n", " ")
    json_data = json.loads(one_str)

    for train_ex in json_data["train"]:
        data = process_example(train_ex)
        train_data.write(json.dumps(data) + '\n')
    for test_ex in json_data["test"]:
        data = process_example(test_ex)
        test_data.write(json.dumps(data) + '\n')
