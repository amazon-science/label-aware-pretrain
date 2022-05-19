import sys
import json
import numpy as np
from transformers import pipeline

def load_json_data(dataset):
    utterances = []
    labels = []
    with open(dataset, "r") as in_data:
        for line in in_data:
            data = json.loads(line)
            utterances.append(data["translation"]["src"])
            labels.append(data["translation"]["tgt"])
    return utterances, labels

is_atis = False
is_snips = False
dataset = sys.argv[1]
if "atis" in dataset:
    is_atis = True
elif "snips" in dataset:
    is_snips = True

device = "cuda"

utterances = []
labels = []
# handle .json inputs
if dataset.endswith(".json"):
    utterances, labels = load_json_data(dataset)
else:
    if not is_atis and not is_snips:
        raise Exception("Unrecognized data format.")
    with open(dataset, "r") as in_data:
        for line in in_data:
            # handle atis input format
            if is_atis:
                intent, utterance = line.split(",")
                intent = " ".join(intent.split("_")[1:])
                labels.append(intent)
                utterances.append(utterance)

'''
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

label_set = list(set(labels))
for utterance, label in zip(utterances, labels):
    input_sent = [utterance, utterance]
    input_ids = tokenizer(input_sent, return_tensors='pt').to(device).input_ids
    target_ids = tokenizer(label_set, return_tensors='pt').to(device).input_ids
    outputs = model(input_ids=input_ids, labels=target_ids, use_cache=False, return_dict=True)
    print(outputs)
'''

label_set = list(set(labels))
print(label_set)
classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")

correct = 0
total = 0
with open(".".join(sys.argv[1].split(".")[:-1])+".cg.bart.preds", "w") as out_preds:
    for idx, (utterance, gold_label) in enumerate(zip(utterances, labels)):
        classifier_out = classifier(utterance, label_set)
        pred_label = classifier_out['labels'][np.argmax(classifier_out['scores'])]
        out_preds.write(pred_label + "\n")
        if pred_label == gold_label:
            correct += 1
        total += 1

        if idx % 100 == 0:
            print("Processing example {}".format(idx))

print("Accuracy: {}".format(correct / total))
