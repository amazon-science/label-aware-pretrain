import sys
import json
import torch
import numpy as np

from zero_shot_generator import ConstrainedGenerationPipeline
from tqdm import tqdm


"""DRIVER CODE"""
model_name = sys.argv[2] if len(sys.argv) > 2 else "t5-base"
label_set = set()

# first read: grab set of possible intent labels
dataset = []    # list of dictionaries
with open(sys.argv[1], "r") as eval_data:
    for example in eval_data:
        data = json.loads(example)
        dataset.append(data)
        intent = data["translation"]["tgt"].strip()
        label_set.update([intent])

label_set = list(label_set)
mean_prob = {}
mean_stddev = {}
classifier = ConstrainedGenerationPipeline(model_name)

# second read: perform constrained generation and evaluation
with open(".".join(sys.argv[1].split(".")[:-1]) + ".cg.preds", "w") as preds_file:
    total = 0.0
    correct = 0.0
    gold_labels = [data["translation"]["tgt"].strip() for data in dataset]
    print("Performing classification")
    classifier_out = [classifier(data["translation"]["src"], label_set) for data in dataset]
    # print(classifier_out)
    print("Calculating accuracy")
    '''
    # get per-label probability and variance, marginalized over utterances
    for lab_idx, label in enumerate(label_set):
        mean_prob[label] = np.mean([classifier_out[idx]['scores'][0][lab_idx] for \
                                    idx, data in enumerate(dataset)])
        mean_stddev[label] = np.std([classifier_out[idx]['scores'][0][lab_idx] for \
                 idx, data in enumerate(dataset)])
        print(f"{label}: {str(mean_prob[label])} ({str(mean_stddev[label])})")
    '''
    pred_labels = [label_set[np.argmax(classifier_out[idx]['scores'][0])] for \
                   idx, data in enumerate(dataset)]
    correct = sum(pred_label == gold_label for pred_label, gold_label in zip(pred_labels, gold_labels))
    total = float(len(dataset))
    print("Accuracy: {}".format(correct / total))

    print("Writing predictions to file")
    for pred_label in pred_labels:
        preds_file.write(pred_label + '\n')
    '''
    for data in tqdm(dataset):
        gold_label = data["translation"]["tgt"].strip()
        classifier_out = classifier(data, label_set)
        pred_label = classifier_out['labels'][np.argmax(classifier_out['scores'][0])]
        preds_file.write(pred_label + "\n")
        if pred_label == gold_label:
            correct += 1
        total += 1
    '''