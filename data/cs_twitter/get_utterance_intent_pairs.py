import json
import pickle
import argparse
import random
import numpy as np

from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="A .json containing the utterances.")
parser.add_argument("predictions", type=str, help="A file containing the intent predictions for the utterances.")
parser.add_argument("-c", "--intent-confidences", required=False, action="store_true",
                    help="Retrieve confidence-like scores for the intent generator. Calculate "
                         "probability quartiles and give example utterance-intent pairs for each "
                         "confidence quartile.")
parser.add_argument("-n", "--new-intents", required=False, action="store_true",
                    help="Print number of utterances tagged with intents that did not appear in the "
                         "training set. Also print the 100 most frequent new intents.")
parser.add_argument("-j", "--print-json", required=False, action="store_true",
                    help="Print utterance-intent .json-formatted training examples.")
parser.add_argument("-freq", "--most-frequent", required=False, action="store_true",
                    help="Print 100 most frequently generated intents.")
parser.add_argument("-a", "--all", required=False, action="store_true",
                    help="Print all utterance-intent pairs.")
args = parser.parse_args()

data_file = args.data
predictions_file = args.predictions
training_intents = pickle.load(open("../../label_set.pkl", "rb"))
training_intents = set([string.strip().lower() for string in training_intents])
intents = []
new_intents = []
utterances_intents = []
confidences = []
with open(data_file, 'r') as data, open(predictions_file, 'r') as predictions:
    for example, prediction in zip(data, predictions):
        prediction = prediction.strip()
        # deal with different formats of predictions
        if "\t" in prediction:
            intent = "\t".join(prediction.split("\t")[:-1])
            classifier_confidence = float(prediction.split("\t")[-1])
            confidences.append(classifier_confidence)
        else:
            if args.intent_confidences:     # usually indicates that an empty string or only spaces were generated
                print("SKIPPED:", example)
                continue
            intent = prediction

        intents.append(intent)
        if intent.lower() not in training_intents:
            new_intents.append(intent)
        utterance = json.loads(example)["translation"]["src"].strip().replace("\n", " ")
        utterances_intents.append(utterance + " ||| " + intent)
        if args.print_json:
            print(json.dumps({"translation": {"src": utterance, "tgt": intent, "prefix": "intent classification: "}}))

        if args.all:
            if args.intent_confidences:
                print(utterance, "|||", intent, "|||", classifier_confidence)
            else:
                print(utterance, "|||", intent)


if args.intent_confidences:
    if len(confidences) == 0:
        raise Exception("Used --intent_confidences argument, but the predictions file does not contain"
                        "confidence scores.")
    quartiles = np.quantile(confidences, [0, 0.25, 0.5, 0.75, 1.0])
    print("Minimum: ", quartiles[0])
    quart_ui = [ui for idx, ui in enumerate(utterances_intents) if confidences[idx] < quartiles[1]]
    print("\n".join(random.sample(quart_ui, 5)))
    print("% new intents: {}".format(len([intent for intent in quart_ui
                        if intent.split("|||")[-1].strip().lower() not in training_intents]) / len(quart_ui)))
    print("")
    print("1st quartile: ", quartiles[1])
    quart_ui = [ui for idx, ui in enumerate(utterances_intents) if quartiles[2] > confidences[idx] > quartiles[1]]
    print("\n".join(random.sample(quart_ui, 5)))
    print("% new intents: {}".format(len([intent for intent in quart_ui
                        if intent.split("|||")[-1].strip().lower() not in training_intents]) / len(quart_ui)))
    print("")
    print("2nd quartile (median): ", quartiles[2])
    quart_ui = [ui for idx, ui in enumerate(utterances_intents) if quartiles[3] > confidences[idx] > quartiles[2]]
    print("\n".join(random.sample(quart_ui, 5)))
    print("% new intents: {}".format(len([intent for intent in quart_ui
                        if intent.split("|||")[-1].strip().lower() not in training_intents]) / len(quart_ui)))
    print("")
    print("3rd quartile: ", quartiles[3])
    quart_ui = [ui for idx, ui in enumerate(utterances_intents) if confidences[idx] > quartiles[3]]
    print("\n".join(random.sample(quart_ui, 5)))
    print("% new intents: {}".format(len([intent for intent in quart_ui
                        if intent.split("|||")[-1].strip().lower() not in training_intents]) / len(quart_ui)))
    print("")
    print("Maximum: ", quartiles[4])


if args.new_intents:
    print("Most frequent generated intents that weren't in training set:")
    print(Counter(new_intents).most_common(100))
    print("Number of utterances tagged with new intents: ", str(len(new_intents)))
    print("% new intents: ", len([ui for ui in utterances_intents
                    if ui.split("|||")[-1].strip().lower() not in training_intents]) / len(utterances_intents))

if args.most_frequent:
    print("Most frequent generated intents:")
    print(Counter(intents).most_common(100))
