import json
import sys

data_file = sys.argv[1]
predictions_file = sys.argv[2]
inbound_count, outbound_count = 0, 0
with open(data_file, 'r') as data, open(predictions_file, 'r') as predictions:
    predictions.readline()
    intentful_count = 0
    for example, prediction in zip(data, predictions):
        json_obj = json.loads(example)
        is_intentful = prediction.strip().split("\t")[1]
        utterance = json_obj["utterance"].strip().replace("\n", " ")
        intent = "s"
        if is_intentful == "yes":
            intentful_count += 1
            if json_obj["inbound"] == "False":
                outbound_count += 1
                utterance = " ".join(utterance.split()[1:]) # remove @<NUM> at start of customer support tweet
            else:
                inbound_count += 1
            print(json.dumps({"translation": {"src": utterance, "tgt": intent,
                                              "prefix": "intent classification: "}}))

# print(f"Intentful utterances: {intentful_count}")
# print(f"Inbound intentful utterances: {inbound_count} || Outbound intentful utterances: {outbound_count}")