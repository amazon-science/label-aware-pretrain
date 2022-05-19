import json
import sys

in_file = sys.argv[1]
out_file = in_file.split(".json")[0] + ".multichoice.json"
utterances = []
intents = []

# first pass: get and store set of all possible intents. store utterances and intents too
with open(in_file, 'r') as in_data:
    for line in in_data:
        data = json.loads(line)
        utterance = data["translation"]["src"]
        intent = data["translation"]["tgt"]
        utterances.append(utterance)
        intents.append(intent)

intent_set = set(intents)
intents_str = ". ".join(intent_set)

with open(out_file, 'w') as out_data:
    for utterance, intent in zip(utterances, intents):
        data = {"translation": {"src": f"intents: {intents_str}. utterance: {utterance}",
                                "tgt": intent+".", "prefix": ""}}
        out_data.write(json.dumps(data) + '\n')