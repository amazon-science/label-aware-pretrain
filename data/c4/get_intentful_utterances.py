import json
import sys
import random
import numpy as np

data_file = sys.argv[1]
predictions_file = sys.argv[2]
with open(data_file, 'r') as data, open(predictions_file, 'r') as predictions:
    predictions.readline()
    intentful_count = 0
    yes_prob_list = []
    intentful_utterances = []
    for example, prediction in zip(data, predictions):
        idx, is_intentful, yes_prob = prediction.strip().split("\t")
        if is_intentful == "yes":
            yes_prob_list.append(float(yes_prob))
            intentful_count += 1
            intentful_utterances.append(json.loads(example)["utterance"])
            print(example.strip(), yes_prob)
    quartiles = np.quantile(yes_prob_list, [0, 0.25, 0.5, 0.75, 1.0])
    # only print utterances tagged as intentful w/ probability over the median for intentful utterances

    '''
    filtered_utterances = [utterance for idx, utterance in enumerate(intentful_utterances)
                           if yes_prob_list[idx] >= quartiles[2]]
    for utterance in filtered_utterances[:150]:
        data = {"translation": {"src": utterance.replace("\n", " "), "tgt": "s", "prefix": "intent classification: "}}
        print(json.dumps(data))
    '''
    # print(len(filtered_utterances))

    '''
    print(quartiles)
    print("Minimum: ", quartiles[0])
    print("\n".join(random.sample([utterance for idx, utterance in enumerate(intentful_utterances)
                         if yes_prob_list[idx] < quartiles[1]], 5)))
    print("")
    print("1st quartile: ", quartiles[1])
    print("\n".join(random.sample([utterance for idx, utterance in enumerate(intentful_utterances)
                         if quartiles[2] > yes_prob_list[idx] > quartiles[1]], 5)))
    print("")
    print("2nd quartile (median): ", quartiles[2])
    print("\n".join(random.sample([utterance for idx, utterance in enumerate(intentful_utterances)
                         if quartiles[3] > yes_prob_list[idx] > quartiles[2]], 5)))
    print("")
    print("3rd quartile: ", quartiles[3])
    print("\n".join(random.sample([utterance for idx, utterance in enumerate(intentful_utterances)
                         if quartiles[4] > yes_prob_list[idx] > quartiles[3]], 5)))
    print("")
    print("Maximum: ", quartiles[4])
    '''

    # print(intentful_count)