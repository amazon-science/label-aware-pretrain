import json
import sys

from nltk import sent_tokenize

less_than_3 = 0
less_than_4 = 0
out_file = sys.argv[1].split(".json")[0] + ".short.json"
with open(sys.argv[1], 'r') as filtered_utts, open(out_file, 'w') as out_data:
    for line in filtered_utts:
        utterance = json.loads(line)["utterance"]
        num_words = len(utterance.split())
        num_sentences = len(sent_tokenize(utterance))
        if num_sentences < 3 and num_words < 128:
            less_than_3 += 1
            out_data.write(line)
        if num_sentences < 4 and num_words < 128:
            less_than_4 += 1

print("3 sents or less: {}".format(less_than_4))
print("2 sents or less: {}".format(less_than_3))