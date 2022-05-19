import json
import sys
import random
import gzip

from nltk import sent_tokenize

in_file = sys.argv[1]
out_file = in_file.replace(".json.gz", "_filter.json")
random.seed(1248)
with gzip.open(in_file, 'r') as in_data, open(out_file, 'w') as out_data:
    for line in in_data:
        json_obj = json.loads(line)
        text = json_obj["text"].strip().replace("\n", " ")

        # start filtering criteria
        if len(text.split()) < 3:       # text is shorter than 3 words
            continue                    # (also filters out [removed] or [deleted] comments)
        if "http://" in text or "https://" in text or ".com " in text:  # contains URL. these utterances tend to be messy/noisy
            continue

        sentences = sent_tokenize(text)
        if len(sentences) > 1:          # take first 3 sentences, treat them as separate training examples
            sentences = sentences[:3]
            for sentence in sentences:
                if len(sentence.split()) > 128:
                    continue
                data = {"utterance": sentence.strip(), "label": random.choice(["yes", "no"])}
                out_data.write(json.dumps(data) + '\n')
            continue
            '''
            # previous approach: randomly take first 1-3 sentences, use that as training example
            num_sents = random.randint(1, 3)
            text = " ".join(sentences[:num_sents])
            '''

        if len(text.split()) > 128:     # text is paragraph-length or longer. filter out
            continue
        # end filtering criteria

        data = {"utterance": text, "label": random.choice(["yes", "no"])}
        out_data.write(json.dumps(data) + "\n")