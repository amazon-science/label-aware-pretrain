import json
import sys

in_file = sys.argv[1]
out_file = in_file.replace(".json", "_filter.json")
with open(in_file, 'r') as in_data, open(out_file, 'w') as out_data:
    for line in in_data:
        json_obj = json.loads(line)
        json_obj["body"] = json_obj["body"].strip().replace("\n", " ")
        text = json_obj["body"]
        json_obj["author"] = json_obj["author"].strip()
        author = json_obj["author"]

        # start filtering criteria
        if len(text.split()) < 3:       # text is shorter than 3 words
            continue                    # (also filters out [removed] or [deleted] comments)
        if len(text.split()) > 200:     # text is paragraph-length or longer. probably not intentful utterance.
            continue
        if "bot" in author.lower():     # author is (probably) a bot
            continue
        if "http://" in text or "https://" in text or ".com " in text:   # contains URL. these utterances tend to be messy/noisy
            continue
        # end filtering criteria

        out_data.write(json.dumps(json_obj) + "\n")