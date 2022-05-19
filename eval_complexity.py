import json
import argparse

from transformers import T5Tokenizer

def load_dataset(dataset):
    """
    Load .json dataset.
    """
    ds = []
    with open(dataset, 'r') as in_data:
        for example in in_data:
            json_data = json.loads(example)
            ds.append(json_data)
    return ds


def calculate_metric(dataset, metric, tokenizer, model=None):
    """Calculate complexity of evaluation set. Returns float or int.
    Arguments:
        dataset: dev or test file on which we evaluate
        metric: quantitative metric of dataset complexity. Options:
            token vocabulary: number of tokens needed to represent all intents in the eval set when using
                              `tokenizer`.
            perplexity: perplexity of the evaluation set according to `model` using the supervised intent
                        classification format.
        model: the saved model (checkpoint). Only used when calculating the perplexity metric.
    """

    if metric not in ("token vocabulary", "perplexity"):
        raise ValueError("Unrecognized metric.")

    ds = load_dataset(dataset)
    if metric == "token vocabulary":
        intent_set = set([example["translation"]["tgt"] for example in ds])
        num_intents = len(intent_set)
        print("Num intents: {}".format(num_intents))

        all_tokens = []
        for intent in intent_set:
            all_tokens.extend(tokenizer.encode(intent))
        num_unique_tokens = len(set(all_tokens))
        return num_unique_tokens

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="Dataset to evaluate.")
    parser.add_argument('metrics', type=str, help="Metric (or comma-separated list of metrics).")
    parser.add_argument('--model', type=str, required=False, help="Model directory or checkpoint directory.")
    args = parser.parse_args()

    if "," in args.metrics:
        metrics = args.metrics.strip().split(",")
    else:
        metrics = [args.metrics.strip()]

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    for metric in metrics:
        if metric == "perplexity":
            model = args.model
        else:
            model = None
        print("{}: {}".format(metric, calculate_metric(args.dataset, metric, tokenizer, model)))