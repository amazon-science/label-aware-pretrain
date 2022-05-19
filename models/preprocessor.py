import torch
import random
import sys
import json
import functools
import argparse
from transformers import T5Tokenizer

"""
This code is heavily based on the TensorFlow preprocessing code from the T5 paper, available here:
https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py

Adapted for use with huggingface (torch) by Aaron Mueller.
"""

def to_dict(text, tokenizer, include_eos=True):
    target = tokenizer.encode(text) if include_eos else tokenizer.encode(text)[:-1]
    return {'inputs': "",
            'targets': torch.tensor(target)}


def load_data(in_file, tokenizer):
    """Expects input of the format
    `{"translation": {"src": utterance, "tgt": label}}`.
    Returns list of dictionaries of the following format:
    {"inputs": "", "targets": tensor([encoded_text])}."""
    utterances, intents = [], []
    punc = (".", ";", "!", "?", ",")
    with open(in_file, 'r') as datastrings:
        for datastring in datastrings:
            data = json.loads(datastring)
            utterance = data["translation"]["src"].strip()
            intent = data["translation"]["tgt"].strip()
            if not utterance.endswith(punc):
                utterance += "."
            if not intent.endswith(punc):
                intent += "."

            utterances.append(to_dict(utterance, tokenizer, include_eos=False))
            intents.append(to_dict(intent, tokenizer, include_eos=True))
    return (utterances, intents)


def write_data(dataset, out_name, tokenizer):
    with open(out_name, "w") as out_file:
        for data in dataset:
            data = {"inputs": tokenizer.decode(data["inputs"]),
                    "targets": tokenizer.decode(data["targets"])}
            json_obj = json.dumps(data)
            out_file.write(json_obj + "\n")


def span_corruption(utterances, intents,
                    sequence_length,
                    mean_noise_span_length=3.0,
                    noise_density=0.15,
                    seq_pack=False,
                    label_semantics="multiple choice",
                    label_noise_density=0.5):
    """Preprocessing for T5 denoising objective. Returns preprocessed
    tokenized and encoded data.
    Args:
        dataset -- list of tensors (N, ?) where N is number of examples.
                   tensor length depends on length of tokenized example.
        sequence_length -- Maximum sequence length (default: 512)
        seq_pack -- pack inputs into sequences of length approximately `sequence_length`.
        label_semantics -- Whether and how to mask the utterance and intent. Can take the following values:
                               None: only use utterances. Intents will not appear in the data.
                               'concat': append intents to utterances, noise as if it were one full sequence.
                               'full label': simply mask the entire label and none of the utterance.
                               'separate': mask tokens in utterance with `noise_density` probability, and mask
                                           tokens in label with `label_noise_density` probability.
                               'label permute': try all possible ways of masking the tokens in the intent. Treat
                                                each permutation as a new training example.
                               'multiple choice': treat as a multiple choice problem. Give correct intent and
                                                  a set of [2, 29] random intents with the utterance in the source
                                                  sequence. Transduce to intent.
    """
    if label_semantics is not None and label_semantics not in ("full label", "label permute", "separate",
                                                               "multiple choice"):
            raise ValueError("Unrecognized label masking strategy. Must be one of "
                             "{'full label', 'label permute', 'separate'}.")

    input_length, targets_length = random_spans_helper(inputs_length=512)

    if sequence_length['targets'] < targets_length:
        # raise Exception("Exception not working?")
        raise ValueError(f'Expected targets length for span corruption ({targets_length}) is '
                         f'greater than configured targets length '
                         f"({sequence_length['targets']})")

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    proc_utterance_label_together = False
    if label_semantics is None:
        proc_utterance_label_together = True
        ds = utterances
    elif label_semantics == "concat":
        proc_utterance_label_together = True
        ds = []
        for utterance, intent in zip(utterances, intents):
            ds.append({'inputs': "",
                       'targets': torch.cat((utterance["targets"], intent["targets"]))})
    if proc_utterance_label_together:
        ds = select_random_chunk(ds)    # deal with inputs longer than 512 tokens
        if seq_pack:                    # pack sequences into training examples of ~512 tokens
            ds = random_concat(ds)
        ds = denoise(
            ds,
            tokenizer=tokenizer,
            inputs_fn=noise_span_to_unique_sentinel,
            targets_fn=nonnoise_span_to_unique_sentinel,
            noise_density=noise_density,
            noise_mask_fn=functools.partial(
                random_spans_noise_mask,
                mean_noise_span_length=mean_noise_span_length
            )
        )
        return ds

    if label_semantics == "full label":  # mask full label, not utterance
        ds = []
        for utterance, intent in zip(utterances, intents):
            sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
            input = torch.cat((utterance["targets"], torch.tensor([sentinel_id])))
            target = torch.cat((torch.tensor([sentinel_id]), intent["targets"]))
            if input.shape[0] > 512:    # if seq too long, truncate
                input = input[:512]
            data = {'inputs': input,
                    'targets': target}
            ds.append(data)
        return ds
    if label_semantics == "multiple choice":
        ds = []
        for utterance, intent in zip(utterances, intents):
            num_choices = int(random.uniform(2, 19))    # between 3 and 30 with correct intent
            intents_list = random.sample(intents, num_choices)
            intents_list.append(intent)
            random.shuffle(intents_list)
            # concatenate intent list (without eos tokens)
            intents_choices = torch.cat([intent_item["targets"][:-1] for intent_item in intents_list])
            int_prefix = torch.tensor(tokenizer.encode("intents: ")[:-1])   # [:-1] gets rid of eos token
            utt_prefix = torch.tensor(tokenizer.encode("utterance: ")[:-1])
            eos_id = tokenizer.convert_tokens_to_ids("</s>")
            source_tok = torch.cat((int_prefix, intents_choices, utt_prefix, utterance["targets"],
                                    torch.tensor([eos_id])))
            if source_tok.shape[0] > 512:   # if seq too long, only give the correct intent.
                source_tok = torch.cat((int_prefix, intent["targets"][:-1], utt_prefix, utterance["targets"],
                                        torch.tensor([eos_id])))
            if source_tok.shape[0] > 512:   # if seq still too long, truncate
                source_tok = source_tok[:512]
            data = {'inputs': source_tok, 'targets': intent["targets"]}
            ds.append(data)
        return ds


    # TODO: implement other label masking strategies



def random_spans_helper(inputs_length=512, noise_density=0.15,
                        mean_noise_span_length=3.0,
                        extra_tokens_per_span_inputs=1,
                        extra_tokens_per_span_targets=1):
    """Helps us avoid padding when masking inputs.
    Assumes that EOS token will be appended to examples."""

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        return (
            num_nonnoise_tokens +
            num_noise_spans * extra_tokens_per_span_inputs + 1,
            num_noise_tokens +
            num_noise_spans * extra_tokens_per_span_targets + 1)

    tokens_length = inputs_length
    while (_tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length):
        tokens_length += 1
    inputs_length, targets_length = (_tokens_length_to_inputs_length_targets_length(tokens_length))
    return tokens_length, targets_length


def select_random_chunk(dataset,
                        feature_key='targets',
                        max_length=512):
    """Extract one span of at most `max_length` tokens.
    If token sequence longer than `max_length`, return a random subsequence.
    If token sequence shorter than `max_length`, return the original sequence.
    """

    # Filter empty examples
    dataset = [example for example in dataset if example[feature_key].shape[0] > 0]
    # Select random chunk of tokens
    def _my_fn(data):
        tokens = data[feature_key]
        if tokens.shape[0] < max_length:
            return {feature_key: tokens}
        n_tokens = torch.tensor(tokens.shape[0])
        num_segments = torch.ceil(n_tokens.float() /
                                  torch.tensor(max_length, dtype=torch.float32)).type(torch.int32)
        start = (max_length * (-num_segments * torch.rand([]) + num_segments)).int()
        end = torch.minimum(start + max_length, n_tokens)
        chunk = {feature_key: tokens[start:end]}
        return chunk
    return [_my_fn(data) for data in dataset]


def random_concat(dataset, max_length=512, feature_key='targets'):
    """Pack random sequences together into training examples (w/o replacement).
    NOTE: expects all sequences to have length <= `max_length`! Be sure to run
    `select_random_chunk` on the data before running this function."""
    random.shuffle(dataset)
    new_dataset = []
    len_example = 0
    example = torch.tensor((), dtype=torch.int32)
    for data in dataset:
        len_example += data[feature_key].shape[0]
        if len_example >= max_length - 2:
            new_dataset.append({feature_key: example})
            example = data[feature_key]
            len_example = example.shape[0]
            continue
        example = torch.cat((example, data[feature_key]))
    # add final example to dataset
    new_dataset.append({feature_key: example})
    return new_dataset


def random_spans_noise_mask(length,
                            noise_density=0.15,
                            mean_noise_span_length=3.0):
    """Calculate which spans to mask given input length.
    Returns a vector of Booleans of length `length`, where `True`
    corresponds to masking and `False` corresponds to keeping a token.
    """
    orig_length = length
    length = torch.tensor(length, dtype=torch.int32)
    # avoid degenerate length values
    length = torch.maximum(length, torch.tensor(2, dtype=torch.int32))
    # helper functions for concise type conversion
    def to_int(x):
        return x.type(torch.int32)
    def to_float(x):
        return x.type(torch.float32)
    # calculate number of noised and non-noised tokens
    num_noise_tokens = to_int(torch.round(to_float(length) * noise_density))
    num_noise_tokens = torch.minimum(
        torch.maximum(num_noise_tokens, torch.tensor(1, dtype=torch.int32)), length-1)
    num_noise_spans = to_int(
        torch.round(to_float(num_noise_tokens) / mean_noise_span_length))
    num_noise_spans = torch.maximum(num_noise_spans, torch.tensor(1, dtype=torch.int32))
    num_nonnoise_tokens = length - num_noise_tokens
    # pick lengths of noise spans and non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition items randomly into non-empty segments."""
        first_in_segment = torch.nn.functional.pad(
            shuffle(to_int(torch.arange(num_items - 1) < num_segments - 1)),
            [1, 0])
        segment_id = torch.cumsum(first_in_segment, 0)
        segment_length = segment_sum(torch.ones_like(segment_id), segment_id)
        return segment_length

    noise_span_lengths = _random_segmentation(
        num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = torch.reshape(
        torch.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
                    [num_noise_spans * 2])
    span_starts = torch.cumsum(interleaved_span_lengths, 0)[:-1]
    span_start_indicator = segment_sum(
        torch.ones_like(span_starts), span_starts, length)
    span_num = torch.cumsum(span_start_indicator, 0)
    is_noise = torch.eq(span_num % 2, torch.tensor(1, dtype=torch.int64))
    return is_noise[:orig_length]


def denoise(dataset,
            noise_density=0.15,
            noise_mask_fn=None,
            inputs_fn=None,
            targets_fn=None,
            tokenizer=None):
    vocab_size = tokenizer.vocab_size
    def map_fn(features):
        tokens = features['targets']
        noise_mask = noise_mask_fn(tokens.shape[0], noise_density)
        inputs = inputs_fn(tokens, noise_mask, vocab_size)
        if targets_fn:
            targets = targets_fn(tokens, noise_mask, vocab_size)
        else:
            targets = tokens
        return {'inputs': inputs, 'targets': targets}
    return [map_fn(data) for data in dataset]


def noise_span_to_unique_sentinel(tokens, noise_mask, vocab_size):
    prev_token_is_noise = torch.nn.functional.pad(
        noise_mask[:-1], [1, 0])

    first_noise_tokens = torch.logical_and(
        noise_mask, torch.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = torch.logical_and(
        noise_mask, prev_token_is_noise)

    sentinel = vocab_size - torch.cumsum(first_noise_tokens.int(), 0)

    tokens = torch.where(first_noise_tokens, sentinel, tokens)
    return torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))


def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocab_size):
    return noise_span_to_unique_sentinel(
        tokens, torch.logical_not(noise_mask), vocab_size)


"""============= UTILITY FUNCTIONS ==============="""
def shuffle(value):
    """Randomly shuffle a tensor."""
    flat_value = torch.reshape(value, [-1])
    indices = torch.argsort(
        torch.rand(flat_value.shape)
    )
    flat_shuffle = torch.gather(flat_value, 0, indices)
    return torch.reshape(flat_shuffle, value.shape)


def segment_sum(data, segment_ids, num_segments=None):
    """Compute the sum along segments of a tensor."""
    if num_segments is None:
        num_segments = len(torch.unique(segment_ids))
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0],
                                                            *data.shape[1:])

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


"""========== DRIVER CODE =========="""
if __name__ == "__main__":
    # get and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help=".json file containing raw training examples.")
    parser.add_argument("--seq_pack", default=False, action='store_true', help="Whether to concatenate multiple "
                                                                "training examples into longer examples "
                                                                "of length ~512.")
    parser.add_argument("--seed", default=1248, type=int, required=False, help="Random seed to use for "
                                                                               "torch and python.random.")
    parser.add_argument("--labelsemantics", type=str, default="full label", help="Method of including and"
                                                                                 "masking intent labels.")

    args = parser.parse_args()
    
    labelsemantics = None if args.labelsemantics.lower() == "none" else args.labelsemantics

    # SET RANDOM SEED FOR REPLICABLE BEHAVIOR
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    sequence_length = {'inputs': 512, 'targets': 512}
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    utterances, intents = load_data(args.dataset, tokenizer)
    dataset = span_corruption(utterances, intents, sequence_length, seq_pack=args.seq_pack,
                              label_semantics=labelsemantics)


    # write output json
    suffix_start = "fulllab"
    if args.labelsemantics == "multiple choice":
        suffix_start = "multichoice"
    elif args.labelsemantics == "concat":
        suffix_start = "concat"
    elif args.labelsemantics is None:
        suffix_start = "nolabsem"
    suffix = f"_{suffix_start}.tok.json" if not args.seq_pack else f"_{suffix_start}.pack.tok.json"
    out_name = sys.argv[1].split(".json")[0] + suffix
    write_data(dataset, out_name, tokenizer)

    '''
    # for debugging
    for data in dataset[:20]:
        print(f'inputs: {tokenizer.decode(data["inputs"])}\ttargets: {tokenizer.decode(data["targets"])}')
    # print(dataset[:5])
    '''

    print("Num examples:", len(dataset))
