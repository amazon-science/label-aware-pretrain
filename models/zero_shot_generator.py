import torch

import numpy as np

from transformers.pipelines.base import ArgumentHandler, Pipeline
from transformers.tokenization_utils import TruncationStrategy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from typing import List, Union

class ConstrainedGenerationArgumentHandler(ArgumentHandler):
    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",")]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")

        '''
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )
        '''

        if isinstance(sequences, str):
            sequences = [sequences]
        labels = self._parse_labels(labels)

        utterances = []
        intents = []
        for sequence in sequences:
            for label in labels:
                '''
                utterances.append(sequence + " <extra_id_0>")
                intents.append("<extra_id_0> " + label)
                '''
                utterances.append(f"mnli hypothesis: This sentence is {label}. premise: " + \
                                  sequence)
                intents.append(hypothesis_template)
            # sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return utterances, intents


class ConstrainedGenerationPipeline(Pipeline):
    def __init__(self, model, args_parser=ConstrainedGenerationArgumentHandler(), *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self._args_parser = args_parser
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, from_tf=bool(".ckpt" in model))

    def _parse_and_tokenize(self, sequences, candidate_labels,
                            hypothesis_template, padding=True,
                            add_special_tokens=True,
                            truncation=TruncationStrategy.ONLY_FIRST,
                            **kwargs):
        utterances, intents = self._args_parser(sequences, candidate_labels, hypothesis_template)
        utterances = self.tokenizer(utterances, add_special_tokens=add_special_tokens,
                                return_tensors='pt', padding=padding,
                                truncation=truncation).input_ids
        intents = self.tokenizer(intents, add_special_tokens=add_special_tokens,
                                 return_tensors='pt', padding=padding,
                                 truncation=truncation).input_ids
        return utterances, intents

    def __call__(self, sequences: Union[str, List[str]],
                 candidate_labels, hypothesis_template="entailment",
                 **kwargs):
        if sequences and isinstance(sequences, str):
            sequences = [sequences]

        seq_scores = []
        with torch.no_grad():
            for sequence in sequences:
                inputs, targets = self._parse_and_tokenize(sequence, candidate_labels,
                                                  hypothesis_template)
                neg_losses = []
                for input, target in zip(inputs, targets):
                    loss = self.model(input_ids=input.unsqueeze(0), labels=target.unsqueeze(0)).loss
                    neg_losses.append(-loss)
                scores = np.exp(neg_losses) / np.exp(neg_losses).sum(-1, keepdims=True)
                seq_scores.append(scores)
        return {'inputs': sequences, 'labels': candidate_labels, 'scores': seq_scores}