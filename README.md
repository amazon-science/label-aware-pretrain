# Label Semantic Aware Pre-training for Goal-oriented Dialogue

This repository contains code for replicating the experiments of our project, Label Semantic Aware Pre-training for Goal-oriented Dialogue. In particular, we focus on few-shot intent classification.

Training scripts are located in `scripts/`. PyTorch code is located in `models/`. Data and preprocessing scripts are located in `data/`.


## Dependencies

- python==3.8.0
- torch==1.8.1
- transformers==4.6.0

Use the `environment.yml` file to replicate our conda environment: `conda env create -f environment.yml`.

## Data Format
For our experiments, we convert the intent labels into a natural language format like "Book hotel" or "airline". Each dataset is preprocessed into .json files, where train, dev, and test examples are of the following format:

```
{"translation": {"src": <utterance>, "tgt": <intent>, "prefix": "intent classification: "}}
```
The placement of the MASK tokens depends on the format used during preprocessing.

### Preprocessing
Our preprocessing script is in `models/preprocessor.py`. This script roughly replicates the preprocessing logic of the TensorFlow code used in the original T5 paper, while also adding additional preprocessing methods. We found **label noising** to be the most effective approach, and the most robust to noisy data. To run preprocessing, first ensure that you have a .json file in the format specified above.

Then, run the following command:
```
python preprocessor.py <json_data> --labelsemantics <label_semantics_type>
```
where `<label_semantics_type>` can be one of the following:
- concat: append intents to utterances, noise spans in the source sequence using the same 15% span noising approach as T5. Reconstruct noised spans in the target sequence.
- full label: append MASK token after utterance in source sequence. Target sequence is the intent.
- multiple choice: use the format "intents: <list_of_intents>. utterance: <utterance>" in the source sequence. Target sequence is the correct intent. During pre-training we randomly select 2 to 29 intents in addition to the correct one, shuffle the list, and present that to the model, where the model must learn to choose the correct one.
- none: just present utterances to the model---no intents. Randomly noise spans in the source sequence using the same approach as T5, and reconstruct the noised spans in the target sequence.

This will output a file in the same directory as the `<json_data>` file, with the `.tok` suffix added as well as a suffix indicating the label semantics type.


## Continued Pre-training
To run continued pre-training, use the `cpt` bash scripts in the `scripts` folder. These scripts call the `models/run_cpt.py` script, which performs continued pre-training (i.e., a second stage of pre-training after the first) using T5 as the pre-trained base model. We use a batch size of 128 and initial learning rate of 5e-4 for our experiments, using defaults otherwise.

We generally find that the best results occur at epochs 3--4, though we train our models for 10 epochs such that we can verify convergence.


## Evaluation
Use the `finetune` scripts in the `scripts` folder to run fine-tuning on the evaluation sets. We use `finetune_t5_full.sh` to run evaluation in the full-resource setting. 

To replicate our evaluation setup in the low-resource case, use `analysis/fewshot_graph.py` by moving to the `analysis` directory and running the following command:

```
python fewshot_graph.py <fewshot_dir> <test_file>
```
where `<fewshot_dir>` is a directory containing files named `train_<num_examples>_examples.json`, with `<num_examples>` being the number of utterances per intent. This script will produce the following:
- A graph of intent classification accuracy at each few-shot split size, where accuracy is averaged across 5 random seeds for each split size. Also plots std. dev. in shaded regions around the means.
- Macroaveraged IC accuracies across split sizes for each model.
- A table of statistical significances between the mean IC accuracies of each model at each split size.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


