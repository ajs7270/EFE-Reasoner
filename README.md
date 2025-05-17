# EFE-Reasoner

This repository contains the research code for the **Explicit Feature Extraction (EFE) Reasoner**. The EFE Reasoner extends the Deductive Reasoner architecture with explicit numerical features so that a language model can better capture comparisons between numbers in math word problems. The approach leads to improved accuracy on the SVAMP benchmark for word-problem solving.

## Features
- Adds explicit numerical tokens to the Deductive Reasoner framework.
- Supports pretrained language models such as RoBERTa for token embeddings.
- Demonstrated accuracy gains on SVAMP compared with the baseline Deductive Reasoner.

## Setup
Python package requirements are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Example Usage
A minimal training example is shown below.

```bash
python main.py \
  --wandb 0 \
  --data_path data/processed/svamp \
  --bert_model roberta-base
```

Hyperparameters can be adjusted via the `get_*_args` functions in `main.py`.

## Data
The repository expects preprocessed math word-problem datasets under `data/`. The SVAMP dataset was used in our experiments. You may need to construct the dataset locally before running the code.

## Code Structure
- `datasets/` – dataset loaders
- `model/` – implementation of the EFE Reasoner (`wrapper_model.py`, etc.)
- `main.py` – entry point for training and evaluation

## License
This project is licensed under the MIT License. See `LICENSE` for details.
