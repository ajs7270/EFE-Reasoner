# EFE Reasoner

This repository contains the implementation of the **Explicit Feature Extraction (EFE) Reasoner**, a model designed to improve reasoning about numerical magnitudes in math word problems. The method extends the Deductive-Reasoner architecture by explicitly providing token embeddings for numbers, enabling the model to capture ordering relationships between quantities.

The approach is described in the paper ["Explicit Feature Extraction(EFE) Reasoner: A model for Understanding the Relationship between Numbers by Size"](https://kiss.kstudy.com/Detail/Ar?key=4059173).

## Features
- Adds explicit number features to the Deductive-Reasoner framework
- Uses pre-trained language models such as RoBERTa to embed problem text
- Demonstrated accuracy gains on the SVAMP dataset

## Setup
Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Example Usage
Run training with the default configuration:

```bash
python main.py \
  --wandb 0 \
  --data_path data/processed/svamp \
  --bert_model roberta-base
```

Hyperparameters can be adjusted in `main.py` via the `get_*_args` functions.

## Data
The repository expects preprocessed SVAMP-style datasets in the `data/` directory. Prepare the dataset separately before running the code.

## Code Structure
- `datasets/` – dataset loading utilities
- `model/` – EFE Reasoner implementation (`wrapper_model.py` etc.)
- `main.py` – training and evaluation script

## License
This project is licensed under the MIT License. See `LICENSE` for details.
