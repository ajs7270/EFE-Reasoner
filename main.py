from typing import Union, Optional
from pytorch_lightning.profilers import SimpleProfiler

import numpy as np
import random
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from lightning_fabric import seed_everything
from datasets.dataset import Dataset
from argparse import ArgumentParser
from model.sunny.wrapper_model import WrapperModel

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = ArgumentParser("Train for MathQA or SVAMP")

# Experiment argument
parser.add_argument("--experiment_name", type=str, default="mathqa", choices=["mathqa", "svamp"], help="data name")

# dataloader argument
parser.add_argument("--train_data_path", type=str, default="data/processed/mathqa/train.json",
                    help="path to the train data")
parser.add_argument("--dev_data_path", type=str, default="data/processed/mathqa/dev.json",
                    help="path to the dev data")
parser.add_argument("--test_data_path", type=str, default="data/processed/mathqa/test.json",
                    help="path to the test data")
parser.add_argument("--configure_path", type=str, default="data/processed/mathqa/config.json",
                    help="path to the configure file")
parser.add_argument("--train_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--test_batch_size", type=int, default=32, help="batch size")

# trainer argument
parser.add_argument("--devices", type=int, default=4, help="number of workers for dataloader")
parser.add_argument("--accelerator", type=str, default="auto", choices=["cpu", "gpu", "tpu", "ipu", "auto"],
                    help="choice computing device")
parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="max grad norm for gradient clipping")
parser.add_argument("--max_epochs", type=int, default=10, help="max epoch")
parser.add_argument("--num_nodes", type=int, default=1, help="Number of GPU nodes for distributed training")
parser.add_argument("--precision", type=Union[int, str], default="16-mixed",
                    choices=[16, 32, 64, "16-mixed", "bf16-mixed"],
                    help="precision")
parser.add_argument("--profiler", type=Optional[str], default="simple", choices=[None, "simple", "advanced"],
                    help="profiler")
parser.add_argument("--enable_progress_bar", type=bool, default=True, help="enable progress bar")
parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"],
                    help="strategy for distributed training(ddp: Data-parallel fsdp: model-parallel)")

# parser.add_argument("--model_save_dir", type=str, default="model_save", help="model save directory")
# parser.add_argument("--result_path", type=str, default="result", help="result save directory")

# reproduction argument
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--deterministic", type=bool, default=False,
                    help="This flag sets the torch.backends.cudnn.deterministic flag")

# model argument
parser.add_argument("--fine_tune", type=int, default=0, help="fine tune the PLM model")
parser.add_argument("--bert_model", type=str, default="roberta-base",
                    choices=["roberta-large", "roberta-base", "t5-small", "t5-large"],
                    help="pretrained model name in huggingface")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup ratio")
parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="optimizer")

# logging with wandb
parser.add_argument("--wandb", type=int, default=0, help="use wandb")
parser.add_argument("--auth_path", type=str, default="auth_key.json", help="path to the auth.json for wandb")

def main():
    args = parser.parse_args()

    # set logging
    # ========================================
    logger = None
    if args.wandb:
        logger = WandbLogger(project="sunny", name=args.experiment_name, save_dir=args.result_path)
    # ========================================

    # set seed
    # ========================================
    seed_everything(args.seed)
    # torch.use_deterministic_algorithms(args.deterministic)

    # set dataset
    # ========================================
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_dataset = Dataset(args.train_data_path, args.configure_path, args.bert_model)
    dev_dataset = Dataset(args.dev_data_path, args.configure_path, args.bert_model)
    test_dataset = Dataset(args.test_data_path, args.configure_path, args.bert_model)
    # ========================================

    # set dataloader
    # ========================================
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  worker_init_fn=seed_worker,
                                  generator=g)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                worker_init_fn=seed_worker,
                                generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 worker_init_fn=seed_worker,
                                 generator=g)
    # ========================================

    # set model
    # ========================================
    model = WrapperModel(
        args.bert_model,
        args.fine_tune,
        args.lr,
        args.weight_decay,
        args.warmup_ratio,
        args.optimizer,
        train_dataset.constant_ids,
        train_dataset.operator_ids,
    )
    # ========================================

    # set Trainer
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=dev_dataloader)
    # ========================================

    trainer.predict(test_dataset)


if __name__ == "__main__":
    main()