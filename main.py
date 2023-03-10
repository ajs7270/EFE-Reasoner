from pytorch_lightning.profilers import SimpleProfiler

import os
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning import Trainer
from lightning_fabric import seed_everything

from datasets.DataModule import DataModule
from argparse import ArgumentParser
from model.sunny.wrapper_model import WrapperModel

parser = ArgumentParser("Train for MathQA or SVAMP")

# Experiment argument
parser.add_argument("--experiment_name", type=str, default="mathqa", choices=["mathqa", "svamp"], help="data name")

# wandb argument
parser.add_argument("--log_path", type=str, default="log", help="result save directory")

# data module argument
parser.add_argument("--data_path", type=str, default="data/processed/mathqa",
                    help="path to the train data")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")

# trainer argument
parser.add_argument("--base_root_dir", type=str, default="result", help="saves checkpoints to 'some/path/' at every epoch end")
parser.add_argument("--devices", type=int, default=-1, help="number of gpus used by accelerator")
parser.add_argument("--accelerator", type=str, default="auto", choices=["cpu", "gpu", "tpu", "ipu", "auto"],
                    help="choice computing device")
parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="max grad norm for gradient clipping")
parser.add_argument("--max_epochs", type=int, default=150, help="max epoch")
parser.add_argument("--num_nodes", type=int, default=1, help="number of GPU nodes(computers) for distributed training")
parser.add_argument("--precision", default="bf16",
                    choices=['64', '32', '16', 'bf16', 64, 32, 16],
                    help="precision")
parser.add_argument("--profiler", default="advanced", choices=[None, "simple", "advanced"],
                    help="profiler")
parser.add_argument("--enable_progress_bar", type=bool, default=True, help="enable progress bar")
parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"],
                    help="strategy for distributed training(ddp: Data-parallel fsdp: model-parallel)")
parser.add_argument("--auto_lr_find", type=bool, default=True, help="Runs a learning rate finder algorithm")
parser.add_argument("--auto_scale_batch_size", type=bool, default=True, help="Automatically tries to find the largest batch size that fits into memory, before any training")
parser.add_argument("--log_every_n_steps", type=int, default=500, help="log every n steps")
parser.add_argument("--early_stop_callback", type=bool, default=True, help="early stop callback")


# reproduction argument
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--deterministic", type=bool, default=False,
                    help="This flag sets the torch.backends.cudnn.deterministic flag")

# model argument
parser.add_argument("--fine_tune", type=int, default=0, help="fine tune the PLM model")
parser.add_argument("--bert_model", type=str, default="facebook/npm",
                    choices=["roberta-large", "roberta-base", "facebook/npm", "facebook/npm-single", "witiko/mathberta",
                            "AnReu/math_pretrained_bert", "AnReu/math_pretrained_roberta"],
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
        logger = WandbLogger(name=f"{args.experiment_name}_{args.bert_model}_{args.optimizer}_{args.batch_size}",
                             project="sunny", save_dir=args.log_path)
    # ========================================

    # set parallelism tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # set seed
    # ========================================
    seed_everything(args.seed)
    # torch.use_deterministic_algorithms(args.deterministic)

    # set data module
    # ========================================
    data_module = DataModule(data_path=args.data_path,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             bert_model=args.bert_model)
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
        data_module.train_dataset.constant_ids,
        data_module.train_dataset.operator_ids,
        num_training_steps=len(data_module.train_dataloader()) * args.max_epochs,
        label_pad_id = data_module.train_dataset.pad_id,
        concat=True,
        dataset_config = data_module.train_dataset.config
    )
    # ========================================

    # set Trainer
    trainer = Trainer.from_argparse_args(args, logger=logger)
    trainer.tune(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    # ========================================

    #trainer.predict(test_dataset)


if __name__ == "__main__":
    main()
