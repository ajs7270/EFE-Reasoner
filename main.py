import os

from lightning.pytorch.trainer import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import DeviceStatsMonitor

from datasets.DataModule import DataModule
from argparse import ArgumentParser
from model.sunny.wrapper_model import WrapperModel

def get_project_args():
    parser = ArgumentParser("Project(Sunny) argument")

    # Experiment argument
    parser.add_argument("--wandb", type=int, default=1, help="use wandb")
    parser.add_argument("--experiment_name", type=str, default="svamp", choices=["mathqa", "svamp"], help="data name")

    # wandb argument
    parser.add_argument("--log_path", type=str, default="log", help="result save directory")
    parser.add_argument("--results_dir", type=str, default="result",
                        help="saves checkpoints to 'some/path/' at every epoch end")

    # reproduction argument
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser.parse_args()

def get_data_args():
    parser = ArgumentParser("Data Module argument")

    # data module argument
    parser.add_argument("--data_path", type=str, default="data/processed/svamp",
                        help="path to the train data")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")

    return parser.parse_args()

def get_model_args():
    parser = ArgumentParser("Model argument")

    # model argument
    parser.add_argument("--bert_model", type=str, default="roberta-large",
                        choices=["roberta-large", "roberta-base", "facebook/npm", "facebook/npm-single",
                                 "witiko/mathberta",
                                 "AnReu/math_pretrained_bert", "AnReu/math_pretrained_roberta"],
                        help="pretrained model name in huggingface")
    parser.add_argument("--lr", type=float, default=1.9e-05, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="optimizer")
    parser.add_argument("--fine_tune", type=int, default=1, help="fine tune the PLM model")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup ratio")

    return parser.parse_args()

def get_trainer_args():
    parser = ArgumentParser("Trainer argument")

    # trainer argument
    parser.add_argument("--devices", type=int, default=-1, help="number of gpus used by accelerator")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["cpu", "gpu", "tpu", "ipu", "auto"],
                        help="choice computing device")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="max grad norm for gradient clipping")
    parser.add_argument("--max_epochs", type=int, default=1000, help="max epoch")
    parser.add_argument("--num_nodes", type=int, default=1, help="number of GPU nodes(computers) for distributed training")
    parser.add_argument("--precision", default="16",
                        choices=['64', '32', '16', 'bf16', 64, 32, 16],
                        help="precision")
    parser.add_argument("--profiler", default="simple", choices=[None, "simple", "advanced"],
                        help="profiler")
    parser.add_argument("--enable_progress_bar", type=bool, default=True, help="enable progress bar")
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_true", choices=["auto", "ddp", "fsdp"],
                        help="strategy for distributed training(ddp: Data-parallel fsdp: model-parallel)")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="log every n steps")
    parser.add_argument("--deterministic", type=bool, default=False,
                        help="This flag sets the torch.backends.cudnn.deterministic flag")

    return parser.parse_args()

def main():
    # set argument
    # ========================================
    project_args = get_project_args()
    data_args = get_data_args()
    model_args = get_model_args()
    trainer_args = get_trainer_args()
    # ========================================

    # set logging
    # ========================================
    logger = None
    if project_args.wandb:
        # if not exist log_path, make directory
        if not os.path.exists(project_args.log_path):
            os.makedirs(project_args.log_path)

        logger = WandbLogger(
            name=f"base_{model_args.bert_model}_{model_args.optimizer}_{data_args.batch_size}_{model_args.lr}",
            project=f"sunny_{project_args.experiment_name}",
            config=vars(model_args),
            save_dir=project_args.log_path)
    # ========================================

    # set parallelism tokenizer
    # ========================================
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # ========================================

    # set seed
    # ========================================
    seed_everything(project_args.seed)
    # ========================================

    # set data module
    # ========================================
    data_module = DataModule(
        data_path=data_args.data_path,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        bert_model=model_args.bert_model
    )
    # ========================================

    # set model
    # ========================================
    model = WrapperModel(
        model_args.bert_model,
        model_args.fine_tune,
        model_args.lr,
        model_args.weight_decay,
        model_args.warmup_ratio,
        model_args.optimizer,
        data_module.train_dataset.constant_ids,
        data_module.train_dataset.operator_ids,
        num_training_steps=len(data_module.train_dataloader()) * trainer_args.max_epochs,
        label_pad_id = data_module.train_dataset.pad_id,
        concat=True,
        dataset_config = data_module.train_dataset.config
    )
    # ========================================

    # set callbacks
    # ========================================
    device_stats_callback = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        dirpath=f"{project_args.results_dir}/checkpoints/",
        filename=f"{project_args.experiment_name}-{model_args.bert_model}-{model_args.optimizer}-"
                 f"{data_args.batch_size}-{model_args.lr}-"
                 "{epoch:02d}-{global_step}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10
    )
    # ========================================

    # set Trainer
    # ========================================
    trainer = Trainer(**vars(trainer_args), logger=logger, callbacks=[device_stats_callback,
                                                                      checkpoint_callback])
    trainer.fit(model, datamodule=data_module)
    # trainer.predict(test_dataset)
    # ========================================

if __name__ == "__main__":
    main()