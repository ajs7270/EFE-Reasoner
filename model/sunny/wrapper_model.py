from typing import Any, Union

import torch
from torch import nn

import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, get_cosine_schedule_with_warmup

from datasets.dataset import Feature
from model.sunny.aware_decoder import AwareDecoder


class WrapperModel(pl.LightningModule):
    def __init__(self,
                 bert_model: str = "roberta-base",
                 fine_tune: int = 0,
                 lr: float = 1e-5,
                 weight_decay: float = 0.0,
                 warmup_ratio: float = 0.1,
                 optimizer: str = "adamw"
                 ):
        super(WrapperModel, self).__init__()

        # equivalent automatic hyperparameter assignment
        # assign : self.hparams = {"bert_model": bert_model, "fine_tune": fine_tune ... } : dict[str, Any]
        self.save_hyperparameters()

        # set metric
        self.metric = nn.CrossEntropyLoss()

        # set encoder
        self.encoder = AutoModel.from_pretrained(self.bert_model)
        self.config = AutoConfig.from_pretrained(self.bert_model)

        # pretrained language model은 fine-tuning하고 싶지 않을 때
        if not self.fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # set decoder
        self.decoder = AwareDecoder(input_hidden_dim=self.config.hidden_size)

    def forward(self, x: Feature):
        encoder_output = self.encoder(x.input_ids).last_hidden_state
        operator_logit, operand_logit = self.decoder(encoder_output)

        return operator_logit, operand_logit  # [[B, T, 1], [B, T, A]] : Operator, Operand prediction

    def _calculate_operator_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass

    def _calculate_operand_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch: Feature, batch_idx: int) -> torch.Tensor:
        gold_operator_label = batch.operator_label
        gold_operand_label = batch.operand_label

        operator_logit, operand_logit = self(batch)

        operator_loss = self._calculate_operator_loss(operator_logit, gold_operator_label)
        operand_loss = self._calculate_operand_loss(operand_logit, gold_operand_label)

        loss = operator_loss + operand_loss

        return loss

    def configure_optimizers(self) -> tuple[list[str], list[str]]:
        optims = []
        schedulers = []
        if self.hparams.optimizeroptimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "adamw":
            optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError

        return optims, schedulers
