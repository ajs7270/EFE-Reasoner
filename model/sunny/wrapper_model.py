import torch
import torchmetrics
from torch import nn

import lightning.pytorch as pl
from torch.optim import Optimizer
# from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoModel, AutoConfig, get_cosine_schedule_with_warmup

from datasets.dataset import Feature
from model.sunny.aware_decoder import AwareDecoder
from utils import equationAccuracy


class WrapperModel(pl.LightningModule):
    def __init__(self,
                 bert_model: str = "roberta-base",
                 fine_tune: int = 0,
                 lr: float = 1e-5,
                 weight_decay: float = 0.0,
                 warmup_ratio: float = 0.1,
                 optimizer: str = "adamw",
                 constant_ids: list[torch.Tensor] = None,
                 operator_ids: list[torch.Tensor] = None,
                 num_training_steps: int = 150000,
                 label_pad_id: int = 1,
                 concat: bool = True,
                 dataset_config = None
                 ):
        super(WrapperModel, self).__init__()

        # equivalent automatic hyperparameter assignment
        # assign : self.hparams = {"bert_model": bert_model, "fine_tune": fine_tune ... } : dict[str, Any]
        self.save_hyperparameters()

        # set metric
        self.train_accuracy = equationAccuracy()
        self.train_operator_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                       num_classes=len(operator_ids))
        self.train_operand_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                      num_classes=len(constant_ids)
                                                                  + dataset_config["max_numbers_size"]
                                                                  + dataset_config["max_operators_size"])
        self.validation_accuracy = equationAccuracy()
        self.validation_operator_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                             num_classes=len(operator_ids))
        self.validation_operand_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                            num_classes=len(constant_ids)
                                                                        + dataset_config["max_numbers_size"]
                                                                        + dataset_config["max_operators_size"])
        self.test_accuracy = equationAccuracy()
        self.test_operator_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                                  num_classes=len(operator_ids))
        self.test_operand_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                                 num_classes=len(constant_ids)
                                                                             + dataset_config["max_numbers_size"]
                                                                             + dataset_config["max_operators_size"])

        self.loss = nn.CrossEntropyLoss()

        # set encoder
        self.encoder = AutoModel.from_pretrained(self.hparams["bert_model"])
        self.config = AutoConfig.from_pretrained(self.hparams["bert_model"])

        # pretrained language model은 fine-tuning하고 싶지 않을 때
        if not self.hparams["fine_tune"]:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # set constant_list_embedding
        constant_vectors = self._get_vectors(constant_ids, concat=concat) # Tensor [N_C, H*2] or [N_C, H]
        # set operator_list_embedding
        operator_vectors = self._get_vectors(operator_ids, concat=concat) # Tensor [N_O, H*2] or [N_O, H]

        # set decoder
        self.decoder = AwareDecoder(input_hidden_dim=self.config.hidden_size,
                                    operator_vector=operator_vectors,
                                    const_vector=constant_vectors,
                                    operator_num=len(operator_ids),
                                    const_num=len(constant_ids),
                                    max_number_size=dataset_config["max_numbers_size"],
                                    max_equation=dataset_config["max_operators_size"],
                                    max_arity=max(map(max, dataset_config['operator_dict'].values())),
                                    label_pad_id=label_pad_id,
                                    concat=concat)

    def _get_vectors(self, ids_list: list[torch.Tensor], concat: bool) -> torch.Tensor:
        # return the sum or concatenation of first and last hidden_state of constant_ids
        # ids_list can be constant_ids or operator_ids
        vectors = [] # list(torch.Tensor[H]) or list(torch.Tensor[H*2]) according to concat
        for ids in ids_list:
            if concat:
                # 만약 첫번째 id와 마지막 id가 같은 const_XXX의 경우에는 구분할 수 없다는 문제가 존재
                vectors.append(torch.cat((self.encoder(ids.unsqueeze(0)).last_hidden_state[0, 0, :],
                               self.encoder(ids.unsqueeze(0)).last_hidden_state[0, -1, :])))
            else:
                vectors.append(self.encoder(ids.unsqueeze(0)).last_hidden_state[0, 0, :] +
                               self.encoder(ids.unsqueeze(0)).last_hidden_state[0, -1, :])

        return torch.stack(vectors) # [N_C, H*2] or [N_C, H] according to concat

    def forward(self, x: Feature):
        encoder_output = self.encoder(x.input_ids).last_hidden_state
        operator_logit, operand_logit = self.decoder(encoder_output, x.attention_mask, x.question_mask, x.number_mask)

        return operator_logit, operand_logit  # [[B, T, N_O], [B, T, A, N_D]] : Operator, Operand prediction

    def _calculate_operator_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                                 op_fin: list[int]) -> torch.Tensor:
        bsz, _, _ = logits.shape #[B, T, N_O]

        loss = None
        loss_count = 0
        for i in range(bsz):
            if loss is None:
                loss = self.loss(torch.reshape(logits[i, :op_fin[i], :], (op_fin[i], -1)),
                                 torch.reshape(labels[i, :op_fin[i]], (-1,)))
            else:
                loss += self.loss(torch.reshape(logits[i, :op_fin[i], :], (op_fin[i], -1)),
                                  torch.reshape(labels[i, :op_fin[i]], (-1,)))
            loss_count += 1

        return loss / loss_count


    def _calculate_operand_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                                op_fin: list[int], oe_fin: list[list[int]]) -> torch.Tensor:
        bsz, max_operator_len, max_arity, _ = logits.shape  # [B, T, A, N_D]

        loss = None
        loss_count = 0
        for i in range(bsz):
            for j in range(op_fin[i]):
                if loss is None:
                    loss = self.loss(torch.reshape(logits[i, :op_fin[i], :oe_fin[i][j], :], (op_fin[i]*oe_fin[i][j], -1)),
                             torch.reshape(labels[i, :op_fin[i], :oe_fin[i][j]], (-1,)))
                else:
                    loss += self.loss(torch.reshape(logits[i, :op_fin[i], :oe_fin[i][j], :], (op_fin[i]*oe_fin[i][j], -1)),
                             torch.reshape(labels[i, :op_fin[i], :oe_fin[i][j]], (-1,)))
                loss_count += 1

        return loss / loss_count

    def _get_operator_finish_indexes(self, operator_label: torch.Tensor) -> list[int]:
        op_fin = []
        none_index = 0
        for i in range(operator_label.shape[0]):
            none_index_list = torch.where(operator_label[i] == none_index)[0]
            if len(none_index_list) == 0:
                op_fin.append(operator_label.shape[1])
            else:
                if none_index_list[0] == 0:
                    op_fin.append(1)
                else:
                    op_fin.append(none_index_list[0])

        return op_fin

    def _get_operand_finish_indexes(self, operand_label: torch.Tensor, op_fin: list[int]) -> list[list[int]]:
        oe_fin = []
        none_index = 0
        for i, fin in enumerate(op_fin):
            oe_fin.append([])
            for j in range(fin):
                none_index_list = torch.where(operand_label[i,j,:] == none_index)[0]
                if len(none_index_list) == 0:
                    oe_fin[i].append(operand_label.shape[2])
                else:
                    if none_index_list[0] == 0:
                        oe_fin[i].append(1)
                    else:
                        oe_fin[i].append(none_index_list[0])

        return oe_fin

    def _calculate_loss(self, batch, operator_logit, operand_logit):
        gold_operator_label = batch.operator_label - 1  # 0 is reserved for unknown, 1 is padding included in loss
        gold_operand_label = batch.operand_label - 1  # 0 is reserved for unknown, 1 is padding included in loss

        # operator finish_indexes
        op_fin = self._get_operator_finish_indexes(gold_operator_label)
        # operand finish_indexes
        oe_fin = self._get_operand_finish_indexes(gold_operand_label, op_fin)

        operator_loss = self._calculate_operator_loss(operator_logit, gold_operator_label, op_fin)
        operand_loss = self._calculate_operand_loss(operand_logit, gold_operand_label, op_fin, oe_fin)

        return operator_loss, operand_loss

    def _calculate_accuracy(self, batch, operator_logit, operand_logit, type: str ="train"):

        gold_operator_label = batch.operator_label - 1  # 0 is reserved for unknown, 1 is padding included in loss
        gold_operand_label = batch.operand_label - 1  # 0 is reserved for unknown, 1 is padding included in loss

        # operator finish_indexes
        op_fin = self._get_operator_finish_indexes(gold_operator_label)
        # operand finish_indexes
        oe_fin = self._get_operand_finish_indexes(gold_operand_label, op_fin)
        # Operator none index
        none_index = 0

        batch_size = operator_logit.shape[0]
        for i in range(batch_size):
            preds = torch.Tensor().to(self.device)  # to calculate accuracy
            golds = torch.Tensor().to(self.device)

            preds = torch.concat((preds, torch.argmax(operator_logit[i, :op_fin[i], :], dim=1)))
            golds = torch.concat((golds, gold_operator_label[i, :op_fin[i]]))

            if type == "train":
                self.train_operator_accuracy(operator_logit[i, :op_fin[i], :], gold_operator_label[i, :op_fin[i]])

                # 만약 마지막 operator none 이라면 이 operator에 해당하는 operand는 정답률을 계산할 때 사용하지 않음
                if golds[-1] != none_index:
                    num_operand = op_fin[i]

                    for j in range(num_operand):
                        preds = torch.concat((preds, torch.argmax(operand_logit[i, j, :oe_fin[i][j], :], dim=1)))
                        golds = torch.concat((golds, gold_operand_label[i, j, :oe_fin[i][j]]))

                        self.train_operand_accuracy(operand_logit[i, j, :oe_fin[i][j], :], gold_operand_label[i, j, :oe_fin[i][j]])
                self.train_accuracy(preds, golds)
            elif type == "validation":
                self.validation_operator_accuracy(operator_logit[i, :op_fin[i], :], gold_operator_label[i, :op_fin[i]])

                # 만약 마지막 operator none 이라면 이 operator에 해당하는 operand는 정답률을 계산할 때 사용하지 않음
                if golds[-1] != none_index:
                    num_operand = op_fin[i]
                    for j in range(num_operand):
                        preds = torch.concat((preds, torch.argmax(operand_logit[i, j, :oe_fin[i][j], :], dim=1)))
                        golds = torch.concat((golds, gold_operand_label[i, j, :oe_fin[i][j]]))

                        self.validation_operand_accuracy(operand_logit[i, j, :oe_fin[i][j], :],
                                                    gold_operand_label[i, j, :oe_fin[i][j]])
                self.validation_accuracy(preds, golds)
            elif type == "test":
                self.test_operator_accuracy(operator_logit[i, :op_fin[i], :], gold_operator_label[i, :op_fin[i]])

                # 만약 마지막 operator none 이라면 이 operator에 해당하는 operand는 정답률을 계산할 때 사용하지 않음
                if golds[-1] != none_index:
                    num_operand = op_fin[i]
                    for j in range(num_operand):
                        preds = torch.concat((preds, torch.argmax(operand_logit[i, j, :oe_fin[i][j], :], dim=1)))
                        golds = torch.concat((golds, gold_operand_label[i, j, :oe_fin[i][j]]))

                        self.test_operand_accuracy(operand_logit[i, j, :oe_fin[i][j], :],
                                                         gold_operand_label[i, j, :oe_fin[i][j]])
                self.test_accuracy(preds, golds)
    def training_step(self, batch: Feature, batch_idx: int) -> torch.Tensor:
        operator_logit, operand_logit = self(batch)  # [B, T, N_O + 1], [B, T, A, N_D + 1]

        operator_loss, operand_loss = self._calculate_loss(batch, operator_logit, operand_logit)

        self._calculate_accuracy(batch, operator_logit, operand_logit, "train")
        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_operator_accuracy", self.train_operator_accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_operand_accuracy", self.train_operand_accuracy, on_step=True, on_epoch=True, sync_dist=True)

        self.log("train_operator_loss", operator_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_operand_loss", operand_loss, on_step=True, on_epoch=True, sync_dist=True)

        loss = operator_loss + operand_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Feature, batch_idx: int) -> torch.Tensor:
        operator_logit, operand_logit = self(batch)  # [B, T, N_O + 1], [B, T, A, N_D + 1]

        operator_loss, operand_loss = self._calculate_loss(batch, operator_logit, operand_logit)

        self._calculate_accuracy(batch, operator_logit, operand_logit, "validation")
        self.log("val_accuracy", self.validation_accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_operator_accuracy", self.validation_operator_accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_operand_accuracy", self.validation_operand_accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_operator_loss", operator_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_operand_loss", operand_loss, on_step=True, on_epoch=True, sync_dist=True)

        loss = operator_loss + operand_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: Feature, batch_idx: int) -> torch.Tensor:
        operator_logit, operand_logit = self(batch)  # [B, T, N_O + 1], [B, T, A, N_D + 1]

        operator_loss, operand_loss = self._calculate_loss(batch, operator_logit, operand_logit)

        self._calculate_accuracy(batch, operator_logit, operand_logit, "test")

        self.log("test_accuracy", self.test_accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test_operator_accuracy", self.test_operator_accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test_operand_accuracy", self.test_operand_accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test_operator_loss", operator_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test_operand_loss", operand_loss, on_step=True, on_epoch=True, sync_dist=True)

        loss = operator_loss + operand_loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # def predict_step(self, batch: Feature, batch_idx: int, dataloader_idx: int = 0) -> Any:
    #     return self.model(batch)

    def configure_optimizers(self) -> tuple[list[Optimizer], list["_LRScheduler"]]:
        optims = []
        schedulers = []
        if self.hparams.optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "adamw":
            optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError

        optims.append(optim)
        schedulers.append(get_cosine_schedule_with_warmup(optim,
                                        num_warmup_steps=self.hparams.warmup_ratio * self.hparams.num_training_steps,
                                        num_training_steps=self.hparams.num_training_steps))

        return optims, schedulers
