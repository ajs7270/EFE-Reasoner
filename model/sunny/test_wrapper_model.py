from unittest import TestCase
from model.sunny.wrapper_model import WrapperModel
import torch


class WrapperModelTest(TestCase):
    batch_size = 3
    max_equation = 3
    max_arity = 2

    dataset_config = {
        "max_numbers_size" : 12,
        "max_operators_size" : max_equation,
        "operator_dict" : {
            "add" : [2],
            "sub" : [2],
            "mul" : [2],
            "div" : [2]
        }
    }
    model = WrapperModel(constant_ids=torch.Tensor([[3], [2], [3], [4], [5]]).long(),
                         operator_ids=torch.Tensor([[3], [2], [3], [4], [5], [6], [7], [8], [9], [10]]).long(),
                         dataset_config=dataset_config)

    operator_num = 10  # none + operator
    operand_num = 20  # none + const + number previous result

    operator_logit = torch.rand(batch_size, max_equation, operator_num)
    operand_logit = torch.rand(batch_size, max_equation, max_arity, operand_num)

    gold_operator_label = torch.Tensor(
        [[2, 0, 0],
         [2, 3, 0],
         [3, 2, 1]]
    ).long()

    gold_operand_label = torch.Tensor(
        [[[1, 1],
          [0, 0],
          [0, 0]],
         [[3, 2],
          [1, 2],
          [0, 0]],
         [[2, 1],
          [1, 1],
          [1, 1]]]
    ).long()

    op_fin = model._get_operator_finish_indexes(gold_operator_label)
    oe_fin = model._get_operand_finish_indexes(gold_operand_label, op_fin)
    def test__calculate_loss(self):
        operator_loss = self.model._calculate_operator_loss(self.operator_logit, self.gold_operator_label, self.op_fin)
        operand_loss = self.model._calculate_operand_loss(self.operand_logit, self.gold_operand_label, self.op_fin, self.oe_fin)

    def test__calculate_accuracy(self):
        none_index = 0
        batch_size = self.operator_logit.shape[0]
        for i in range(batch_size):
            preds = torch.Tensor()  # to calculate accuracy
            golds = torch.Tensor()

            preds = torch.concat((preds, torch.argmax(self.operator_logit[i, :self.op_fin[i], :], dim=1)))
            golds = torch.concat((golds, self.gold_operator_label[i, :self.op_fin[i]]))


            self.model.train_operator_accuracy(self.operator_logit[i, :self.op_fin[i], :], self.gold_operator_label[i, :self.op_fin[i]])

            # 만약 마지막 operator none 이라면 이 operator에 해당하는 operand는 정답률을 계산할 때 사용하지 않음
            if golds[-1] != none_index:
                num_operand = self.op_fin[i]

                for j in range(num_operand):
                    preds = torch.concat((preds, torch.argmax(self.operand_logit[i, j, :self.oe_fin[i][j], :], dim=1)))
                    golds = torch.concat((golds, self.gold_operand_label[i, j, :self.oe_fin[i][j]]))

                    self.model.train_operand_accuracy(self.operand_logit[i, j, :self.oe_fin[i][j], :],
                                                self.gold_operand_label[i, j, :self.oe_fin[i][j]])
            self.model.train_accuracy(preds, golds)