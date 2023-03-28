from unittest import TestCase

import torch
from model.sunny.aware_decoder import AwareDecoder


class AwareDecoderTest(TestCase):
    input_hidden_dim = 512
    num_layers = 4
    operator_num = 9  # unk + operator + none
    constant_num = 5
    max_equation = 4
    max_arity = 3
    max_number_size = 5
    label_pad_id = 1
    concat = True

    const_vectors = torch.rand(constant_num, input_hidden_dim * 2)
    op_vectors = torch.rand(operator_num, input_hidden_dim * 2)

    model = AwareDecoder(input_hidden_dim=input_hidden_dim,
                         num_layers = num_layers,
                         operator_num=operator_num,
                         const_num=constant_num,
                         operator_vector=op_vectors,
                         const_vector=const_vectors,
                         max_arity=max_arity,
                         max_equation=max_equation,
                         max_number_size=max_number_size,
                         label_pad_id=label_pad_id,
                         concat=concat)

    def test_forward(self):
        input = torch.rand(5, 11, 512)
        number_mask = torch.Tensor([[1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0],
                                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 2, 2, 0, 3, 3, 0, 0, 0]])
        attention_mask = torch.rand(5, 11)
        question_mask = torch.rand(5, 11)
        gold_operators = torch.randint(1, 3, (5, self.max_equation)) # [B, T]
        gold_operands = torch.randint(1, 3, (5, self.max_equation, self.max_arity)) # [B, T, A]
        self.model.train()
        self.model(
            input=input,
            attention_mask=attention_mask,
            question_mask=question_mask,
            number_mask=number_mask,
            gold_operators = gold_operators,
            gold_operands = gold_operands
        )
        self.model.eval()
        self.model(
            input=input,
            attention_mask=attention_mask,
            question_mask=question_mask,
            number_mask=number_mask,
            gold_operators=gold_operators,
            gold_operands=gold_operands
        )

    def test__get_num_vec(self):
        input = torch.rand(5, 11, 512)
        number_mask = torch.Tensor([[1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0],
                                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 2, 2, 0, 3, 3, 0, 0, 0]])
        max_number_size = 5
        self.model._get_num_vec(input=input, number_mask=number_mask, max_number=max_number_size, concat=True)
