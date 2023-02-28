from unittest import TestCase
from model.sunny.wrapper_model import WrapperModel
import torch

class WrapperModelTest(TestCase):
    model = WrapperModel()
    def test__calculate_operator_loss(self):
        logits = torch.rand(2, 3, 4)
        labels = torch.randint(4, size = (2, 3))
        loss = self.model._calculate_operator_loss(logits, labels)
        self.assertEqual(loss.shape, (6,))

    def test__calculate_operatand_loss(self):
        logits = torch.rand(2, 3, 4, 5)
        labels = torch.randint(5, size = (2, 3, 4))
        loss = self.model._calculate_operand_loss(logits, labels)
        self.assertEqual(loss.shape, (24,))

