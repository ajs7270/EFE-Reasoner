from unittest import TestCase
from dataset import Dataset, Problem, Feature
import torch


class TestDataset(TestCase):
    dataset = Dataset("data/processed/mathqa/train.json", "data/processed/mathqa/train_constant_list.json",
                      "roberta-base")

    def test_dataset(self):
        pass
