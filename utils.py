import torch
from torchmetrics import Metric

class equationAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.correct += torch.equal(preds,target)
        self.total += 1

    def compute(self):
        return self.correct.float() / self.total