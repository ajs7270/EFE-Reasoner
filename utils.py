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

# A dictionary to convert the index to the corresponding word
ORDER = {0: '<first>', 1: '<second>', 2: '<third>', 3: '<fourth>', 4: '<fifth>', 5: '<sixth>', 6: '<seventh>',
         7: '<eighth>', 8: '<ninth>', 9: '<tenth>', 10: '<eleventh>', 11: '<twelfth>', 12: '<thirteenth>',
         13: '<fourteenth>', 14: '<fifteenth>', 15: '<sixteenth>', 16: '<seventeenth>', 17: '<eighteenth>',
         18: '<nineteenth>', 19: '<twentieth>', 20: '<twenty-first>', 21: '<twenty-second>', 22: '<twenty-third>',
         23: '<twenty-fourth>', 24: '<twenty-fifth>', 25: '<twenty-sixth>'}