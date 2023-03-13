import os
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets.dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, num_workers: int, bert_model: str):
        super().__init__()
        self.train_data_path = Path(data_path, "train.json")
        self.dev_data_path = Path(data_path, "dev.json")
        self.test_data_path = Path(data_path, "test.json")
        if not os.path.exists(self.test_data_path):
            self.test_data_path = self.dev_data_path
        self.configure_path = Path(data_path, "config.json")
        self.bert_model = bert_model
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = Dataset(self.train_data_path, self.configure_path, self.bert_model)
        self.dev_dataset = Dataset(self.dev_data_path, self.configure_path, self.bert_model)
        self.test_dataset = Dataset(self.test_data_path, self.configure_path, self.bert_model)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      collate_fn=self.train_dataset.collate_function)
        return train_dataloader

    def val_dataloader(self):
        dev_dataloader = DataLoader(self.dev_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    collate_fn=self.dev_dataset.collate_function)
        return dev_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     collate_fn=self.test_dataset.collate_function)
        return test_dataloader
