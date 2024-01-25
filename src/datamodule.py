from pathlib import Path
from typing import Dict, Optional

import torch
from clearml import Dataset as ClearmlDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import DataConfig
from src.dataset import Dataset
from src.transforms import Transforms


class DataModule(LightningDataModule):  # noqa: WPS230
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.transforms = Transforms(*cfg.img_size)

        # Prevent hyperparameters from being stored in checkpoints.
        self.save_hyperparameters(logger=False)

        self.data_path: Optional[Path] = None

        self.all_data: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def class_to_idx(self) -> Dict[str, int]:
        if not (self.data_path and self.all_data):
            self.prepare_data()
            self._load_data('fit')
        return self.all_data.class_to_idx

    def prepare_data(self) -> None:
        self.data_path = Path(ClearmlDataset.get(dataset_name=self.cfg.dataset_name).get_local_copy())

    def setup(self, stage: str):
        if stage == 'fit':
            if not self.all_data:
                self._load_data(stage)
            self.data_train, self.data_val, self.data_test = torch.utils.data.random_split(
                self.all_data,
                self.cfg.data_split,
            )

            self.data_val.transform = self.transforms.compose('test')
            self.data_test.transform = self.transforms.compose('test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            shuffle=False,
        )

    def _load_data(self, stage) -> None:
        self.all_data = Dataset(
            str(self.data_path / 'Stanford_Online_Products'),
            transform=self.transforms.compose(stage),
        )
