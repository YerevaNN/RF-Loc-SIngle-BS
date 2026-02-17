import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.datamodules.datasets import WAIRDDatasetPathLoss


class WAIRDPathLossDatamodule(pl.LightningDataModule):
    
    def __init__(
        self, batch_size: int, num_workers: int, drop_last: bool,
        epoch_counter, multi_gpu: bool = False, *args, **kwargs
    ):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._drop_last = drop_last
        
        self._multi_gpu = multi_gpu
        self._args = args
        self._kwargs = kwargs
    
    @staticmethod
    def collate_fn(
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_resized, base_stations_data, ue_loc_img, orig_image_size, ue_loc_y_x = zip(*batch)
        
        map_resized_batch = torch.stack([torch.tensor(m) for m in map_resized])
        # map_resized_batch = map_resized_batch.unsqueeze(1)
        # map_resized_batch = map_resized_batch.expand(-1, 3, -1, -1)
        ue_loc_img_batch = torch.stack([torch.tensor(u) for u in ue_loc_img])
        # ue_loc_img_batch = ue_loc_img_batch.unsqueeze(1)
        
        max_base_stations = max([b.shape[0] for b in base_stations_data])
        
        base_stations_data_padded = [
            np.pad(
                b, ((0, max_base_stations - b.shape[0]), (0, 0)),
                mode='constant', constant_values=0
            ) for b in base_stations_data
        ]
        base_stations_data_batch = torch.stack([torch.tensor(b) for b in base_stations_data_padded])
        base_station_lengths = torch.tensor([len(b) for b in base_stations_data])
        
        orig_image_size_batch = torch.tensor(orig_image_size)
        ue_loc_y_x = torch.tensor(ue_loc_y_x)
        
        return map_resized_batch, base_stations_data_batch, base_station_lengths, \
               ue_loc_img_batch, orig_image_size_batch, ue_loc_y_x
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetPathLoss(
            split="train",
            *self._args, **self._kwargs,
        )
        self._val_set = WAIRDDatasetPathLoss(
            split="val",
            *self._args, **self._kwargs,
        )
        self._test_set = WAIRDDatasetPathLoss(
            split="test",
            *self._args, **self._kwargs,
        )
    
    def train_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._train_set) if self._multi_gpu else None
        return DataLoader(
            self._train_set, batch_size=self._batch_size, num_workers=self._num_workers,
            sampler=sampler, shuffle=None if self._multi_gpu else True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._val_set) if self._multi_gpu else None
        return DataLoader(
            self._val_set, batch_size=self._batch_size, num_workers=self._num_workers,
            sampler=sampler, shuffle=None if self._multi_gpu else True,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self._test_set) if self._multi_gpu else None
        return DataLoader(
            self._test_set, batch_size=self._batch_size, num_workers=self._num_workers,
            sampler=sampler, shuffle=None if self._multi_gpu else True,
            collate_fn=self.collate_fn,
        )
