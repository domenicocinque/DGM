import os.path as osp
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url


def read_tadpole_data(path, fold, val_split: float = 0.2):
    data = pd.read_pickle(path)
    train_mask = data[2].astype('bool')[:, fold]
    test_mask = data[3].astype('bool')[:, fold]

    val_mask = np.zeros_like(train_mask)
    val_mask[train_mask] = np.random.rand(train_mask.sum()) < val_split
    train_mask[val_mask] = False

    X = data[0][:, :, fold]
    y = data[1].argmax(1)[:, fold]
    return {'x': X, 'y': y, 'train_mask': train_mask,
            'val_mask': val_mask, 'test_mask': test_mask}


class TadpoleDataset(InMemoryDataset):
    url = 'https://github.com/lcosmo/DGM_pytorch/blob/main/data/tadpole_data.pickle'

    def __init__(
            self,
            root: str,
            fold: int = 0,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None
    ):
        self.fold = fold
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'tadpole_data.pickle'

    def download(self):
        path = download_url(self.url, self.raw_dir)

    @property
    def processed_file_names(self):
        return f"tadpole_fold_{self.fold}.pt"

    def process(self):
        path = osp.join(self.raw_dir, 'tadpole_data.pickle')
        print(path)
        data = read_tadpole_data(path, self.fold)

        edge_index = torch.tensor([[], []], dtype=torch.long)
        data = Data(
            x=torch.from_numpy(data['x']).float(),
            y=torch.from_numpy(data['y']).long(),
            edge_index=edge_index,
            train_mask=torch.from_numpy(data['train_mask']),
            val_mask=torch.from_numpy(data['val_mask']),
            test_mask=torch.from_numpy(data['test_mask'])
        )

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_classes(self):
        return 3

    @property
    def num_features(self):
        return 60

