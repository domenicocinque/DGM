import os
import pandas as pd
from typing import Optional, Callable

import torch
import pickle
import numpy as np
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


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


# class TadpoleDataset(torch.utils.data.Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, fold=0, split='train', samples_per_epoch=100, device='cpu'):

#         with open('data/train_data.pickle', 'rb') as f:
#             X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data

#         X_ = X_[...,:30,:] # For DGM we use modality 1 (M1) for both node representation and graph learning.

#         self.X = torch.from_numpy(X_[:,:,fold]).float().to(device)
#         self.y = torch.from_numpy(y_[:,:,fold]).float().to(device)
#         self.weight = torch.from_numpy(np.squeeze(weight_[:1,fold])).float().to(device)

#         # split train set in train/val
#         train_mask = train_mask_[:,fold]
#         nval = int(train_mask.sum()*0.2)
#         val_idxs = np.random.RandomState(1).choice(np.nonzero(train_mask.flatten())[0],(nval,),replace=False)
#         train_mask[val_idxs] = 0;
#         val_mask = train_mask*0
#         val_mask[val_idxs] = 1

#         print('DATA STATS: train: %d val: %d' % (train_mask.sum(),val_mask.sum()))

#         if split=='train':
#             self.mask = torch.from_numpy(train_mask).to(device)
#         if split=='val':
#             self.mask = torch.from_numpy(val_mask).to(device)
#         if split=='test':
#             self.mask = torch.from_numpy(test_mask_[:,fold]).to(device)

#         self.samples_per_epoch = samples_per_epoch

#     def __len__(self):
#         return self.samples_per_epoch

#     def __getitem__(self, idx):
#         return self.X,self.y,self.mask


def get_planetoid_dataset(name, normalize_features=True, transform=None, split="complete"):
    path = osp.join('.', 'data', name)
    if split == 'complete':
        dataset = Planetoid(path, name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(path, name, split=split)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]


class PlanetoidDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', samples_per_epoch=100, name='Cora', device='cpu'):
        dataset = get_planetoid_dataset(name)
        self.X = dataset[0].x.float().to(device)
        self.y = one_hot_embedding(dataset[0].y, dataset.num_classes).float().to(device)
        self.edge_index = dataset[0].edge_index.to(device)
        self.num_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes

        if split == 'train':
            self.mask = dataset[0].train_mask.to(device)
        if split == 'val':
            self.mask = dataset[0].val_mask.to(device)
        if split == 'test':
            self.mask = dataset[0].test_mask.to(device)

        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X, self.y, self.mask, self.edge_index
