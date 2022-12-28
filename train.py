import os.path as osp
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from datasets import PlanetoidDataset, TadpoleDataset
import pytorch_lightning as pl
from DGMlib.model_dDGM import DGM
import torch_geometric.transforms as T
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def run_training_process(run_params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = T.Compose([
        T.ToDevice(device=device),
    ])

    path = osp.join('data', 'Tadpole')
    data = TadpoleDataset(root=path, fold=run_params.fold, transform=transform)

    dataloader = DataLoader(data, batch_size=1)
    model = DGM(run_params)

    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer.from_argparse_args(run_params, logger=logger)
    trainer.fit(model, dataloader, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args([
        '--max_epochs', '500',
        '--check_val_every_n_epoch', '5'])

    parser.add_argument("--num_gpus", default=0, type=int)

    parser.add_argument("--dataset", default='Cora')
    parser.add_argument("--fold", default='0', type=int)  # Used for k-fold cross validation in tadpole/ukbb

    parser.add_argument("--conv_layers", default=[32, 16, 8], type=lambda x: eval(x))
    parser.add_argument("--dgm_layers", default=[32, 16, 4], type=lambda x: eval(x))
    parser.add_argument("--fc_layers", default=[8, 8, 3], type=lambda x: eval(x))
    parser.add_argument("--pre_fc", default=[60, 32], type=lambda x: eval(x))

    parser.add_argument("--gfun", default='gcn')
    parser.add_argument("--ffun", default='gcn')
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--pooling", default='add')
    parser.add_argument("--distance", default='euclidean')

    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--test_eval", default=10, type=int)

    parser.set_defaults(default_root_path='./log')
    params = parser.parse_args(namespace=params)

    run_training_process(params)
