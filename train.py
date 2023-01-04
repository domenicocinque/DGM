import os.path as osp
import warnings

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from datasets import TadpoleDataset
import pytorch_lightning as pl
from lightning_lite.utilities.warnings import PossibleUserWarning
from model.dDGM import SmalldDGM
from model.cDGM import SmallCDGM
import torch_geometric.transforms as T
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def run_training_process(params):
    warnings.filterwarnings("ignore", category=PossibleUserWarning)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = T.Compose([
        T.ToDevice(device=device),
    ])

    if params.dataset == 'Tadpole':
        path = osp.join('data', 'Tadpole')
        data = TadpoleDataset(root=path, fold=params.fold, transform=transform)
    elif params.dataset == 'Cora':
        data = Planetoid(root='data', name='Cora', transform=transform)

    dataloader = DataLoader(data, batch_size=1)
    params.pre_fc[0] = data.num_features
    params.final_fc[-1] = data.num_classes
    model = SmallCDGM(data.num_classes, lr=1e-2, p_dropout=0.3)

    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer.from_argparse_args(
        params,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger
    )
    trainer.fit(model, dataloader, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args([
        '--max_epochs', '500',
        '--check_val_every_n_epoch', '5',
        '--log_every_n_steps', '1'
    ])
    parser.add_argument("--dataset", default='Tadpole')
    parser.add_argument("--fold", default='0', type=int)
    parser.add_argument("--conv_layers", default=[32, 128, 32], type=lambda x: eval(x))
    parser.add_argument("--dgm_layers", default=[32, None, None], type=lambda x: eval(x))
    parser.add_argument("--final_fc", default=[32, 8, -1], type=lambda x: eval(x))
    parser.add_argument("--pre_fc", default=[-1, 32], type=lambda x: eval(x))
    parser.add_argument("--gfun", default='gcn')
    parser.add_argument("--ffun", default='gcn')
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--test_eval", default=10, type=int)

    parser.set_defaults(default_root_path='./log')
    params = parser.parse_args(namespace=params)

    run_training_process(params)
