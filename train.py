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
        #T.NormalizeFeatures(),
        T.ToDevice(device=device),
    ])

    path = osp.join('data', 'Tadpole')
    train_data = TadpoleDataset(root=path, fold=run_params.fold, transform=transform)

    print(train_data[0])
    loader = DataLoader(train_data, batch_size=1)

    class MyDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return loader

        def val_dataloader(self):
            return loader

        def test_dataloader(self):
            return loader

    # configure input feature size
    if run_params.pre_fc is None or len(run_params.pre_fc) == 0:
        if len(run_params.dgm_layers[0]) > 0:
            run_params.dgm_layers[0][0] = train_data.n_features
        run_params.conv_layers[0][0] = train_data.n_features
    else:
        run_params.pre_fc[0] = train_data.num_features
    run_params.fc_layers[-1] = train_data.num_classes

    model = DGM(run_params)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min')
    #callbacks = [checkpoint_callback, early_stop_callback]


    logger = TensorBoardLogger("logs/")
    trainer = pl.Trainer.from_argparse_args(run_params, logger=logger)#,
                                          #  callbacks=callbacks)

    trainer.fit(model, datamodule=MyDataModule())
    trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    params = parser.parse_args([  # '--gpus','1',
        #'--log_every_n_steps', '1',
        '--max_epochs', '500',
        # '--progress_bar_refresh_rate','10',
        '--check_val_every_n_epoch', '5'])
    parser.add_argument("--num_gpus", default=0, type=int)

    parser.add_argument("--dataset", default='Cora')
    parser.add_argument("--fold", default='0', type=int)  # Used for k-fold cross validation in tadpole/ukbb

    parser.add_argument("--conv_layers", default=[[32, 32], [32, 16], [16, 8]], type=lambda x: eval(x))
    parser.add_argument("--dgm_layers", default=[[32, 16, 4], [], []], type=lambda x: eval(x))
    parser.add_argument("--fc_layers", default=[8, 8, 3], type=lambda x: eval(x))
    parser.add_argument("--pre_fc", default=[-1, 32], type=lambda x: eval(x))

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
