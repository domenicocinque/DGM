from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch.nn import ModuleList
from torch_geometric.nn import MLP, DenseGCNConv
from torchmetrics.functional.classification import multiclass_accuracy

from model.utils import *


class CDGMBlock(nn.Module):
    def __init__(
            self,
            embedding_channels: int,
            embedding_f: str = 'gcn',
            k: int = 4
    ):
        super().__init__()
        self.embedding_f = conv_resolver(embedding_f)(-1, embedding_channels)
        self.temperature = nn.Parameter(torch.tensor(4.))
        self.offset = nn.Parameter(torch.tensor(0.))
        self.k = k

    def forward(self, x, edge_index):
        x = self.embedding_f(x, edge_index).relu()
        adj = self.compute_dense_adjacency(x)
        return x, adj

    def compute_dense_adjacency(self, x):
        dist = torch.cdist(x, x)**2
        exponent = -self.temperature*(dist - self.offset)
        return torch.reciprocal(1+torch.exp(exponent))


class SmallCDGM(pl.LightningModule):
    def __init__(self, num_classes, lr, p_dropout):
        super().__init__()
        self.save_hyperparameters()

        self.dropout = torch.nn.Dropout(p_dropout)
        self.pre_embed = GCNConv(-1, 64)
        self.dgm1 = CDGMBlock(embedding_channels=64, embedding_f='gat', k=4)
        self.conv1 = DenseGCNConv(-1, 128)
        self.conv2 = DenseGCNConv(-1, 64)
        self.classification = MLP([64, num_classes], dropout=p_dropout, plain_last=True)

        self.avg_accuracy = None
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        x = self.pre_embed(x, edge_index)
        graph_x = x.detach()

        graph_x, adj = self.dgm1(graph_x, edge_index)
        x = self.conv1(x, adj).relu()

        x = self.dropout(self.conv2(x, adj).relu())
        return self.classification(x).squeeze(0)

    def common_step(self, data, stage='train'):
        if stage == 'train':
            mask = data.train_mask
        elif stage == 'val':
            mask = data.val_mask
        elif stage == 'test':
            mask = data.test_mask

        pred = self(data.x, data.edge_index)
        loss = self.cross_entropy(pred[mask, :], data.y[mask])
        acc = multiclass_accuracy(pred[mask, :], data.y[mask], num_classes=self.hparams.num_classes)
        return loss, acc

    def training_step(self, data, batch_idx):
        loss, acc = self.common_step(data, 'train')
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return {'loss': loss}

    def validation_step(self, data, batch_idx):
        loss, acc = self.common_step(data, 'val')
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return {'loss': loss}

    def test_step(self, data, batch_idx):
        loss, acc = self.common_step(data, 'test')
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer