from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import MLP, DenseGCNConv
from torchmetrics.functional.classification import multiclass_accuracy

from model.utils import *


class CDGMBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            embedding_channels: int,
            embedding_f: str = 'gcn',
            k: int = 4
    ):
        super().__init__()
        self.embedding_f = conv_resolver(embedding_f)(in_channels, embedding_channels)
        self.temperature = nn.Parameter(torch.tensor(4.))
        self.offset = nn.Parameter(torch.tensor(0.5))
        self.k = k
        
        self.scale = nn.Parameter(torch.tensor(-1).float(),requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1,1, in_channels)).float(),requires_grad=False)

    def forward(self, x, edge_index):
        x = self.embedding_f(x, edge_index)
        self.centroid.data = x.mean(-2,keepdim=True).detach()
        self.scale.data = (0.9/(x-self.centroid).abs().max()).detach()
        adj = self.compute_dense_adjacency((x-self.centroid)*self.scale)
        return x, adj

    def compute_dense_adjacency(self, x):
        dist = torch.cdist(x, x)**2
        exponent = self.temperature*(self.offset.abs() - dist)
        return torch.sigmoid(exponent)


class SmallCDGM(pl.LightningModule):
    def __init__(self, num_classes, lr, p_dropout):
        super().__init__()
        self.save_hyperparameters()

        self.dropout = torch.nn.Dropout(p_dropout)
        self.pre_embed = GCNConv(-1, 64)
        self.dgm1 = CDGMBlock(in_channels=64, embedding_channels=64, embedding_f='gcn', k=4)
        self.conv1 = DenseGCNConv(-1, 64)
        self.conv2 = DenseGCNConv(-1, 64)
        self.classification = MLP([64, num_classes], dropout=p_dropout, plain_last=True)

        self.avg_accuracy = None
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.pre_embed(x, edge_index), negative_slope=0.2)
        graph_x = x.detach()

        graph_x, adj = self.dgm1(x, edge_index)
        adj = fix_self_loops(adj, type='dense')
        #tensorboard = self.logger.experiment
        #tensorboard.add_image(img_tensor=adj, tag='adj', dataformats='HW', global_step=self.global_step)       
        x = F.leaky_relu(self.conv1(x, adj), negative_slope=0.2)
        #x = F.leaky_relu(self.conv2(x, adj), negative_slope=0.2)
        x = self.dropout(x)
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
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
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