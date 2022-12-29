from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch.nn import ModuleList
from torch_geometric.nn import MLP

from model.utils import *


class DGMBlock(nn.Module):
    def __init__(
            self,
            embedding_channels: int,
            embedding_f: str = 'gcn',
            k: int = 4
    ):
        super().__init__()
        self.embedding_f = conv_resolver(embedding_f)(-1, embedding_channels)
        self.temperature = nn.Parameter(torch.tensor(4.))
        self.k = k

    def forward(self, x, edge_index):
        x = self.embedding_f(x, edge_index)
        edge_index_hat, logprobs = self.sample_without_replacement(x)
        edge_index_hat = fix_self_loops(edge_index_hat)
        return x, edge_index_hat, logprobs

    def sample_without_replacement(self, x):
        logprobs = - self.temperature*torch.cdist(x, x)**2
        n = logprobs.shape[1]

        q = torch.rand_like(logprobs) + 1e-8
        lq = (logprobs - torch.log(-torch.log(q)))
        logprobs, indices = torch.topk(lq, self.k)

        rows = torch.repeat_interleave(torch.arange(n), self.k).to(x.device)
        edges = torch.stack([indices.flatten(), rows])

        return edges, logprobs


class DGM(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)

        self.dgm_f = ModuleList()
        self.diffusion_g = ModuleList()
        for i, (dgm_l, conv_l) in enumerate(zip(hparams.dgm_layers, hparams.conv_layers)):
            if dgm_l is not None:
                self.dgm_f.append(DGMBlock(embedding_channels=dgm_l, embedding_f=hparams.ffun, k=hparams.k))
            else:
                self.dgm_f.append(Identity(retparam=0))
            self.diffusion_g.append(conv_resolver(hparams.gfun)(-1, conv_l))

        self.pre_fc = MLP(hparams.pre_fc, plain_last=False)
        self.classification = MLP(hparams.final_fc, dropout=hparams.dropout, plain_last=True)
        self.avg_accuracy = None
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, edge_index=None):
        x = self.pre_fc(x)
        graph_x = x.detach()
        lprobslist = []
        for f, g in zip(self.dgm_f, self.diffusion_g):
            graph_x, edge_index, lprobs = f(graph_x, edge_index)
            x = g(torch.dropout(x, self.hparams.dropout, train=self.training), edge_index).relu()
            graph_x = torch.cat([graph_x, x.detach()], -1)
            if lprobs is not None:
                lprobslist.append(lprobs)
        return self.classification(x), torch.stack(lprobslist,-1) if len(lprobslist)>0 else None

    def training_step(self, data, batch_idx):
        X, y, mask, edges = data.x, data.y, data.train_mask, data.edge_index
        pred, logprobs = self(X, edges)

        train_pred = pred[mask, :]
        train_lab = y[mask]

        class_loss = self.cross_entropy(train_pred, train_lab)

        # GRAPH LOSS
        if logprobs is not None:
            corr_pred = (train_pred.argmax(-1) == train_lab).float().detach()
            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred) * 0.5

            point_w = self.avg_accuracy - corr_pred
            graph_loss = point_w * (logprobs[mask, :].exp()).mean([-1, -2])
            graph_loss = graph_loss.mean()
        else:
            graph_loss = torch.tensor(0.).to(class_loss.device)

        loss = class_loss + graph_loss
        self.avg_accuracy = self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred

        self.log('train/acc', corr_pred.mean(), on_step=True, prog_bar=True, batch_size=1)
        self.log('train/class_loss', class_loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/graph_loss', graph_loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return {'loss': loss}

    def validation_step(self, data, batch_idx):
        X, y, mask, edge_index = data.x, data.y, data.val_mask, data.edge_index

        pred, logprobs = self(X, edge_index)
        pred = pred.softmax(-1)
        for i in range(1, self.hparams.test_eval):
            pred_, logprobs = self(X, edge_index)
            pred += pred_.softmax(-1)

        test_pred = pred[mask, :]
        test_lab = y[mask]

        correct_t = (test_pred.argmax(-1) == test_lab).float().mean().item()
        loss = self.cross_entropy(test_pred, test_lab)

        self.log('val/class_loss', loss.detach(), prog_bar=False, batch_size=1)
        self.log('val/acc', correct_t, prog_bar=True, batch_size=1)

    def test_step(self, data, batch_idx):
        X, y, mask, edge_index = data.x, data.y, data.test_mask, data.edge_index

        pred, logprobs = self(X, edge_index)
        pred = pred.softmax(-1)
        for i in range(1, self.hparams.test_eval):
            pred_, logprobs = self(X, edge_index)
            pred += pred_.softmax(-1)

        test_pred = pred[mask, :]
        test_lab = y[mask]

        correct_t = (test_pred.argmax(-1) == test_lab).float().mean().item()
        loss = self.cross_entropy(test_pred, test_lab)

        self.log('test/class_loss', loss.detach().cpu(), batch_size=1)
        self.log('test/acc', correct_t, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
