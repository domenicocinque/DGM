from argparse import Namespace

import pytorch_lightning as pl
import torch
from torch import nn
import torch_geometric
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv

from DGMlib.layers import MLP, Identity


def conv_resolver(name):
    if name == 'gcn':
        return GCNConv
    elif name == 'gat':
        return GATConv
    else:
        raise ValueError(f"{name} not supported.")


def fix_self_loops(edge_index):
    edges, _ = torch_geometric.utils.remove_self_loops(edge_index)
    edges, _ = torch_geometric.utils.add_self_loops(edge_index)
    return edges


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

    def forward(self, x, edge_index, *args):
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

        rows = torch.repeat_interleave(torch.arange(n), self.k)
        edges = torch.stack([indices.flatten(), rows])

        return edges, logprobs


class DGM(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        k = hparams.k

        self.graph_f = ModuleList()
        self.node_g = ModuleList()
        for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
            self.graph_f.append(DGMBlock(embedding_channels=dgm_l, embedding_f=hparams.ffun, k=k))
            self.node_g.append(conv_resolver(hparams.gfun)(-1, conv_l))

        self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.classification = MLP(fc_layers, final_activation=False)
        self.avg_accuracy = None
        self.cross_entropy = nn.CrossEntropyLoss()

        # torch lightning specific
        self.automatic_optimization = False

    def forward(self, x, edge_index=None):
        x = self.pre_fc(x)

        graph_x = x.detach()
        lprobslist = []
        for f, g in zip(self.graph_f, self.node_g):
            graph_x, edge_index, lprobs = f(graph_x, edge_index, None)

            x = g(torch.dropout(x, self.hparams.dropout, train=self.training), edge_index).relu()
            graph_x = torch.cat([graph_x, x.detach()], -1)
            if lprobs is not None:
                lprobslist.append(lprobs)

        return self.classification(x), torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()

        X, y, mask, edges = data.x, data.y, data.train_mask, data.edge_index
        pred, logprobs = self(X, edges)

        train_pred = pred[mask, :]
        train_lab = y[mask]

        loss = self.cross_entropy(train_pred, train_lab)
        loss.backward()

        # GRAPH LOSS
        corr_pred = (train_pred.argmax(-1) == train_lab).float().detach()

        if self.avg_accuracy is None:
            self.avg_accuracy = torch.ones_like(corr_pred) * 0.5

        point_w = self.avg_accuracy - corr_pred
        graph_loss = point_w * (logprobs[mask, :].exp()).mean([-1, -2])
        graph_loss = graph_loss.mean()
        graph_loss.backward()

        self.avg_accuracy = self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred

        optimizer.step()

        self.log('train/acc', corr_pred.mean(), on_step=True, prog_bar=True)
        self.log('train/class_loss', loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/graph_loss', graph_loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=True)

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

        self.log('val/class_loss', loss.detach(), prog_bar=False)
        self.log('val/acc', correct_t, prog_bar=True)

    def test_step(self, data, batch_idx):
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

        self.log('test/loss', loss.detach().cpu())
        self.log('test/acc', correct_t)

