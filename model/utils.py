import torch_geometric
from torch import nn
from torch_geometric.nn import GCNConv, GATConv


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


class Identity(nn.Module):
    def __init__(self, retparam=None):
        self.retparam = retparam
        super(Identity, self).__init__()

    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params