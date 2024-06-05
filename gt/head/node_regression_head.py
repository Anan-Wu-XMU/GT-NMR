import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_scatter import scatter

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('node_regression_head')
class SANnoderegressionHead(nn.Module):
    """
    SAN prediction head for node regression tasks.
    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.deg_scaler = False
        self.fwl = False
        self.name = cfg.dataset.name
        self.mode = cfg.train.mode

        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()
        # note: modified to add () in the end from original code of 'GPS'
        #   potentially due to the change of PyG/GraphGym version

    def _apply_index(self, batch):
        return batch.x, batch.y

    def _apply_mask(self, batch):
        if batch.mask_node is not None:
            x = batch.x
            y = batch.y
            mask = batch.mask_node
            x = x[mask]
            y = y[mask]
            batch.x = x
            batch.y = y
        return batch

    def _apply_infer_mask(self, batch):
        if batch.infer_mask is not None:
            x = batch.x
            y = batch.y
            mask = batch.infer_mask
            x = x[mask]
            batch.x = x
            batch.y = y
        return batch

    def forward(self, batch):
        if self.mode == 'custom':
            batch = self._apply_mask(batch)
        else:
            batch = self._apply_infer_mask(batch)

        graph_emb = batch.x
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.x = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
