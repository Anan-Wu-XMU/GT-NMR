import torch

from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
from torch_geometric.graphgym import cfg


@register_node_encoder('customAtom')
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        num_features = cfg.dataset.customAtom_num_features
        for _ in range(num_features):
            emb = torch.nn.Embedding(150, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.x.shape[1]):
            encoded_features += self.atom_embedding_list[i](batch.x[:, i])

        batch.x = encoded_features
        return batch


@register_edge_encoder('customBond')
class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim: int):
        super().__init__()

        num_edge_features = cfg.dataset.customBond_num_features

        self.bond_embedding_list = torch.nn.ModuleList()

        for _ in range(num_edge_features):
            emb = torch.nn.Embedding(50, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_attr.shape[1]):
            edge_attr = batch.edge_attr
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        batch.edge_attr = bond_embedding
        return batch
