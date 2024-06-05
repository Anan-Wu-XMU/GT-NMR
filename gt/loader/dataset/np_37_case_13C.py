from gt.features.gcn2019.mol2graph import mol2graph
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


import os
import os.path as osp
import pickle
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
import numpy as np


class np37case13C(InMemoryDataset):
    def __init__(self,
                 root,
                 mol2graph=mol2graph,
                 transform=None,
                 pre_transform=None,
                 ):
        self.root = root
        self.mol2graph = mol2graph
        super(np37case13C, self).__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return ['example_raw_data1.txt', 'example_raw_data2.txt']
        return 'np-37.pickle'

    @property
    def processed_file_names(self):
        return 'np_37_case_13C_data_processed.pt'

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root,'processed')

    def process(self):
        nmr_data = pickle.load(open(self.raw_paths[0], 'rb'))

        # mol to graph
        mol_list = nmr_data['mol']
        mol_idx = nmr_data['molidx']
        mol_value = nmr_data['value']

        print('Converting molecules to graphs...')
        data_list = []

        for i, (mol, label, idx) in enumerate(tqdm(zip(mol_list, mol_value, mol_idx), total=len(mol_list))):
            data = Data()

            total_num_nodes = mol.GetNumAtoms()

            label_tensor = torch.zeros(total_num_nodes, 2)
            #for data_dict in label:
            #    for key, value in data_dict.items():
            #        label_tensor[key][1] = value

            for atom in range(mol.GetNumAtoms()):
                label_tensor[atom][0] = mol.GetAtomWithIdx(atom).GetAtomicNum()

            mol_withoutHs = Chem.RemoveHs(mol)
            mask_hs = mask_H(label_tensor)
            label_tensor = label_tensor[mask_hs]

            graph = mol2graph(mol_withoutHs)

            infer_mask = infer_mask_fun(label_tensor)
            x = torch.from_numpy(graph['node_feat']).to(torch.long)
            edge_index = torch.from_numpy(graph['edge_index']).to(torch.long)
            edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.long)

            data.__num_nodes__ = int(graph['num_nodes'])
            data.x = x
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data.y = None
            data.mask_node = None
            data.infer_mask = infer_mask
            data.data_id = idx

            data_list.append(data)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])


def mask_H(x: torch.Tensor):
    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for i in range(x.shape[0]):
        if x[i][0] == 1.0000:
            mask[i] = False  # H is False
        else:
            mask[i] = True
    return mask


def infer_mask_fun(x: torch.Tensor):
    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for i in range(x.shape[0]):
        if x[i][0] == 6.0000:
            mask[i] = True  # C is True
        else:
            mask[i] = True
    return mask

