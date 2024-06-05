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


class gt1H(InMemoryDataset):
    def __init__(self,
                 root,
                 mol2graph=mol2graph,
                 transform=None,
                 pre_transform=None,
                 ):
        self.root = root
        self.mol2graph = mol2graph
        super(gt1H, self).__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return ['example_raw_data1.txt', 'example_raw_data2.txt']
        return 'graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.1.mol_dict.pickle'

    @property
    def processed_file_names(self):
        return 'nmr_1H_data_processed.pt'

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    def process(self):
        nmr_data = pickle.load(open(self.raw_paths[0], 'rb'))

        train_df, test_df = nmr_data['train_df'], nmr_data['test_df']
        train_df['mask'] = True
        test_df['mask'] = False

        train_df.rename(columns={'value': 'label'}, inplace=True)
        test_df.rename(columns={'value': 'label'}, inplace=True)

        train_df = train_df[['rdmol', 'label', 'mask']]
        test_df = test_df[['rdmol', 'label', 'mask']]
        total_df = pd.concat([train_df, test_df], ignore_index=True)
        rdmol = total_df['rdmol']
        value = total_df['label']
        is_train = total_df['mask']

        print('Converting molecules to graphs...')
        data_list = []

        for i, (mol, label) in enumerate(tqdm(zip(rdmol, value), total=len(rdmol))):
            data = Data()

            total_num_nodes = mol.GetNumAtoms()
            label_tensor = torch.zeros(total_num_nodes, 2)
            for data_dict in label:
                for key, value in data_dict.items():
                    label_tensor[key][1] = value

            for atom in range(mol.GetNumAtoms()):
                label_tensor[atom][0] = mol.GetAtomWithIdx(atom).GetAtomicNum()

            label_tensor_after = assign_labels_to_connected_atoms(mol, label_tensor)
            mol_withoutHs = Chem.RemoveHs(mol)

            mask_hs = mask_H(label_tensor_after)
            label_tensor_after = label_tensor_after[mask_hs]

            graph = mol2graph(mol_withoutHs)
            mask_other = mask_others(label_tensor_after)
            infer_mask = infer_mask_fun(label_tensor_after)

            y = label_tensor_after[:, 1]
            x = torch.from_numpy(graph['node_feat']).to(torch.long)
            edge_index = torch.from_numpy(graph['edge_index']).to(torch.long)
            edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.long)

            data.__num_nodes__ = int(graph['num_nodes'])
            data.x = x
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data.y = y

            data.mask_node = mask_other
            data.infer_mask = infer_mask
            data.is_train = is_train[i]
            data_list.append(data)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Saving...')
        torch.save(self.collate(data_list), self.processed_paths[0])


def assign_labels_to_connected_atoms(mol, atom_labels_tensor):
    result_tensor = torch.zeros_like(atom_labels_tensor)
    result_tensor[:, 0] = atom_labels_tensor[:, 0]
    valid_indices = [i for i, (idx, label) in enumerate(atom_labels_tensor) if idx == 1 and label > 0]
    # print(valid_indices)
    labels_sum = {}
    count = {}

    for valid_idx in valid_indices:
        atom_idx = int(valid_idx)
        label = atom_labels_tensor[valid_idx, 1].item()
        # print(label)
        atom = mol.GetAtomWithIdx(atom_idx)
        # mapped_idx = int(get_connected_atom_indices(mol, atom_idx)[0])
        # print(mapped_idx)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()

            if n_idx not in labels_sum:
                labels_sum[n_idx] = 0
                count[n_idx] = 0

            labels_sum[n_idx] += label
            count[n_idx] += 1
    # print(labels_sum)
    # print(count)

    for n_idx in labels_sum:
        if count[n_idx] > 0:
            avg_label = labels_sum[n_idx] / count[n_idx]
            result_tensor[n_idx, 1] = avg_label

    for valid_idx in valid_indices:
        result_tensor[valid_idx, 1] = 0

    return result_tensor

def mask_H(x: torch.Tensor):
    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for i in range(x.shape[0]):
        if x[i][0] == 1.0000:
            mask[i] = False  # H is False
        else:
            mask[i] = True
    return mask

def mask_others(x: torch.Tensor):
    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for i in range(x.shape[0]):
        if x[i][0] == 6.0000 and x[i][1] != 0:
            mask[i] = True  # C with label is True
        else:
            mask[i] = False
    return mask


def infer_mask_fun(x: torch.Tensor):
    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for i in range(x.shape[0]):
        if x[i][0] == 6.0000:
            mask[i] = True  # C is True
        else:
            mask[i] = False
    return mask
