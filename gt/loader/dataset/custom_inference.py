from gt.features.gcn2019.mol2graph import mol2graph
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


import os
import os.path as osp
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
import numpy as np
from torch_geometric.graphgym import cfg
from datetime import datetime
from pathlib import Path

class CustomInferenceDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 mol2graph=mol2graph,
                 transform=None,
                 pre_transform=None,
                 ):
        self.root = root
        self.mol2graph = mol2graph
        super(CustomInferenceDataset, self).__init__(root, transform, pre_transform)

        inference = cfg.dataset.inference

        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'custom_inference_data_processed_{current_time}.pt'

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    def process(self):
        inference = cfg.dataset.inference
        current_dir = Path(__file__).resolve().parent
        base_path = current_dir.parents[2] / 'inference_input_files'
        if inference is None:
            raise ValueError('Please provide SMILES/smiles.csv/mol.mol file for inference')

        if inference.endswith('.csv'):
            file_path = base_path / inference
            df = pd.read_csv(file_path)
            smiles_list = df['smiles'].tolist()
            mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
            name_list = [f"{inference}-smiles{i+1}" for i in range(len(smiles_list))]

        elif inference.endswith('.mol'):
            file_path = base_path / inference
            file_name = inference.split('.')[0]
            mol = Chem.MolFromMolFile(str(file_path))
            mol = Chem.RemoveHs(mol)
            mol_list = [mol]
            name_list = [file_name]
        elif inference =='mols':
            file_path = base_path / 'mols'
            mol_files = os.listdir(file_path)
            mol_list = []
            name_list = []
            for mol_file in mol_files:
                mol = Chem.MolFromMolFile(str(file_path / mol_file))
                mol = Chem.RemoveHs(mol)
                mol_list.append(mol)
                name_list.append(mol_file.split('.')[0])

        elif isinstance(inference, str) and not any(inference.endswith(ext) for ext in ['.csv', '.mol']):
            try:
                mol = Chem.MolFromSmiles(inference)
                mol = Chem.RemoveHs(mol)
                mol_list = [mol]
                name_list = [inference]
            except:
                raise ValueError('Please provide Valid SMILES string')
        else:
            raise ValueError('Please provide valid input file')
        data_list = []

        for i, mol in enumerate(mol_list):
            data = Data()
            try:
                total_num_nodes = mol.GetNumAtoms()
                label_tensor = torch.zeros(total_num_nodes, 2)
                for atom in range(mol.GetNumAtoms()):
                    label_tensor[atom][0] = mol.GetAtomWithIdx(atom).GetAtomicNum()

                infer_mask = infer_mask_fun(label_tensor)

                graph = mol2graph(mol)
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
                data.mol = mol
                data.name = name_list[i]
                data_list.append(data)
            except:
                print(f'Error in processing molecule {i}')

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('Saving...')
        print(f'Processed {len(data_list)} molecules')

def infer_mask_fun(x: torch.Tensor):
    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    for i in range(x.shape[0]):
        if x[i][0] == 6.0000:
            mask[i] = True  # C is True
        else:
            mask[i] = False
    return mask


