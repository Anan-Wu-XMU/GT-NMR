"""
https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0374-3#Sec2
Node and Edge features for datasets NMRDataset

Node labels:                       Description:                                                                Number:
Atomic number                                                                                                  1
Atomic number                      One-hot encoded {H,C,O,N,P,S,F,Cl}                                          8
Valence                                                                                                        1
Valence                            One-hot encoded 1-6                                                         6
Aromaticity                        Does RDKit identify this atom as being part of an aromatic structure        1
Hybridization                      One-hot encoded {S,SP,SP2,SP3,SP3D,SP3D2}                                   6
Formal charge                      Does this atom have a net charge, one-hot encoded {− 1, 0, + 1}             3
Default valence                    One-hot encoded 1-6                                                         6
Rings                              Is this atom part of a ring, one-hot encoded 3-7                            5
total                                                                                                          37

valence: atom.GetTotalValence(), range(1, 7)
default_valence: pt.GetDefaultValence(atomic_num), range(1, 7)
rings: [atom.IsInRingSize(r) for r in range(3, 8)]


Edge labels:
For a given molecule,This paper create 4 adjacency matrices G1, G1.5, G2, G3
where entry gi,j indicates a bond of the relevant order between vertices vi and vj.

'None or Others': 0 0 0 0
'the bond order is 1': 1 0 0 0
'the bond order is 1.5': 0 1 0 0
'the bond order is 2': 0 0 1 0
'the bond order is 3': 0 0 0 1

"""

import numpy as np
from rdkit import Chem


def get_atom_features(atom=None):
    # Atomic number
    if atom is None:
        print('No atom found')
    atomic_num = atom.GetAtomicNum()

    # Atomic number one-hot encoding
    atom_dict = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7}
    atomic_num_onehot = [int(i == atomic_num) for i in atom_dict]

    # Valence
    valence = atom.GetTotalValence()

    # Valence one-hot encoding
    valence_onehot = [int(i == valence) for i in range(1, 7)]

    # Aromaticity
    aromatic_onehot = int(atom.GetIsAromatic())

    # Hybridization
    possible_hybridization_list = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
    hybridization = str(atom.GetHybridization())
    onehot_hybridization = [int(h == hybridization) for h in possible_hybridization_list]

    # Formal charge
    formal_charge = atom.GetFormalCharge()
    formal_charge_onehot = [int(i == formal_charge) for i in [-1, 0, 1]]

    # Default valence
    default_valence = Chem.GetPeriodicTable().GetDefaultValence(atomic_num)
    default_valence_onehot = [int(i == default_valence) for i in range(1, 7)]

    # Rings
    rings = [int(atom.IsInRingSize(r)) for r in range(3, 8)]

    # Total
    total = [atomic_num] + atomic_num_onehot + [valence] + valence_onehot + [aromatic_onehot] + onehot_hybridization + \
            formal_charge_onehot + default_valence_onehot + rings

    return total


def get_bond_features(bond=None):
    if bond is None:
        print("No bond found")
    # Bond type

    bond_type = bond.GetBondTypeAsDouble()

    # Bond type one-hot encoding
    bond_type_onehot = [int(i == bond_type) for i in [1, 1.5, 2, 3]]

    # Total
    total = bond_type_onehot

    return total



#smiles = 'CC1=CC=C(C=C1)C(=O)O'
#mol = Chem.MolFromSmiles(smiles)
#for atom in mol.GetAtoms():
#    print(get_atom_features(atom))
#for bond in mol.GetBonds():
#    print(get_bond_features(bond))

def mol2graph(mol):
    atom_encoder, bond_encoder = get_atom_features, get_bond_features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_encoder(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_encoder(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 4), dtype=np.int64)

    graph = dict()
    graph['num_nodes'] = len(x)
    graph['node_feat'] = x
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr

    return graph


















