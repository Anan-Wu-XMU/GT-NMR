import logging
import os.path as osp
import time
import warnings
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import (GNNBenchmarkDataset, Planetoid, TUDataset,
                                      WikipediaNetwork, ZINC)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader
from gt.loader.split_generator import (prepare_splits,
                                         set_dataset_splits)
from gt.transform.posenc_stats import compute_posenc_stats, ComputePosencStat
from gt.transform.transforms import (pre_transform_in_memory,
                                       typecast_x, concat_x_and_pos,
                                       clip_graphs_to_size)

from gt.loader.dataset.gt_13C import gt13C
from gt.loader.dataset.gt_1H import gt1H
from gt.loader.dataset.np_37_case_13C import np37case13C
from gt.loader.dataset.custom_inference import CustomInferenceDataset



def log_loaded_dataset(dataset, format, name):
    if cfg.dataset.name == 'CustomInference':
        logging.info(f"[*] Loaded dataset '{name}' from '{format}'")
        return
    else:
        logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
        logging.info(f"  {dataset.data}")
        logging.info(f"  undirected: {dataset[0].is_undirected()}")
        logging.info(f"  num graphs: {len(dataset)}")

        total_num_nodes = 0
        if hasattr(dataset.data, 'num_nodes'):
            total_num_nodes = dataset.data.num_nodes
        elif hasattr(dataset.data, 'x'):
            total_num_nodes = dataset.data.x.size(0)
        logging.info(f"  avg num_nodes/graph: "
                     f"{total_num_nodes // len(dataset)}")
        logging.info(f"  num node features: {dataset.num_node_features}")
        logging.info(f"  num edge features: {dataset.num_edge_features}")
        if hasattr(dataset, 'num_tasks'):
            logging.info(f"  num tasks: {dataset.num_tasks}")

        if hasattr(dataset.data, 'y') and dataset.data.y is not None:
            if isinstance(dataset.data.y, list):
                # A special case for ogbg-code2 dataset.
                logging.info(f"  num classes: n/a")
            elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                    torch.is_floating_point(dataset.data.y):
                logging.info(f"  num classes: (appears to be a regression task)")
            else:
                logging.info(f"  num classes: {dataset.num_classes}")
        elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
            # Edge/link prediction task.
            if hasattr(dataset.data, 'train_edge_label'):
                labels = dataset.data.train_edge_label  # Transductive link task
            else:
                labels = dataset.data.edge_label  # Inductive link task
            if labels.numel() == labels.size(0) and \
                    torch.is_floating_point(labels):
                logging.info(f"  num edge classes: (probably a regression task)")
            else:
                logging.info(f"  num edge classes: {len(torch.unique(labels))}")

        ## Show distribution of graph sizes.
        graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
                        for d in dataset]
        hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
        logging.info(f'   Graph size distribution:')
        logging.info(f'     mean: {np.mean(graph_sizes)}')
        for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
             logging.info(
                 f'     bin {i}: [{start:.2f}, {end:.2f}]: '
                 f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
             )

@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'gt13C':
            dataset = preformat_gt13C(dataset_dir, name)

        elif pyg_dataset_id == 'gt1H':
            dataset = preformat_gt1H(dataset_dir, name)

        elif pyg_dataset_id == 'np_37_case_13C':
            dataset = preformat_np37case13C(dataset_dir, name)

        elif pyg_dataset_id == 'CustomInference':
            dataset = preformat_CustomInference(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")
    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")

    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        if not cfg.dataset.pe_transform_on_the_fly:
            pre_transform_in_memory(dataset,
                                    partial(compute_posenc_stats,
                                            pe_types=pe_enabled_list,
                                            is_undirected=is_undirected,
                                            cfg=cfg,
                                            ),
                                    show_progress=True,
                                    cfg=cfg,
                                    posenc_mode=True
                                    )

            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")
        else:
            warnings.warn('PE transform on the fly to save memory consumption; experimental, please only use for RWSE/RWPSE')
            pe_transform = ComputePosencStat(pe_types=pe_enabled_list,
                                             is_undirected=is_undirected,
                                             cfg=cfg
                                             )
            if dataset.transform is None:
                dataset.transform = pe_transform
            else:
                dataset.transform = T.compose([pe_transform, dataset.transform])


    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])
        # print(f"Indegrees: {cfg.gt.pna_degrees}")
        # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    return dataset


def add_pe_transform_to_dataset(format, name, dataset_dir, pe_transform=None):
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'gt13C':
            dataset = preformat_gt13C(dataset_dir, name)

        elif pyg_dataset_id == 'gt1H':
            dataset = preformat_gt1H(dataset_dir, name)

        elif pyg_dataset_id == 'np_37_case_13C':
            dataset = preformat_np37case13C(dataset_dir, name)

        elif pyg_dataset_id == 'CustomInference':
            dataset = preformat_CustomInference(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")

    dataset_dir = dataset_dir.replace("/GNNBenchmarkDataset", "")
    dataset = join_dataset_splits(
        [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
         for split in ['train', 'val', 'test']]
    )
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset


def preformat_gt13C(dataset_dir,name):
    """Load and preformat NMR dataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific NMR dataset

    Returns:
        PyG dataset object
    """

    dataset = gt13C(dataset_dir)
    dataset.name = 'gt13C'

    return dataset


def preformat_gt1H(dataset_dir,name):

    dataset = gt1H(dataset_dir)
    dataset.name = 'gt1H'

    return dataset


def preformat_np37case13C(dataset_dir,name):

    dataset = np37case13C(dataset_dir)
    dataset.name = 'np37case13C'

    return dataset


def preformat_CustomInference(dataset_dir,name):
    dataset = CustomInferenceDataset(dataset_dir)
    dataset.name = 'CustomInference'

    return dataset




def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
