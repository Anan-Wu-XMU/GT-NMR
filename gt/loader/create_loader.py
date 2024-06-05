import os.path as osp
from typing import Callable

import torch

import torch_geometric.graphgym.register as register

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.transform import (
    create_link_label,
    neg_sampling_transform,
)
from torch_geometric.loader import (
    ClusterLoader,
    DataLoader,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    NeighborSampler,
    RandomNodeLoader,
)
from torch_geometric.utils import (
    index_to_mask,
    negative_sampling,
    to_undirected,
)

index2mask = index_to_mask  # TODO Backward compatibility


def load_dataset():
    r"""

    Load dataset objects.

    Returns: PyG dataset object

    """
    format = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir
    # Try to load customized data format
    for func in register.loader_dict.values():
        dataset = func(format, name, dataset_dir)

        if dataset is not None:
            return dataset


def set_dataset_info(dataset):
    r"""
    Set global dataset information

    Args:
        dataset: PyG dataset object

    """

    # get dim_in and dim_out
    try:
        cfg.share.dim_in = dataset._data.x.shape[1]
    except Exception:
        cfg.share.dim_in = 1
    try:
        if cfg.dataset.task_type == 'classification':
            cfg.share.dim_out = torch.unique(dataset._data.y).shape[0]
        else:
            cfg.share.dim_out = dataset._data.y.shape[1]
    except Exception:
        cfg.share.dim_out = 1

    # count number of dataset splits
    cfg.share.num_splits = 1
    for key in dataset._data.keys():
        if 'val' in key:
            cfg.share.num_splits += 1
            break
    for key in dataset._data.keys():
        if 'test' in key:
            cfg.share.num_splits += 1
            break


def create_dataset():
    r"""
    Create dataset object

    Returns: PyG dataset object

    """
    dataset = load_dataset()
    set_dataset_info(dataset)

    return dataset


def get_loader(dataset, sampler, batch_size, shuffle=True):
    pw = cfg.num_workers > 0
    if sampler == "full_batch" or len(dataset) > 1:
        loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=cfg.num_workers,
                                  pin_memory=True, persistent_workers=pw)
    elif sampler == "neighbor":
        loader_train = NeighborSampler(
            dataset[0], sizes=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=True)
    elif sampler == "random_node":
        loader_train = RandomNodeLoader(dataset[0],
                                        num_parts=cfg.train.train_parts,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True, persistent_workers=pw)
    elif sampler == "saint_rw":
        loader_train = \
            GraphSAINTRandomWalkSampler(dataset[0],
                                        batch_size=batch_size,
                                        walk_length=cfg.train.walk_length,
                                        num_steps=cfg.train.iter_per_epoch,
                                        sample_coverage=0,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True,
                                        persistent_workers=pw)
    elif sampler == "saint_node":
        loader_train = \
            GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=pw)
    elif sampler == "saint_edge":
        loader_train = \
            GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=pw)
    elif sampler == "cluster":
        loader_train = ClusterLoader(
            dataset[0],
            num_parts=cfg.train.train_parts,
            save_dir=osp.join(
                cfg.dataset.dir,
                cfg.dataset.name.replace("-", "_"),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=pw,
        )

    else:
        raise NotImplementedError(f"'{sampler}' is not implemented")

    return loader_train


def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph' or (cfg.dataset.task == 'node' and cfg.dataset.task_type == 'regression'):
        if cfg.train.mode != 'GTNMR-inference' and cfg.train.mode != 'GTNMR-custom-inference':
            id = dataset.data['train_graph_index']
            import numpy as np
            id = np.array(id)
            id = id.astype(np.int64)
            loaders = [
                get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                           shuffle=True)
            ]
            delattr(dataset.data, 'train_graph_index')
        else:
            id = dataset.data['infer_graph_index']
            import numpy as np
            id = np.array(id)
            id = id.astype(np.int64)
            loaders = [
                get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                           shuffle=True)
            ]

    else:
        loaders = [
            get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True)
        ]

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph' or (cfg.dataset.task == 'node' and cfg.dataset.task_type == 'regression'):
            if cfg.train.mode != 'GTNMR-inference' and cfg.train.mode != 'GTNMR-custom-inference':
                split_names = ['val_graph_index', 'test_graph_index']
                id = dataset.data[split_names[i]]
                id = np.array(id)
                id = id.astype(np.int64)
                loaders.append(
                    get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                               shuffle=False))
                delattr(dataset.data, split_names[i])

        else:
            loaders.append(
                get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))

    return loaders