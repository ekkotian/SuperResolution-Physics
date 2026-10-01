"""Shared data utilities for the coarse-grid fidelity correction models (LFLR -> HFLR)."""
import random

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

H, W, T = 20, 20, 11


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sw_pair(h5_path: str):
    """
    LFLR = coarse-simulation saturation (sw_grid)
    HFLR = fine-simulation saturation downsampled to the coarse grid (sw_grid_downsampled)
    Both returned as float32 arrays of shape (N, H, W, T).
    """
    with h5py.File(h5_path, "r") as f:
        lflr = f["sw_grid"][:]               # (N, 11, 20, 20, 1)
        hflr = f["sw_grid_downsampled"][:]
    x = np.transpose(lflr, (0, 2, 3, 1, 4)).reshape(-1, H, W, T)
    y = np.transpose(hflr, (0, 2, 3, 1, 4)).reshape(-1, H, W, T)
    return x.astype(np.float32), y.astype(np.float32)


def normalize(arr: np.ndarray):
    """Global min-max normalization; returns (normalized, min, max)."""
    mi, ma = float(arr.min()), float(arr.max())
    return (arr - mi) / (ma - mi + 1e-8), mi, ma


def denormalize(arr, mi: float, ma: float):
    return arr * (ma - mi + 1e-8) + mi


def split_dataset(dataset, test_ratio: float = 0.2, seed: int = 42):
    """
    Case-level split: (1 - test_ratio) train+valid / test_ratio test, then 90/10 train/valid.
    The same seed and call order are used by every correction model, so all of them
    share the same test cases.
    """
    n_total = len(dataset)
    n_train_total = int((1 - test_ratio) * n_total)
    torch.manual_seed(seed)
    train_val_ds, test_ds = random_split(dataset, [n_train_total, n_total - n_train_total])
    n_train = int(0.9 * n_train_total)
    train_ds, val_ds = random_split(train_val_ds, [n_train, n_train_total - n_train])
    return train_ds, val_ds, test_ds


def make_split_indices(n_cases: int, test_ratio: float = 0.2, seed: int = 42):
    """Train / valid / test case indices (into the full data set) identical to split_dataset()."""
    dummy = TensorDataset(torch.arange(n_cases))
    tr, va, te = split_dataset(dummy, test_ratio, seed)
    # tr / va are subsets of the train+valid subset: map their indices back to case ids
    parent = np.array(tr.dataset.indices)
    return parent[tr.indices], parent[va.indices], np.array(te.indices)
