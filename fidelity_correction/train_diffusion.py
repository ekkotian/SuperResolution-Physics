"""
Train the flow-based diffusion model for coarse-grid fidelity correction (LFLR -> HFLR).

Saturation is mapped to logit space, g(x) = log(x / (1 - x)). Training uses a two-stage
curriculum (Table B4):
  stage 1: 300 epochs, 20 virtual steps
  stage 2: 200 epochs, 10 virtual steps, each split into 2 sub-steps
Validation integrates tau: 0 -> 1 in 10 virtual steps.
"""
import argparse
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from common import set_seed, make_split_indices
from diffusion_model import DiffModel

T_PHYS = 5            # physical horizon (yr) mapped onto tau in [0, 1]
N_VIR_EVAL = 10       # virtual steps at inference
FEATURE_TYPES = ['space', 'location', 'time', 'location']   # perm, injectors, inj. rate, producers


# ---------------- helpers ----------------
def g(x):
    eps = 1e-6
    return torch.log((x + eps) / (1 - x + eps))


def g_inv(x):
    return 1 / (1 + torch.exp(-x))


def minmax(data):
    return (data - data.min()) / (data.max() - data.min() + 0.01)


def location_map(uba):
    """Gaussian kernel map (N, 20, 20) of well locations, uba: (N, n_wells, 2) 1-indexed."""
    res = torch.zeros(uba.shape[0], 20, 20)
    yy, xx = np.meshgrid(np.arange(20), np.arange(20))
    for i in range(uba.shape[0]):
        m = 0
        for point in uba[i]:
            m += torch.tensor(np.exp(-20 * ((xx - point[1] + 1) ** 2 + (yy - point[0] + 1) ** 2) / (20 ** 2)))
        res[i] = m
    return res


def loss_re_fn(pred, target):
    """Relative L2 error per sample, root-mean over the batch."""
    return torch.sqrt(torch.mean(
        torch.norm((pred - target).reshape(pred.shape[0], -1), dim=1) /
        torch.norm(target.reshape(pred.shape[0], -1), dim=1)
    ))


class PathDataset(Dataset):
    """Pairs of consecutive points on the linear LFLR -> HFLR path (Eq. 14), in logit space."""
    def __init__(self, data, data_ds, features, time_vir):
        self.features = features
        self.time_vir = time_vir
        self.path = torch.stack([data + (data_ds - data) * vt / time_vir
                                 for vt in range(time_vir + 1)], dim=-1)

    def __len__(self):
        return self.path.shape[0] * self.time_vir

    def __getitem__(self, index):
        idx, k = index // self.time_vir, index % self.time_vir
        return (g(self.path[idx, ..., k]), g(self.path[idx, ..., k + 1]),
                [f[idx].float() for f in self.features])


# ---------------- main ----------------
def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with h5py.File(args.data, 'r') as f:
        solu      = np.array(f['sw_grid'])               # LFLR (N, 11, 20, 20, 1)
        solu_ds   = np.array(f['sw_grid_downsampled'])   # HFLR
        perm_grid = np.array(f['perm_grid'])
        rate_inj  = np.array(f['rateinj_ts_1'])
        inj_uba   = np.array(f['inj_uba'])
        prd_uba   = np.array(f['prd_uba'])

    # Same case split as the FNO / FTNO / DeepONet scripts
    train_idx, val_idx, test_idx = make_split_indices(solu.shape[0], args.test_ratio, args.seed)
    print(f'train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    perm_grid = (perm_grid - perm_grid.min()) / (perm_grid.max() - perm_grid.min())
    rate_inj  = (rate_inj - rate_inj.min()) / (rate_inj.max() - rate_inj.min())
    features = [
        torch.tensor(perm_grid[:, :, :, 0]),
        location_map(inj_uba),
        torch.tensor(rate_inj),
        location_map(prd_uba),
    ]

    def _fields(idx):
        # (n, 1, 10, 20, 20): timesteps 1-10 (t = 0 is identical for LFLR and HFLR)
        to_t = lambda a: torch.tensor(a[idx]).permute(0, 4, 1, 2, 3)[:, :, 1:].float()
        return to_t(solu), to_t(solu_ds), [fea[idx] for fea in features]

    x_tr, y_tr, f_tr = _fields(train_idx)
    x_va, y_va, f_va = _fields(val_idx)

    net = DiffModel(FEATURE_TYPES, 12).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, 'diff_sw_best.pth')
    best_val = float('inf')

    val_loader = DataLoader(PathDataset(x_va, y_va, f_va, 1), batch_size=1000)
    h_eval = T_PHYS / N_VIR_EVAL

    def _validate():
        total = 0.
        with torch.no_grad():
            for x0, x1, fea in val_loader:
                fea = [v.to(device) for v in fea]
                pred = x0.to(device)
                for _ in range(N_VIR_EVAL):
                    pred = net(pred, fea, h_eval)
                total += loss_re_fn(pred, x1.to(device)).item()
        return total / len(val_loader)

    # (epochs, virtual steps, sub-steps per virtual step)
    stages = [(args.epochs_stage1, 20, 1), (args.epochs_stage2, 10, 2)]
    for s, (epochs, vir_time, vir_div) in enumerate(stages, start=1):
        h = T_PHYS / vir_time / vir_div
        loader = DataLoader(PathDataset(x_tr, y_tr, f_tr, vir_time),
                            batch_size=200, shuffle=True, drop_last=True)
        for epoch in range(epochs):
            net.train()
            loss_sum = 0.
            for x0, x1, fea in loader:
                fea = [v.to(device) for v in fea]
                pred = x0.to(device)
                for _ in range(vir_div):
                    pred = net(pred, fea, h)
                loss = loss_re_fn(pred, x1.to(device))
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item()

            net.eval()
            val = _validate()
            print(f'stage{s} epoch {epoch}, train {loss_sum / len(loader):.6f}, val {val:.6f}')
            if val < best_val:
                best_val = val
                torch.save(net.state_dict(), best_path)
                print(f'  -> saved {best_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Flow-based diffusion fidelity correction (LFLR -> HFLR)')
    p.add_argument('--data', default='../data/coarse_pool_20x20.h5')
    p.add_argument('--out_dir', default='./checkpoints')
    p.add_argument('--epochs_stage1', type=int, default=300)
    p.add_argument('--epochs_stage2', type=int, default=200)
    p.add_argument('--test_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
