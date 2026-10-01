"""
DeepONet for coarse-grid fidelity correction, LFLR -> HFLR (Table B2).

- Branch net encodes the whole flattened LFLR trajectory (20 x 20 x 11).
- Trunk net encodes query coordinates (x, y, t) in [0, 1]^3.
- Output = <branch, trunk> + bias at each queried point.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import set_seed, load_sw_pair, normalize, split_dataset


def make_full_coord_grid(H: int, W: int, T: int, device="cpu"):
    """(H*W*T, 3) grid of (x, y, t) coordinates in [0, 1]."""
    ys = torch.linspace(0.0, 1.0, H)
    xs = torch.linspace(0.0, 1.0, W)
    ts = torch.linspace(0.0, 1.0, T)
    Y, X, Tm = torch.meshgrid(ys, xs, ts, indexing="ij")
    return torch.stack([X, Y, Tm], dim=-1).reshape(-1, 3).to(device)


def sample_point_indices(num_points_total: int, k: int):
    if k >= num_points_total:
        idx = np.arange(num_points_total)
    else:
        idx = np.random.choice(num_points_total, size=k, replace=False)
    return torch.from_numpy(idx.astype(np.int64))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(self, branch_in_dim, trunk_in_dim=3, latent_dim=128,
                 branch_hidden=(512, 512, 256), trunk_hidden=(256, 256, 256)):
        super().__init__()
        self.branch = MLP(branch_in_dim, branch_hidden, latent_dim)
        self.trunk = MLP(trunk_in_dim, trunk_hidden, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_batch, coords_batch):
        """
        u_batch      : (B, H*W*T) flattened LFLR field
        coords_batch : (P, 3) shared query points, or (B, P, 3)
        returns      : (B, P)
        """
        if coords_batch.dim() == 2:
            coords_batch = coords_batch.unsqueeze(0).expand(u_batch.size(0), -1, -1)
        B, P, _ = coords_batch.shape
        b = self.branch(u_batch)                                         # (B, latent)
        t = self.trunk(coords_batch.reshape(B * P, -1)).reshape(B, P, -1)  # (B, P, latent)
        return (b.unsqueeze(1) * t).sum(dim=-1) + self.bias


def train_deeponet(args, device):
    set_seed(args.seed)
    x_np, y_np = load_sw_pair(args.data)
    x_np, x_min, x_max = normalize(x_np)
    y_np, y_min, y_max = normalize(y_np)
    N, H, W, T = x_np.shape

    dataset = TensorDataset(torch.tensor(x_np), torch.tensor(y_np))
    train_ds, val_ds, _ = split_dataset(dataset, args.test_ratio, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DeepONet(branch_in_dim=H * W * T, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    full_coords = make_full_coord_grid(H, W, T, device=device)
    P_total = full_coords.size(0)

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "deeponet_sw_best.pth")
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # random subset of query points, shared across the batch
            idx = sample_point_indices(P_total, args.points_per_sample).to(device)
            pred = model(xb.reshape(xb.size(0), -1), full_coords[idx])
            loss = loss_fn(pred, yb.reshape(yb.size(0), -1)[:, idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation on the full grid
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb.reshape(xb.size(0), -1), full_coords)
                val_losses.append(loss_fn(pred, yb.reshape(yb.size(0), -1)).item())

        mean_val = float(np.mean(val_losses))
        print(f"Epoch {ep:04d}/{args.epochs} | train {np.mean(train_losses):.6f} | "
              f"val {mean_val:.6f} | {time.time() - t0:.1f}s")

        if mean_val < best_val:
            best_val = mean_val
            torch.save({'model_state': model.state_dict(),
                        'H': H, 'W': W, 'T': T, 'latent_dim': args.latent_dim,
                        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max},
                       best_path)
            print(f"  -> saved {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DeepONet fidelity correction (LFLR -> HFLR)")
    p.add_argument("--data", default="../data/coarse_pool_20x20.h5")
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--points_per_sample", type=int, default=4096)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_deeponet(args, "cuda" if torch.cuda.is_available() else "cpu")
