"""
Fourier Neural Operator (FNO) for coarse-grid fidelity correction, LFLR -> HFLR (Table B1).

Input / output: saturation trajectory of shape (B, 20, 20, 11), the 11 timesteps as channels.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import set_seed, load_sw_pair, normalize, split_dataset


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT -> linear transform of the lowest modes -> inverse FFT."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=11, width=32, in_channels=11, out_channels=11):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)        # lifting

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)                # projection
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B, H, W, C)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = torch.relu(self.conv0(x) + self.w0(x))
        x = torch.relu(self.conv1(x) + self.w1(x))
        x = torch.relu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_fno(args, device):
    set_seed(args.seed)
    x, y = load_sw_pair(args.data)
    x, x_min, x_max = normalize(x)
    y, y_min, y_max = normalize(y)
    dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
    train_ds, val_ds, _ = split_dataset(dataset, args.test_ratio, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FNO2d().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "fno_sw_best.pth")
    best_loss = float("inf")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([loss_fn(model(xb.to(device)), yb.to(device)).item()
                                      for xb, yb in val_loader]))
        print(f"Epoch {ep}/{args.epochs} | train {np.mean(train_loss):.6f} | "
              f"val {val_loss:.6f} | {time.time() - t0:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model_state': model.state_dict(),
                        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max},
                       best_path)
            print(f"  -> saved {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FNO fidelity correction (LFLR -> HFLR)")
    p.add_argument("--data", default="../data/coarse_pool_20x20.h5")
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_fno(args, "cuda" if torch.cuda.is_available() else "cpu")
