"""
Fourier-Transformer Neural Operator (FTNO) for coarse-grid fidelity correction,
LFLR -> HFLR (Table B3).

Model: per-pixel lifting -> FT blocks (spectral conv + local conv + self-attention over
spatial tokens) -> projection. Loss: MSE + 0.05 x spatial-gradient loss.

After training, the held-out test cases are corrected with the best model and written,
together with all original keys, to an HDF5 file that adds
  sw_pred      : FTNO-corrected coarse saturation, (N_test, 11, 20, 20, 1)
  test_indices : case ids of these cases in the input file
This file is the coarse-grid input of the SR network (sr/main.py --debiased).
"""
import argparse
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from common import H, W, T, set_seed, load_sw_pair, normalize, denormalize, split_dataset


# ---------------- Model ----------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                                        dtype=torch.cfloat))

    def forward(self, x):
        B, C, Hx, Wx = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros((B, self.out_channels, Hx, Wx // 2 + 1), dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, x_ft.shape[2])
        m2 = min(self.modes2, x_ft.shape[3])
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy",
                                              x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(Hx, Wx), norm="ortho")


class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class FTBlock(nn.Module):
    def __init__(self, width, modes1, modes2, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(width)
        self.transformer = SpatialTransformerBlock(dim=width, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, kernel_size=1),
        )

    def forward(self, x):
        x = self.norm(x + self.spectral(x) + self.local(x))
        B, C, Hx, Wx = x.shape
        x_flat = self.transformer(x.permute(0, 2, 3, 1).reshape(B, Hx * Wx, C))
        x = x_flat.reshape(B, Hx, Wx, C).permute(0, 3, 1, 2).contiguous()
        return x + self.ffn(x)


class FTNO(nn.Module):
    def __init__(self, in_channels=T, out_channels=T, width=128, modes1=8, modes2=8,
                 layers=6, heads=8, mlp_ratio=4):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FTBlock(width, modes1, modes2, num_heads=heads, mlp_ratio=mlp_ratio)
                                     for _ in range(layers)])
        self.fc1 = nn.Linear(width, 256)
        self.fc2 = nn.Linear(256, out_channels)

    def forward(self, x):
        # x: (B, H, W, T)
        x = self.fc0(x).permute(0, 3, 1, 2).contiguous()
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.fc2(F.gelu(self.fc1(x)))


def spatial_grad_loss(pred, target):
    dxp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    dxt = target[:, 1:, :, :] - target[:, :-1, :, :]
    dyp = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dyt = target[:, :, 1:, :] - target[:, :, :-1, :]
    return torch.mean((dxp - dxt) ** 2) + torch.mean((dyp - dyt) ** 2)


# ---------------- Training ----------------
def train_ftno(args, device):
    set_seed(args.seed)
    x, y = load_sw_pair(args.data)
    x, x_min, x_max = normalize(x)
    y, y_min, y_max = normalize(y)
    ds = TensorDataset(torch.tensor(x), torch.tensor(y))
    train_ds, val_ds, test_ds = split_dataset(ds, args.test_ratio, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FTNO().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "ftno_sw_best.pth")
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb) + args.lambda_grad * spatial_grad_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        model.eval()
        vlosses, vmse = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                mse_val = loss_fn(pred, yb).item()
                vlosses.append(mse_val + args.lambda_grad * spatial_grad_loss(pred, yb).item())
                vmse.append(mse_val)

        avg_val = float(np.mean(vlosses))
        psnr = 10.0 * math.log10(1.0 / max(float(np.mean(vmse)), 1e-12))
        print(f"Epoch {ep}/{args.epochs} | train {np.mean(losses):.6e} | val {avg_val:.6e} "
              f"| PSNR {psnr:.2f} | {time.time() - t0:.1f}s")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({'model_state': model.state_dict(),
                        'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max},
                       best_path)
            print(f"  -> saved {best_path}")

    # reload the best weights for export
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state'])
    return model, test_ds, (y_min, y_max)


# ---------------- Export corrected test cases ----------------
def export_test_cases(model, test_ds, y_range, h5_path, out_path, device, batch_size=32):
    """Write all original keys of the test cases plus sw_pred and test_indices to out_path."""
    y_min, y_max = y_range
    test_indices = np.array(test_ds.indices)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in DataLoader(test_ds, batch_size=batch_size, shuffle=False):
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = denormalize(np.concatenate(preds, axis=0), y_min, y_max)  # (N, H, W, T)
    preds = preds.transpose(0, 3, 1, 2)[..., np.newaxis]              # (N, T, H, W, 1)

    # h5py fancy indexing requires increasing indices
    order = np.argsort(test_indices)
    sorted_idx = test_indices[order]
    preds = preds[order]

    with h5py.File(h5_path, "r") as f_src, h5py.File(out_path, "w") as f_dst:
        for key in f_src.keys():
            if key in ("sw_pred", "test_indices"):
                continue
            src = f_src[key]
            data = src[:] if (len(src.shape) == 1 or src.shape[0] == 1) else src[sorted_idx]
            f_dst.create_dataset(key, data=data, compression="gzip", compression_opts=4)
        f_dst.create_dataset("sw_pred", data=preds.astype(np.float32),
                             compression="gzip", compression_opts=4)
        f_dst.create_dataset("test_indices", data=sorted_idx.astype(np.int64))
    print(f"Saved {len(sorted_idx)} FTNO-corrected test cases -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FTNO fidelity correction (LFLR -> HFLR)")
    p.add_argument("--data", default="../data/coarse_pool_20x20.h5")
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--export", default="../data/sr_coarse_20x20.h5",
                   help="Output HDF5 with the corrected test cases (input of the SR network)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_grad", type=float, default=0.05)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, test_ds, y_range = train_ftno(args, device)
    export_test_cases(model, test_ds, y_range, args.data, args.export, device)
