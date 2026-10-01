"""
Flow-based diffusion model for coarse-grid fidelity correction (Table B4).

The network predicts one virtual step along the linear path
u(tau) = (1 - tau) * u_LFLR + tau * u_HFLR (Eq. 14), conditioned on spatial,
temporal and well-location features.
"""
import torch
import torch.nn as nn


class FeatureBlock(nn.Module):
    """Encodes one conditioning feature into a (B, rank, 10, 20, 20) tensor."""
    def __init__(self, feature_type, rank=4):
        super().__init__()
        self.fea_type = feature_type
        self.rank = rank
        if feature_type in ('space', 'location'):
            # (B, 20, 20) map
            self.encoder1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=3, padding=1), nn.GELU())
            self.encoder2 = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(64, rank, kernel_size=3, padding=1),
            )
        elif feature_type == 'time':
            # (B, 11) time series
            self.encoder1 = nn.Linear(11, 1024)
            self.encoder2 = nn.Linear(1024, rank * 10 * 20 * 20)
            self.act = nn.GELU()
        elif feature_type == 'space_time':
            self.encoder1 = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(64, rank, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f'Unknown feature type: {feature_type}')

    def forward(self, feature):
        if self.fea_type in ('space', 'location'):
            out = self.encoder1(feature.unsqueeze(1)).unsqueeze(1)
            return self.encoder2(out)
        if self.fea_type == 'time':
            out = self.encoder2(self.act(self.encoder1(feature)))
            return out.reshape(-1, self.rank, 10, 20, 20)
        return self.encoder1(feature)


class DiffModel(nn.Module):
    """One virtual step: u_next = u + h * f(u, features)."""
    def __init__(self, feature_type, rank):
        super().__init__()
        self.rank = rank
        self.en = nn.ModuleList([FeatureBlock(ft, rank=rank) for ft in feature_type])
        self.conv1 = nn.Conv3d(1, rank, kernel_size=3, padding=1)
        self.diff = nn.Sequential(
            nn.Conv3d(rank, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),   nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),    nn.GELU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),    nn.GELU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
        )
        self.lin_diff = nn.Sequential(
            nn.Linear(self.rank * 4000, 1024), nn.GELU(),
            nn.Linear(1024, 512),              nn.GELU(),
            nn.Linear(512, 4000),
        )
        self.act = nn.GELU()
        self.logits = nn.Parameter(torch.ones(len(feature_type)))

    def forward(self, x, feature, h):
        # x: (B, 1, 10, 20, 20) logit-transformed saturation
        weights = torch.softmax(self.logits, dim=0)
        out = self.act(self.conv1(x))
        for fea, en, weight in zip(feature, self.en, weights):
            out = out + weight * self.act(en(fea))
        return (h * self.diff(out)
                + h * self.lin_diff(out.view(-1, self.rank * 4000)).view(-1, 1, 10, 20, 20)
                + x)
