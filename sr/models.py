"""
SR backbones compared in the paper: Bicubic, EDSR, RCAN, SwinIR and SR-DNN.

All models share the same forward signature:
    forward(coarse_inputs, fine_perm=None) -> (B, H_fine, W_fine)

  coarse_inputs : (B, 9, H_c, W_c)   nine-channel physics-aware input (Table 2)
  fine_perm     : (B, 1, H_f, W_f)   normalized log fine-grid permeability

Every learned model encodes all 9 channels, upsamples with PixelShuffle x2 followed
by bicubic interpolation to the fine grid, conditions the features on fine_perm, and
predicts a residual on top of the bicubic-upsampled target channel.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_upsampler(channels: int, upscale_factor: int) -> nn.Sequential:
    """
    Progressive pixel-shuffle upsampler.
    For 5x/10x: 2x (then 2x again for 10x) via PixelShuffle.
    The remaining fractional factor is handled by bicubic interpolation (_upsample_to).
    """
    layers = []
    steps = 2 if upscale_factor == 10 else 1   # 5x: one 2x step; 10x: two 2x steps
    for _ in range(steps):
        layers += [
            nn.Conv2d(channels, channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.GELU(),
        ]
    return nn.Sequential(*layers)


def _upsample_to(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if x.shape[-2] == target_h and x.shape[-1] == target_w:
        return x
    return F.interpolate(x, size=(target_h, target_w),
                         mode='bicubic', align_corners=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Bicubic baseline (no parameters)
# ═══════════════════════════════════════════════════════════════════════════════

class BicubicSR(nn.Module):
    def __init__(self, upscale_factor: int, target_channel: int = 0):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.target_channel = target_channel

    def forward(self, coarse_inputs, fine_perm=None):
        x = coarse_inputs[:, self.target_channel:self.target_channel + 1]
        B, _, H, W = x.shape
        out = F.interpolate(x, size=(H * self.upscale_factor, W * self.upscale_factor),
                            mode='bicubic', align_corners=False)
        return out.squeeze(1).clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# EDSR  (Enhanced Deep SR, CVPR 2017)
# ═══════════════════════════════════════════════════════════════════════════════

class _EDSRBlock(nn.Module):
    def __init__(self, channels: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, upscale_factor: int = 5, target_channel: int = 0,
                 num_features: int = 64, num_blocks: int = 16,
                 in_channels: int = 9):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.target_channel = target_channel

        self.head    = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.body    = nn.Sequential(*[_EDSRBlock(num_features) for _ in range(num_blocks)])
        self.tail_up = _make_upsampler(num_features, upscale_factor)
        self.fine_perm_gate = FinePermGate(num_features)
        self.tail_out = nn.Conv2d(num_features, 1, 3, 1, 1)

    def forward(self, coarse_inputs, fine_perm=None):
        B, _, H, W = coarse_inputs.shape
        tH, tW = H * self.upscale_factor, W * self.upscale_factor

        baseline = F.interpolate(
            coarse_inputs[:, self.target_channel:self.target_channel + 1],
            size=(tH, tW), mode='bicubic', align_corners=False
        )

        feat = self.head(coarse_inputs)
        feat = feat + self.body(feat)
        feat = self.tail_up(feat)
        feat = _upsample_to(feat, tH, tW)
        if fine_perm is not None:
            feat = self.fine_perm_gate(feat, fine_perm)
        residual = self.tail_out(feat)

        return (baseline + residual).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# FinePermGate  (spatial attention conditioned on fine-grid permeability)
# ═══════════════════════════════════════════════════════════════════════════════

class FinePermGate(nn.Module):
    """
    Spatial attention gate driven by fine-grid permeability.
    High-perm regions amplify features; low-perm regions suppress them.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, fine_perm: torch.Tensor) -> torch.Tensor:
        return feat * self.gate(fine_perm)


# ═══════════════════════════════════════════════════════════════════════════════
# RCAN  (Residual Channel Attention Network, ECCV 2018)
# ═══════════════════════════════════════════════════════════════════════════════

class _CALayer(nn.Module):
    """Channel Attention Layer (squeeze-and-excitation)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))


class _RCAB(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            _CALayer(channels, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


class _ResGroup(nn.Module):
    def __init__(self, channels: int, n_rcab: int = 10, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            *[_RCAB(channels, reduction) for _ in range(n_rcab)],
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class RCAN(nn.Module):
    def __init__(self, upscale_factor: int = 5, target_channel: int = 0,
                 num_features: int = 64, num_groups: int = 5, num_rcab: int = 10,
                 in_channels: int = 9):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.target_channel = target_channel

        self.head    = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.body    = nn.Sequential(
            *[_ResGroup(num_features, num_rcab) for _ in range(num_groups)],
            nn.Conv2d(num_features, num_features, 3, 1, 1),
        )
        self.tail_up       = _make_upsampler(num_features, upscale_factor)
        self.fine_perm_gate = FinePermGate(num_features)
        self.tail_out       = nn.Conv2d(num_features, 1, 3, 1, 1)

    def forward(self, coarse_inputs, fine_perm=None):
        B, _, H, W = coarse_inputs.shape
        tH, tW = H * self.upscale_factor, W * self.upscale_factor

        baseline = F.interpolate(
            coarse_inputs[:, self.target_channel:self.target_channel + 1],
            size=(tH, tW), mode='bicubic', align_corners=False
        )

        feat = self.head(coarse_inputs)        # all 9 channels as input
        feat = feat + self.body(feat)
        feat = self.tail_up(feat)
        feat = _upsample_to(feat, tH, tW)
        if fine_perm is not None:
            feat = self.fine_perm_gate(feat, fine_perm)
        residual = self.tail_out(feat)

        return (baseline + residual).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# SwinIR  (Swin Transformer for Image Restoration, ICCV 2021)
# ═══════════════════════════════════════════════════════════════════════════════

def _window_partition(x, ws):
    """x: (B,H,W,C) → (B*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows, ws, H, W):
    """(B*nW, ws, ws, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class _WindowAttn(nn.Module):
    def __init__(self, dim: int, ws: int, num_heads: int):
        super().__init__()
        self.ws        = ws
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.rel_bias = nn.Parameter(torch.zeros((2*ws-1)**2, num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        ch = torch.arange(ws)
        cw = torch.arange(ws)
        grid = torch.stack(torch.meshgrid(ch, cw, indexing='ij'))  # 2,ws,ws
        flat = torch.flatten(grid, 1)                               # 2,ws²
        rel  = flat[:, :, None] - flat[:, None, :]                  # 2,ws²,ws²
        rel  = rel.permute(1, 2, 0).contiguous()                    # ws²,ws²,2
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer('rel_idx', rel.sum(-1))                # ws²,ws²

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.rel_bias[self.rel_idx.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class _SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ws: int, shift: bool):
        super().__init__()
        self.ws         = ws
        self.shift_size = ws // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _WindowAttn(dim, ws, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x, attn_mask=None):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))

        wins = _window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        wins = self.attn(wins, mask=attn_mask if self.shift_size > 0 else None)
        wins = wins.view(-1, self.ws, self.ws, C)
        x    = _window_reverse(wins, self.ws, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class _RSTB(nn.Module):
    """Residual Swin Transformer Block."""
    def __init__(self, dim: int, num_heads: int, ws: int, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            _SwinBlock(dim, num_heads, ws, shift=(i % 2 == 1))
            for i in range(num_layers)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, attn_mask=None):
        B, C, H, W = x.shape
        res = x
        x = x.permute(0, 2, 3, 1)          # B,H,W,C
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = x.permute(0, 3, 1, 2)          # B,C,H,W
        return res + self.conv(x)


class SwinIR(nn.Module):
    def __init__(self, upscale_factor: int = 5, target_channel: int = 0,
                 embed_dim: int = 60, num_heads: int = 4,
                 window_size: int = 4, num_rstb: int = 4, num_swin: int = 4,
                 in_channels: int = 9):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.target_channel = target_channel
        self.ws = window_size

        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        self.rstbs      = nn.ModuleList([
            _RSTB(embed_dim, num_heads, window_size, num_swin)
            for _ in range(num_rstb)
        ])
        self.norm            = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample        = _make_upsampler(embed_dim, upscale_factor)
        self.fine_perm_gate  = FinePermGate(embed_dim)
        self.conv_last       = nn.Conv2d(embed_dim, 1, 3, 1, 1)

    def _attn_mask(self, H: int, W: int, device):
        shift = self.ws // 2
        if shift == 0:
            return None
        img = torch.zeros(1, H, W, 1, device=device)
        for cnt, (hs, ws_) in enumerate(
            (hs, ws_)
            for hs in (slice(0, -self.ws), slice(-self.ws, -shift), slice(-shift, None))
            for ws_ in (slice(0, -self.ws), slice(-self.ws, -shift), slice(-shift, None))
        ):
            img[:, hs, ws_, :] = cnt
        wins = _window_partition(img, self.ws).view(-1, self.ws * self.ws)
        mask = wins.unsqueeze(1) - wins.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        return mask

    def forward(self, coarse_inputs, fine_perm=None):
        B, _, H, W = coarse_inputs.shape
        tH, tW = H * self.upscale_factor, W * self.upscale_factor

        baseline = F.interpolate(
            coarse_inputs[:, self.target_channel:self.target_channel + 1],
            size=(tH, tW), mode='bicubic', align_corners=False
        )

        x = coarse_inputs   # all 9 channels

        # Pad so H,W divisible by window_size
        ph = (self.ws - H % self.ws) % self.ws
        pw = (self.ws - W % self.ws) % self.ws
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        _, _, Hp, Wp = x.shape

        attn_mask = self._attn_mask(Hp, Wp, x.device)

        feat = self.conv_first(x)
        body = feat
        for rstb in self.rstbs:
            body = rstb(body, attn_mask)
        body = body.permute(0, 2, 3, 1)
        body = self.norm(body)
        body = body.permute(0, 3, 1, 2)
        feat = feat + self.conv_after_body(body)

        # Remove padding
        feat = feat[:, :, :H, :W]

        feat = self.upsample(feat)
        feat = _upsample_to(feat, tH, tW)
        if fine_perm is not None:
            feat = self.fine_perm_gate(feat, fine_perm)
        residual = self.conv_last(feat)

        return (baseline + residual).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# SR-DNN  (domain-specific reservoir SR, Aslam et al.)
# ═══════════════════════════════════════════════════════════════════════════════

class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class _SpatialAttn(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv  = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.proj(out)


class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm([dim])
        self.attn  = _SpatialAttn(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm([dim])
        self.mlp   = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_ratio, dim, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        xn = self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)
        x  = x + self.attn(xn)
        xn = self.norm2(x.permute(0,2,3,1)).permute(0,3,1,2)
        return x + self.mlp(xn)


class SRDNN(nn.Module):
    """
    Domain-specific SR model.
    Input: all 9 coarse channels + fine permeability fusion.
    """
    def __init__(self, upscale_factor: int = 5, target_channel: int = 0,
                 hidden_dim: int = 64, num_transformer_blocks: int = 3):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.target_channel = target_channel

        self.feature_extract = nn.Sequential(
            nn.Conv2d(9, hidden_dim, 3, 1, 1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.GELU(),
        )
        self.res_blocks  = nn.ModuleList([_ResBlock(hidden_dim) for _ in range(4)])
        self.trans_blocks = nn.ModuleList([
            _TransformerBlock(hidden_dim) for _ in range(num_transformer_blocks)
        ])

        # Upsampling (same strategy as shared upsampler)
        self.upscale = _make_upsampler(hidden_dim, upscale_factor)

        # Fine perm fusion (after upsampling to fine scale)
        self.fine_perm_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, 3, 1, 1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),     nn.GELU(),
        )

        # Multi-scale residual output
        self.out_small  = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim//2, 3,1,1), nn.GELU(), nn.Conv2d(hidden_dim//2, 1, 1))
        self.out_medium = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim//2, 5,1,2), nn.GELU(), nn.Conv2d(hidden_dim//2, 1, 1))
        self.out_large  = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim//2, 7,1,3), nn.GELU(), nn.Conv2d(hidden_dim//2, 1, 1))
        self.res_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, coarse_inputs, fine_perm=None):
        B, _, H, W = coarse_inputs.shape
        tH, tW = H * self.upscale_factor, W * self.upscale_factor

        baseline = F.interpolate(
            coarse_inputs[:, self.target_channel:self.target_channel+1],
            size=(tH, tW), mode='bicubic', align_corners=False
        )

        feat = self.feature_extract(coarse_inputs)
        for blk in self.res_blocks:
            feat = blk(feat)
        for blk in self.trans_blocks:
            feat = blk(feat)   # _TransformerBlock already has internal residuals

        feat = self.upscale(feat)
        feat = _upsample_to(feat, tH, tW)

        if fine_perm is not None:
            feat = self.fine_perm_fusion(torch.cat([feat, fine_perm], dim=1))

        residual = (self.out_small(feat) + self.out_medium(feat) + self.out_large(feat)) * self.res_weight
        return (baseline + residual).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(args, upscale_factor: int) -> nn.Module:
    """Return the SR model selected by args.model."""
    tc = 0 if args.target_field == 'pressure' else 1   # target channel index

    if args.model == 'bicubic':
        return BicubicSR(upscale_factor, tc)

    if args.model == 'edsr':
        return EDSR(upscale_factor, tc, in_channels=9)

    if args.model == 'rcan':
        return RCAN(upscale_factor, tc, in_channels=9)

    if args.model == 'swinir':
        return SwinIR(upscale_factor, tc, in_channels=9)

    if args.model == 'srdnn':
        return SRDNN(upscale_factor, tc,
                     hidden_dim=args.hidden_dim,
                     num_transformer_blocks=args.num_transformer_blocks)

    raise ValueError(f'Unknown model: {args.model}')
