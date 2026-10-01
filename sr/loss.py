import torch
import torch.nn.functional as F


def gradient_loss(pred, target):
    """L1 difference of spatial gradients along x and y (L_grad, Eq. 6)."""
    def _grad(t):
        p = F.pad(t, (1, 1, 1, 1), mode='replicate')
        dx = p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]
        dy = p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]
        return dx, dy
    if pred.dim()   == 3: pred   = pred.unsqueeze(1)
    if target.dim() == 3: target = target.unsqueeze(1)
    dx_p, dy_p = _grad(pred)
    dx_t, dy_t = _grad(target)
    return torch.mean(torch.abs(dx_p - dx_t) + torch.abs(dy_p - dy_t))


# ── SSIM (evaluation metric) ──────────────────────────────────────────────────

def ssim(img1, img2, window_size=11, data_range=1.0, window=None, size_average=True):
    """SSIM with an 11x11 Gaussian window (sigma = 1.5)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    if window is None:
        window = create_window(window_size, img1.size(1))
    window = window.to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def create_window(window_size, channels):
    """2D Gaussian window."""
    _1d_window = gaussian_window(window_size, 1.5)
    _2d_window = _1d_window.unsqueeze(1) * _1d_window.unsqueeze(0)
    return _2d_window.expand(channels, 1, window_size, window_size).contiguous()


def gaussian_window(window_size, sigma):
    x = torch.arange(window_size)
    x = x - window_size // 2
    x = x.float() / sigma
    window = torch.exp(-0.5 * x ** 2)
    return window / window.sum()
