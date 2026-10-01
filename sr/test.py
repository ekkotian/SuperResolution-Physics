import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import ReservoirSRDataset
from models  import build_model, BicubicSR
from loss    import ssim as compute_ssim


# ── Metric helpers ────────────────────────────────────────────────────────────

def _psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target).item()
    return 10 * np.log10(data_range ** 2 / (mse + 1e-10))


def _r2(pred, target):
    ss_res = torch.sum((target - pred) ** 2).item()
    ss_tot = torch.sum((target - target.mean()) ** 2).item()
    if ss_tot < 1e-4:           # near-uniform field: R² is numerically meaningless
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _rel_mae(pred, target):
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8)).item() * 100.0


def _percentile(pred, target, q):
    return torch.quantile(torch.abs(pred - target).flatten(), q).item()


# ── Visualisation ─────────────────────────────────────────────────────────────

def _visualise(coarse_in, pred, target, well_mask, case_info, args, label, norm_params):
    tc = 0 if args.target_field == 'pressure' else 1
    coarse_f = coarse_in[tc].cpu().numpy()
    pred_f   = pred.cpu().numpy()
    true_f   = target.cpu().numpy()

    np_ = norm_params
    lo, hi = float(np_['target_min']), float(np_['target_max'])
    coarse_f = coarse_f * (hi - lo) + lo
    pred_f   = pred_f   * (hi - lo) + lo
    true_f   = true_f   * (hi - lo) + lo

    signed_err = pred_f - true_f
    rel_err    = np.abs(signed_err) / (np.abs(true_f) + 1e-10)
    rel_err    = np.minimum(rel_err, 1.0)
    cmap  = 'jet' if args.target_field == 'pressure' else 'viridis'
    unit  = 'Pressure (psi)' if args.target_field == 'pressure' else 'Saturation'
    vmin  = min(coarse_f.min(), true_f.min(), pred_f.min())
    vmax  = max(coarse_f.max(), true_f.max(), pred_f.max())

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, data, title in zip(axes[:3],
                                [coarse_f, true_f, pred_f],
                                ['Coarse', 'Ground Truth', 'Prediction']):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(unit)

    im3 = axes[3].imshow(rel_err, cmap=cmap, vmin=0, vmax=0.5)
    axes[3].set_title('Relative Error')
    cb = fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    tks = cb.get_ticks()
    cb.set_ticks(tks)
    cb.set_ticklabels([f'{int(t*100)}' for t in tks])
    cb.set_label('Relative Error (%)')

    err_lim = 200.0 if args.target_field == 'pressure' else 0.5
    im4 = axes[4].imshow(signed_err, cmap='RdBu_r', vmin=-err_lim, vmax=err_lim)
    axes[4].set_title('Signed Error (Pred - True)')
    cb4 = fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    cb4.set_label(unit)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    ci  = case_info['case_idx']
    ts  = case_info['timestep']
    mse = np.mean((pred_f - true_f) ** 2)
    ci_val = ci[0].item() if isinstance(ci, torch.Tensor) else ci
    ts_val = ts[0].item() if isinstance(ts, torch.Tensor) else ts
    plt.suptitle(f'[{args.model}] Case {ci_val}, T={ts_val}, MSE={mse:.5f}')
    plt.tight_layout(); plt.subplots_adjust(top=0.88)

    vis_dir = os.path.join(args.output_dir, 'test_vis')
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f'{label}.png'), dpi=150)
    plt.close()


# ── Main test function ─────────────────────────────────────────────────────────

def test_model(args, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_ds = ReservoirSRDataset(
        fine_file=args.fine_file, coarse_file=args.coarse_file,
        split='test', max_cases=args.max_cases,
        target_field=args.target_field,
        timesteps=args.timesteps, debiased=args.debiased,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    upscale_factor = test_ds.upscale_factor
    tc = 0 if args.target_field == 'pressure' else 1

    model = build_model(args, upscale_factor).to(device)

    if not isinstance(model, BicubicSR):
        ckpt_path = args.model_path or os.path.join(
            args.output_dir,
            f'sr_{args.model}_{args.target_field}.pth'
        )
        print(f'Loading checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  epoch={ckpt.get('epoch','?')}  "
              f"loss_config={ckpt.get('loss_config','N/A')}")

    model.eval()

    results_dir = os.path.join(args.output_dir,
                               f'test_{args.model}_{args.target_field}')
    os.makedirs(results_dir, exist_ok=True)

    # Accumulators
    keys = ['mse', 'mae', 'well_mae', 'well_mse',
            'ssim', 'psnr', 'r2', 'rel_mae_pct',
            'p90', 'p99', 'bicubic_mse', 'bicubic_mae']
    acc   = {k: 0. for k in keys}
    count = {k: 0  for k in keys}   # per-metric count to handle NaN (e.g. R²)
    ts_metrics = {}
    n = 0

    t0 = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader)):
            ci  = batch['coarse_inputs'].to(device)
            ft  = batch['fine_target'].to(device)
            fwm = batch['fine_well_mask'].to(device)
            fp  = batch['fine_perm'].to(device)
            info = batch['case_info']

            out = model(ci, fp)

            np_ = batch['norm_params']

            if args.target_field == 'saturation' and not isinstance(model, BicubicSR):
                out = _adaptive_bg(out)

            # Denormalize to physical units
            lo = float(np_['target_min'])
            hi = float(np_['target_max'])
            dr = hi - lo if hi > lo else 1.0   # data_range (psi or Sw)
            out_r = out * dr + lo
            ft_r  = ft  * dr + lo

            mse_v  = F.mse_loss(out_r, ft_r).item()
            mae_v  = F.l1_loss(out_r, ft_r).item()

            wl = fwm > 0
            well_mae_v = F.l1_loss(out_r[wl], ft_r[wl]).item() if wl.sum() > 0 else 0.
            well_mse_v = F.mse_loss(out_r[wl], ft_r[wl]).item() if wl.sum() > 0 else 0.

            ssim_v = compute_ssim(out_r.unsqueeze(1), ft_r.unsqueeze(1),
                                  data_range=dr).item()
            psnr_v = _psnr(out_r, ft_r, data_range=dr)
            r2_v   = _r2(out_r, ft_r)
            rel_v  = _rel_mae(out_r, ft_r)
            p90_v  = _percentile(out_r, ft_r, 0.90)
            p99_v  = _percentile(out_r, ft_r, 0.99)

            bicubic = F.interpolate(
                ci[:, tc:tc+1], scale_factor=upscale_factor,
                mode='bicubic', align_corners=False
            ).squeeze(1).clamp(0., 1.)
            bic_r   = bicubic * dr + lo
            bic_mse = F.mse_loss(bic_r, ft_r).item()
            bic_mae = F.l1_loss(bic_r, ft_r).item()

            sample_vals = dict(mse=mse_v, mae=mae_v,
                               well_mae=well_mae_v, well_mse=well_mse_v,
                               ssim=ssim_v, psnr=psnr_v, r2=r2_v,
                               rel_mae_pct=rel_v, p90=p90_v, p99=p99_v,
                               bicubic_mse=bic_mse, bicubic_mae=bic_mae)
            for k in keys:
                v = sample_vals[k]
                if not (isinstance(v, float) and np.isnan(v)):
                    acc[k]   += v
                    count[k] += 1
            n += 1

            ts = int(info['timestep'].item())
            if ts not in ts_metrics:
                ts_metrics[ts] = {k: [] for k in keys}
            for k in keys:
                v = sample_vals[k]
                if not (isinstance(v, float) and np.isnan(v)):
                    ts_metrics[ts][k].append(v)

            if idx % args.viz_freq == 0:
                ci_val = int(info['case_idx'].item())
                _visualise(ci[0], out[0], ft[0], fwm[0], info, args,
                            f'case{ci_val}_ts{ts}_{idx}', np_)

    avg = {k: (acc[k] / count[k] if count[k] > 0 else float('nan'))
           for k in keys}
    ts_avg = {ts: {k: float(np.mean(v)) for k, v in m.items()}
              for ts, m in ts_metrics.items()}

    # Print
    elapsed = time.time() - t0
    print(f'\nTest done in {elapsed:.1f}s  ({n} samples)')
    print('=== Results ===')
    for k, v in avg.items():
        print(f'  {k}: {v:.6f}')
    if avg['bicubic_mse'] > 0:
        print(f'  MSE improvement vs bicubic: '
              f'{(1 - avg["mse"] / avg["bicubic_mse"])*100:.1f}%')

    # Save summary
    _save_summary(avg, ts_avg, elapsed, n, args, results_dir, upscale_factor)
    _plot_ts_metrics(ts_avg, args, results_dir)

    return avg


# ── Adaptive background correction (saturation) ───────────────────────────────

def _adaptive_bg(predictions, val_thr=0.15, grad_thr=0.01):
    """Replace flat, low-saturation (un-swept) regions by the background level."""
    cand = (predictions <= val_thr).float()
    pad  = F.pad(predictions.unsqueeze(1), (1,1,1,1), mode='replicate')
    dx   = pad[:,:, 2:, 1:-1] - pad[:,:, :-2, 1:-1]
    dy   = pad[:,:, 1:-1, 2:] - pad[:,:, 1:-1, :-2]
    grad = torch.sqrt(dx**2 + dy**2).squeeze(1)
    mask = (grad < grad_thr).float() * cand

    out = predictions.clone()
    for b in range(predictions.size(0)):
        k  = max(1, int(predictions[b].numel() * 0.05))
        bg = torch.topk(predictions[b].view(-1), k, largest=False).values.mean()
        out[b] = predictions[b] * (1 - mask[b]) + bg * mask[b]
    return out.clamp(0., 1.)


# ── Summary text ─────────────────────────────────────────────────────────────

def _save_summary(avg, ts_avg, elapsed, n, args, results_dir, upscale_factor):
    path = os.path.join(results_dir, 'summary.txt')
    with open(path, 'w') as f:
        f.write(f'Model: {args.model}\n')
        f.write(f'Field: {args.target_field}  Upscale: {upscale_factor}x\n')
        units = 'psi' if args.target_field == 'pressure' else 'Sw (frac)'
        f.write(f'Metrics on denormalized data  '
                f'(MSE/MAE in {units}^2/{units}, PSNR/SSIM use physical range)\n')
        f.write(f'Samples: {n}  Time: {elapsed:.1f}s\n\n')
        f.write('=== Average Metrics ===\n')
        for k, v in avg.items():
            f.write(f'  {k}: {v:.6f}\n')
        if avg['bicubic_mse'] > 0:
            f.write(f'\n  MSE improvement vs bicubic: '
                    f'{(1 - avg["mse"] / avg["bicubic_mse"])*100:.2f}%\n')
        f.write('\n=== Per-timestep ===\n')
        disp = ['mse', 'mae', 'ssim', 'psnr', 'r2', 'well_mae', 'well_mse']
        f.write(f'{"TS":>3}  ' + '  '.join(f'{k:>9}' for k in disp) + '\n')
        f.write('-' * (5 + 11 * len(disp)) + '\n')
        for ts in sorted(ts_avg):
            row = f'{ts:>3}  ' + '  '.join(f'{ts_avg[ts].get(k,0):>9.5f}' for k in disp)
            f.write(row + '\n')
    print(f'Summary saved -> {path}')


# ── Timestep metric plot ──────────────────────────────────────────────────────

def _plot_ts_metrics(ts_avg, args, results_dir):
    ts_list = sorted(ts_avg)
    metrics_to_plot = ['mse', 'mae', 'ssim', 'well_mse']
    plt.figure(figsize=(12, 5))
    for m, style in zip(metrics_to_plot, ['o-', 's-', '^-', 'D-']):
        vals = [ts_avg[ts].get(m, 0) for ts in ts_list]
        plt.plot(ts_list, vals, style, label=m.upper())
    plt.xlabel('Timestep'); plt.ylabel('Value')
    plt.title(f'{args.model} - {args.target_field} metrics by timestep')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ts_metrics.png'), dpi=150)
    plt.close()
