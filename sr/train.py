import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset     import ReservoirSRDataset
from models      import build_model, BicubicSR
from loss        import gradient_loss
from mbe_physics import global_balance


# Loss weights (Eq. 5); lambda_MBE is set from the command line
LAMBDA_PIXEL = 1.0
LAMBDA_GRAD  = 2.0
LAMBDA_WELL  = 10.0


def _mbe_terms(out, batch, device):
    """Water / oil relative mass-balance errors of the de-normalized prediction."""
    tlo = batch['norm_params']['target_min'].float().to(device).view(-1, 1, 1)
    thi = batch['norm_params']['target_max'].float().to(device).view(-1, 1, 1)
    out_phys = out * (thi - tlo) + tlo
    return global_balance(
        out_phys,
        batch['Sw_cur_phys'].to(device),
        batch['poro_ref'].to(device),
        batch['inj_maps'].to(device),
        batch['prod_maps'].to(device),
        batch['rates_inj'].to(device),
        batch['rates_wtr'].to(device),
        batch['rates_oil'].to(device),
        dt=365.0, grid='fine',
    )


# ── Main training function ────────────────────────────────────────────────────

def train_model(args, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'bicubic':
        print('Bicubic: no training needed.')
        return None, [], []

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    def _make_ds(split):
        return ReservoirSRDataset(
            fine_file=args.fine_file, coarse_file=args.coarse_file,
            split=split, max_cases=args.max_cases,
            target_field=args.target_field,
            timesteps=args.timesteps, debiased=args.debiased,
        )

    train_ds = _make_ds('train')
    valid_ds = _make_ds('valid')
    upscale_factor = train_ds.upscale_factor

    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args, upscale_factor).to(device)
    if isinstance(model, BicubicSR):
        print('Bicubic: no training needed.')
        return model, [], []

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model [{args.model}]: {n_params:,} parameters')

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience)
    use_amp = device.type == 'cuda'
    scaler  = GradScaler('cuda', enabled=use_amp)

    # ── Loss configuration ────────────────────────────────────────────────────
    is_sat       = args.target_field == 'saturation'
    lambda_mbe   = args.lambda_mbe if is_sat else 0.0   # MBE applies to saturation only
    use_well     = not args.no_well_loss
    use_gradient = not args.no_gradient_loss

    active = ['Pixel(MSE)']
    if use_gradient:   active.append(f'Gradient(x{LAMBDA_GRAD})')
    if use_well:       active.append(f'Well(x{LAMBDA_WELL})')
    if lambda_mbe > 0: active.append(f'MBE(x{lambda_mbe})')
    print(f'Active loss terms: {" + ".join(active)}')

    def _loss(out, ft, fwm, batch, training):
        pixel = nn.functional.mse_loss(out, ft)
        wl    = fwm > 0
        well  = nn.functional.mse_loss(out[wl], ft[wl]) if wl.sum() > 0 else pixel.detach() * 0
        grad  = gradient_loss(out, ft)
        if lambda_mbe > 0:
            w_rel, o_rel = _mbe_terms(out, batch, device)
            # Training back-propagates the water-phase term; validation monitors both phases
            mbe = w_rel.mean() if training else (w_rel.mean() + o_rel.mean()) * 0.5
        else:
            mbe = pixel.detach() * 0

        loss = LAMBDA_PIXEL * pixel
        if use_gradient:   loss = loss + LAMBDA_GRAD * grad
        if use_well:       loss = loss + LAMBDA_WELL * well
        if lambda_mbe > 0: loss = loss + lambda_mbe * mbe
        return loss, dict(pixel=pixel.item(), well=well.item(), grad=grad.item(), mbe=mbe.item())

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.output_dir, f'sr_{args.model}_{args.target_field}.pth')
    log_path  = os.path.join(args.output_dir, f'log_{args.model}.txt')

    start_epoch = 0
    best_valid  = float('inf')
    do_resume   = args.resume and os.path.exists(ckpt_path)

    if do_resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_valid  = ckpt['valid_loss']
        print(f'Resumed from epoch {start_epoch}  (best valid so far: {best_valid:.6f})')

    with open(log_path, 'a' if do_resume else 'w') as f:
        if do_resume:
            f.write(f'\n--- Resumed from epoch {start_epoch} ---\n')
        else:
            f.write(f'Model: {args.model}\n')
            f.write(f'Loss terms: {" + ".join(active)}\n')
            f.write('Epoch,TrainLoss,ValidLoss,Pixel,Well,Grad,MBE,Time,LR\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, valid_losses = [], []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        model.train()
        tr_sum, tr_n = 0., 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)
        for batch in pbar:
            ci  = batch['coarse_inputs'].to(device)
            ft  = batch['fine_target'].to(device)
            fwm = batch['fine_well_mask'].to(device)
            fp  = batch['fine_perm'].to(device)

            with autocast('cuda', enabled=use_amp):
                out = model(ci, fp)
                loss, _ = _loss(out, ft, fwm, batch, training=True)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = ci.size(0)
            tr_sum += loss.item() * bs
            tr_n   += bs
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_train = tr_sum / tr_n

        # Validation
        model.eval()
        va = dict(loss=0., pixel=0., well=0., grad=0., mbe=0.)
        va_n = 0
        with torch.no_grad():
            for batch in valid_loader:
                ci  = batch['coarse_inputs'].to(device)
                ft  = batch['fine_target'].to(device)
                fwm = batch['fine_well_mask'].to(device)
                fp  = batch['fine_perm'].to(device)

                out = model(ci, fp)
                vloss, terms = _loss(out, ft, fwm, batch, training=False)

                bs = ci.size(0)
                va['loss'] += vloss.item() * bs
                for k, v in terms.items():
                    va[k] += v * bs
                va_n += bs

        va = {k: v / va_n for k, v in va.items()}
        avg_valid = va['loss']
        train_losses.append(avg_train)
        valid_losses.append(avg_valid)

        scheduler.step(avg_valid)
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        print(f'[{epoch+1:>4}/{args.epochs}]  train={avg_train:.4f}  valid={avg_valid:.4f}  '
              f'pixel={va["pixel"]:.4f}  well={va["well"]:.4f}  grad={va["grad"]:.4f}  '
              + (f'mbe={va["mbe"]:.4f}  ' if lambda_mbe > 0 else '')
              + f'lr={lr:.2e}  t={dt:.0f}s')

        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{avg_train:.6f},{avg_valid:.6f},{va["pixel"]:.6f},'
                    f'{va["well"]:.6f},{va["grad"]:.6f},{va["mbe"]:.6f},{dt:.1f},{lr:.2e}\n')

        if avg_valid < best_valid:
            best_valid = avg_valid
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': avg_valid,
                'upscale_factor': upscale_factor,
                'target_field': args.target_field,
                'model_name': args.model,
                'loss_config': {'use_well': use_well, 'use_gradient': use_gradient,
                                'lambda_mbe': lambda_mbe},
            }, ckpt_path)
            print(f'  -> checkpoint saved (valid={avg_valid:.4f})')

    _plot_loss(train_losses, valid_losses, args)
    print(f'Training done. Best valid loss: {best_valid:.6f}')
    return model, train_losses, valid_losses


# ── Loss curve plot ───────────────────────────────────────────────────────────

def _plot_loss(train, valid, args):
    fig_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    ep = range(1, len(train) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(ep, train, label='Train')
    plt.plot(ep, valid, label='Valid')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{args.model} - {args.target_field}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'loss_{args.model}_{args.target_field}.png'), dpi=150)
    plt.close()
