"""
Physics-informed super-resolution (SR) of reservoir state variables.

Train and/or test one SR backbone on the paired coarse (20x20) / fine (100x100)
data set. See README.md for the exact command behind each table of the paper.
"""
import argparse
import os

import torch

from train import train_model
from test import test_model

# Default data location: <repo>/data, independent of the working directory
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def parse_args():
    p = argparse.ArgumentParser(description='Physics-informed reservoir SR')

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument('--fine_file',   default=os.path.join(_DATA, 'sr_fine_100x100.h5'))
    p.add_argument('--coarse_file', default=os.path.join(_DATA, 'sr_coarse_20x20.h5'))
    p.add_argument('--target_field', default='saturation', choices=['pressure', 'saturation'])
    p.add_argument('--timesteps', type=int, nargs='+', default=None,
                   help='Timesteps to use (1-10). Default: 1-10 (t=0 is excluded)')
    p.add_argument('--max_cases', type=int, default=500,
                   help='Number of cases used (split 80/10/10 into train/valid/test)')
    p.add_argument('--debiased', action='store_true',
                   help='Use the FTNO-corrected coarse saturation (sw_pred) as input channel 1')

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument('--model', default='rcan',
                   choices=['bicubic', 'edsr', 'rcan', 'swinir', 'srdnn'])
    p.add_argument('--hidden_dim',             type=int, default=64,  help='SR-DNN only')
    p.add_argument('--num_transformer_blocks', type=int, default=3,   help='SR-DNN only')

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument('--mode',       default='both', choices=['train', 'test', 'both'])
    p.add_argument('--epochs',     type=int,   default=150)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-4)  # saturation: 1e-4 | pressure: 5e-5
    p.add_argument('--patience',   type=int,   default=5)     # saturation: 5    | pressure: 10

    # ── Loss terms ────────────────────────────────────────────────────────────
    p.add_argument('--no_well_loss',     action='store_true')
    p.add_argument('--no_gradient_loss', action='store_true')
    p.add_argument('--lambda_mbe', type=float, default=0.1,
                   help='Weight of the MBE loss (saturation only, 0 = off)')

    # ── Output / IO ───────────────────────────────────────────────────────────
    p.add_argument('--output_dir', default='./results/rcan_saturation')
    p.add_argument('--model_path', default=None,
                   help='Checkpoint to test (default: <output_dir>/sr_<model>_<field>.pth)')
    p.add_argument('--resume',   action='store_true', help='Resume from the saved checkpoint')
    p.add_argument('--viz_freq', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    # Bicubic has no trainable weights
    if args.model == 'bicubic':
        args.mode = 'test'

    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f'Device : {device}  ({torch.cuda.get_device_name(args.gpu)})')
    else:
        device = torch.device('cpu')
        print('Device : cpu')
    print(f'Model  : {args.model}')
    print(f'Field  : {args.target_field}')
    print(f'Cases  : {args.max_cases}')

    if args.mode in ('train', 'both'):
        train_model(args, device)

    if args.mode in ('test', 'both'):
        metrics = test_model(args, device)
        print('\n=== Test Results ===')
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f'  {k}: {v:.6f}')


if __name__ == '__main__':
    main()
