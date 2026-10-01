# Physics-Informed Super-Resolution for Two-Phase Multi-scale Reservoir Simulation

Code for the paper *Physics-Informed Super-Resolution for Two-Phase Multi-scale Reservoir Simulation*
(H. Li, B. Aslam, J. Ma, Y. Wang, B. Yan).

The framework reconstructs fine-grid (100 x 100) pressure and water-saturation fields from
coarse-grid (20 x 20) simulation outputs. It combines

- a nine-channel physics-aware input (coarse state variables, permeability and well controls),
- a differentiable mass balance equation (MBE) loss that enforces global two-phase mass conservation,
- five SR backbones: Bicubic, EDSR, RCAN, SwinIR and SR-DNN,
- four coarse-grid fidelity correction models (LFLR -> HFLR) used as a preprocessing stage:
  FNO, FTNO, DeepONet and a flow-based diffusion model.

## Repository structure

```
sr/                          super-resolution framework (Sections 3.1-3.3, 4.2-4.5)
  main.py                    entry point: train and/or test one SR model
  train.py / test.py         training loop / test metrics (SSIM, MAE, Well MAE, R2)
  dataset.py                 nine-channel input (Table 2), 80/10/10 case split
  models.py                  Bicubic, EDSR, RCAN, SwinIR, SR-DNN
  loss.py                    gradient loss, SSIM
  mbe_physics.py             MBE loss (Eqs. 8-13)
fidelity_correction/         coarse-grid fidelity correction (Sections 3.4, 4.6)
  common.py                  data loading and case split shared by all models
  fno.py                     FNO (Table B1)
  deeponet.py                DeepONet (Table B2)
  ftno.py                    FTNO (Table B3); also exports the corrected coarse data for SR
  diffusion_model.py         flow-based diffusion model (Table B4)
  train_diffusion.py         training of the diffusion model
  extract_fine_cases.py      extracts the fine-grid cases matching the FTNO export
data/                        put the HDF5 files here (see below)
```

## Installation

```bash
conda create -n physr python=3.10 -y
conda activate physr
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -r requirements.txt
```

Experiments in the paper were run on a single NVIDIA RTX A6000 (48 GB).

## Data

The data sets were generated with CMG IMEX and are available on Zenodo:
**https://doi.org/10.5281/zenodo.XXXXXXX**. Download them into `data/`.

| File | Content | Used by |
|---|---|---|
| `coarse_pool_20x20.h5` | 9,000 coarse-grid cases (LFLR `sw_grid`, HFLR `sw_grid_downsampled`, ...) | fidelity correction models |
| `sr_coarse_20x20.h5` | the 1,800 cases held out from fidelity-correction training, with the FTNO-corrected saturation `sw_pred` | SR (coarse input) |
| `sr_fine_100x100.h5` | fine-grid fields of the same 1,800 cases | SR (target) |

Main keys (N cases, 11 timesteps with dt = 365 days): `pres_grid`, `sw_grid` (N, 11, H, W, 1);
`perm_grid`, `poro_grid` (N, H, W, 1); `rateinj_ts_{1,2}`, `ratewtr_ts_{1,2}`, `rateoil_ts_{1,2}`,
`bhpprd_ts_{1,2}` (N, 11); `inj_uba`, `prd_uba` (N, 2, 2) well locations (1-indexed).

The SR files can be regenerated from the pools with `ftno.py` followed by `extract_fine_cases.py`
(the latter also needs the 9,000-case fine-grid pool `fine_pool_100x100.h5`).

## Reproducing the experiments

All SR commands are run from `sr/`. `--max_cases N` uses the first N cases of the SR data,
split 80/10/10 into train/validation/test by case index (seed 42). Metrics are computed on the
test cases for timesteps 1-10 in physical units and written to
`<output_dir>/test_<model>_<field>/summary.txt`.

```bash
cd sr
```

### Table 3: saturation backbones (500 cases, lambda_MBE = 0.1)

```bash
for m in bicubic edsr swinir srdnn rcan; do
  python main.py --model $m --target_field saturation --max_cases 500 \
                 --lambda_mbe 0.1 --output_dir ../results/table3_$m
done
```

### Table 4: pressure backbones (500 cases, no MBE, no well loss)

```bash
for m in bicubic edsr swinir srdnn rcan; do
  python main.py --model $m --target_field pressure --max_cases 500 \
                 --no_well_loss --lr 5e-5 --patience 10 --output_dir ../results/table4_$m
done
```

### Table 5: MBE weight (RCAN, 500 cases)

```bash
for l in 0 0.01 0.1 0.5; do
  python main.py --model rcan --max_cases 500 --lambda_mbe $l --output_dir ../results/table5_mbe$l
done
```

### Table 7 / Figure 8: training data size (RCAN, with and without MBE)

```bash
for n in 200 500 1000 1500 1800; do
  for l in 0 0.1; do
    python main.py --model rcan --max_cases $n --lambda_mbe $l --output_dir ../results/table7_n${n}_mbe$l
  done
done
```

### Table 8: coarse-grid fidelity correction (LFLR -> HFLR)

Run from `fidelity_correction/`. All four models use the same 72/8/20 train/validation/test
split (seed 42), so they share the same test cases.

```bash
cd ../fidelity_correction
python fno.py             --data ../data/coarse_pool_20x20.h5
python deeponet.py        --data ../data/coarse_pool_20x20.h5
python train_diffusion.py --data ../data/coarse_pool_20x20.h5
python ftno.py            --data ../data/coarse_pool_20x20.h5 --export ../data/sr_coarse_20x20.h5
python extract_fine_cases.py --coarse ../data/sr_coarse_20x20.h5 \
                             --fine_pool ../data/fine_pool_100x100.h5 --out ../data/sr_fine_100x100.h5
```

Checkpoints are written to `fidelity_correction/checkpoints/`.

### Table 9: FTNO preprocessing + SR (RCAN)

`--debiased` replaces the coarse saturation input channel by the FTNO-corrected field `sw_pred`.

```bash
cd ../sr
for n in 500 1800; do
  python main.py --model rcan --max_cases $n --lambda_mbe 0   --output_dir ../results/table9_n${n}_lflr
  python main.py --model rcan --max_cases $n --lambda_mbe 0.1 --output_dir ../results/table9_n${n}_lflr_mbe
  python main.py --model rcan --max_cases $n --lambda_mbe 0   --debiased --output_dir ../results/table9_n${n}_ftno
  python main.py --model rcan --max_cases $n --lambda_mbe 0.1 --debiased --output_dir ../results/table9_n${n}_ftno_mbe
done
```

### Testing a trained model

```bash
python main.py --mode test --model rcan --max_cases 500 \
               --model_path ../results/table5_mbe0.1/sr_rcan_saturation.pth --output_dir ../results/eval
```

## Main options of `sr/main.py`

| Option | Default | Meaning |
|---|---|---|
| `--model` | `rcan` | `bicubic`, `edsr`, `rcan`, `swinir`, `srdnn` |
| `--target_field` | `saturation` | `saturation` or `pressure` |
| `--max_cases` | 500 | number of cases (80/10/10 split) |
| `--lambda_mbe` | 0.1 | MBE loss weight (saturation only) |
| `--debiased` | off | use FTNO-corrected coarse saturation |
| `--no_well_loss` | off | drop the well-region loss (used for pressure) |
| `--epochs` / `--batch_size` | 150 / 64 | |
| `--lr` / `--patience` | 1e-4 / 5 | 5e-5 / 10 for pressure |

Loss weights: lambda_grad = 2, lambda_well = 10 (Eq. 5). Training uses Adam, mixed precision on
GPU, and halves the learning rate when the validation loss plateaus. At test time, saturation
predictions are post-processed by setting flat, low-saturation regions to the background level
(`_adaptive_bg` in `test.py`).

## Citation

```
@article{li2026physr,
  title   = {Physics-Informed Super-Resolution for Two-Phase Multi-scale Reservoir Simulation},
  author  = {Li, Haotian and Aslam, Billal and Ma, Jiahao and Wang, Yating and Yan, Bicheng},
  year    = {2026}
}
```
