# Physics-Informed Super-Resolution for Two-Phase Multi-scale Reservoir Simulation

Code for the paper *Physics-Informed Super-Resolution for Two-Phase Multi-scale Reservoir Simulation*
(H. Li, B. Aslam, J. Ma, Y. Wang, B. Yan).

The framework reconstructs fine-grid (100 x 100) pressure and water-saturation fields from
coarse-grid (20 x 20) simulation outputs, using a nine-channel physics-aware input and a
differentiable mass balance equation (MBE) loss. Coarse-grid fidelity correction models
(FNO, FTNO, DeepONet, flow-based diffusion) can be applied as a preprocessing stage.

## Structure

```
sr/                     super-resolution framework (Bicubic, EDSR, RCAN, SwinIR, SR-DNN; MBE loss)
fidelity_correction/    coarse-grid fidelity correction models (LFLR -> HFLR)
data/                   HDF5 data sets (download from Zenodo)
```

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -r requirements.txt
```

## Data

Download the data sets from **https://doi.org/10.5281/zenodo.XXXXXXX** into `data/`:

- `sr_coarse_20x20.h5`, `sr_fine_100x100.h5`: paired coarse/fine cases for SR training
  (the coarse file also contains the FTNO-corrected saturation `sw_pred`)
- `coarse_pool_20x20.h5`: coarse-grid cases for training the fidelity correction models

## Usage

Super-resolution (run from `sr/`):

```bash
# saturation, RCAN with MBE loss
python main.py --model rcan --target_field saturation --max_cases 500 --lambda_mbe 0.1

# pressure
python main.py --model rcan --target_field pressure --max_cases 500 --no_well_loss --lr 5e-5 --patience 10

# saturation with FTNO-corrected coarse input
python main.py --model rcan --max_cases 500 --lambda_mbe 0.1 --debiased
```

Options: `--model` (`bicubic`, `edsr`, `rcan`, `swinir`, `srdnn`), `--max_cases` (number of cases,
split 80/10/10), `--lambda_mbe` (MBE weight, 0 = off), `--mode` (`train`, `test`, `both`),
`--output_dir`. Test metrics are written to `<output_dir>/test_<model>_<field>/summary.txt`.

Fidelity correction (run from `fidelity_correction/`):

```bash
python fno.py
python deeponet.py
python train_diffusion.py
python ftno.py   # also exports the corrected coarse data used by sr/main.py --debiased
```

## Citation

```
@article{li2026physr,
  title   = {Physics-Informed Super-Resolution for Two-Phase Multi-scale Reservoir Simulation},
  author  = {Li, Haotian and Aslam, Billal and Ma, Jiahao and Wang, Yating and Yan, Bicheng},
  year    = {2026}
}
```
