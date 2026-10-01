import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import List, Optional

from mbe_physics import build_well_maps


class ReservoirSRDataset(Dataset):
    """
    Paired coarse (20x20) / fine (100x100) samples, one sample = one case x one timestep.

    Coarse input: 9 channels (Table 2 of the paper)
      0 pressure | 1 Sw (or FTNO-corrected Sw if debiased) | 2 log-permeability |
      3-4 cumulative injection I1, I2 | 5-6 injection rate I1, I2 | 7-8 BHP P1, P2
    Cases are split 80/10/10 (train/valid/test) by case index with seed 42.
    """
    def __init__(self,
                 fine_file: str,
                 coarse_file: str,
                 split: str = 'train',
                 max_cases: int = 500,
                 target_field: str = 'saturation',
                 timesteps: Optional[List[int]] = None,
                 debiased: bool = False):

        self.fine_file_path   = fine_file
        self.coarse_file_path = coarse_file
        self.split        = split
        self.target_field = target_field
        self.timesteps    = timesteps if timesteps is not None else list(range(1, 11))
        self.debiased     = debiased
        self._coarse_sw_key = 'sw_pred' if debiased else 'sw_grid'

        self.fine_data   = h5py.File(fine_file,   'r')
        self.coarse_data = h5py.File(coarse_file, 'r')

        self.max_cases    = min(max_cases, len(self.fine_data['pres_grid']))
        self.fine_shape   = self.fine_data['pres_grid'].shape[2:4]
        self.coarse_shape = self.coarse_data['pres_grid'].shape[2:4]
        self.upscale_factor = self.fine_shape[0] // self.coarse_shape[0]

        print(f'Grid: coarse {self.coarse_shape} -> fine {self.fine_shape}  '
              f'({self.upscale_factor}x)')

        self._compute_global_stats()
        self._build_case_list()
        self._split()

    # ── Serialisation (h5py can't be pickled) ────────────────────────────────
    def __getstate__(self):
        s = self.__dict__.copy()
        s.pop('fine_data',   None)
        s.pop('coarse_data', None)
        s.pop('_static_cache', None)   # per-process cache, don't pickle
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fine_data   = h5py.File(self.fine_file_path,   'r')
        self.coarse_data = h5py.File(self.coarse_file_path, 'r')
        self._static_cache = {}

    def _get_static(self, ci: int) -> dict:
        """Cache static per-case fields (same for all timesteps)."""
        if not hasattr(self, '_static_cache'):
            self._static_cache = {}
        if ci not in self._static_cache:
            c_perm   = self.coarse_data['perm_grid'][ci].squeeze(-1)
            f_perm   = self.fine_data['perm_grid'][ci].squeeze(-1)
            c_perm_n, _, _ = self._log_minmax(c_perm)
            f_perm_n, _, _ = self._log_minmax(f_perm)
            _, f_well_mask = self._well_mask(ci)
            poro_ref = self.fine_data['poro_grid'][ci].squeeze(-1).astype(np.float32)
            fH, fW   = self.fine_shape
            inj_uba  = self.fine_data['inj_uba'][ci]
            prd_uba  = self.fine_data['prd_uba'][ci]
            inj_maps, prod_maps = build_well_maps(inj_uba, prd_uba, fH, fW)
            self._static_cache[ci] = dict(
                c_perm_n=c_perm_n, f_perm_n=f_perm_n,
                f_well_mask=f_well_mask, poro_ref=poro_ref,
                inj_maps=inj_maps, prod_maps=prod_maps,
            )
        return self._static_cache[ci]

    # ── Global statistics ─────────────────────────────────────────────────────
    def _compute_global_stats(self):
        """Compute global min/max for pressure, Sw, injection rate and cumulative injection."""
        n       = min(self.max_cases, len(self.fine_data['pres_grid']))
        step    = max(1, n // 50)                             # sample ≤50 cases
        all_ts  = list(range(11))                             # always iterate all t for cumulative

        pres_min_vals, pres_max_vals = [], []
        sw_min_vals,   sw_max_vals   = [], []
        inj_rate_vals  = []
        cum_inj_vals   = []

        for ci in range(0, n, step):
            cum1 = cum2 = 0.
            for t in all_ts:
                # Pressure & Sw  (sampled timesteps only to save time)
                if t in self.timesteps:
                    p  = self.fine_data['pres_grid'][ci, t]
                    sw = self.fine_data['sw_grid'][ci, t]
                    pres_min_vals.append(float(p.min()));  pres_max_vals.append(float(p.max()))
                    sw_min_vals.append(float(sw.min()));   sw_max_vals.append(float(sw.max()))
                    inj_rate_vals += [
                        float(self.fine_data['rateinj_ts_1'][ci, t]),
                        float(self.fine_data['rateinj_ts_2'][ci, t]),
                    ]
                # Cumulative injection (always sum in order 0→10)
                cum1 += float(self.fine_data['rateinj_ts_1'][ci, t])
                cum2 += float(self.fine_data['rateinj_ts_2'][ci, t])
                if t in self.timesteps:
                    cum_inj_vals += [cum1, cum2]

        self.global_pres_min    = min(pres_min_vals)
        self.global_pres_max    = max(pres_max_vals)
        self.global_sw_min      = min(sw_min_vals)
        self.global_sw_max      = max(sw_max_vals)
        self.global_inj_min     = min(inj_rate_vals)
        self.global_inj_max     = max(inj_rate_vals)
        self.global_cum_inj_min = min(cum_inj_vals) if cum_inj_vals else 0.
        self.global_cum_inj_max = max(cum_inj_vals) if cum_inj_vals else 1.

        print(f'Global pressure  : [{self.global_pres_min:.1f}, {self.global_pres_max:.1f}] psi')
        print(f'Global Sw        : [{self.global_sw_min:.4f}, {self.global_sw_max:.4f}]')
        print(f'Global inj rate  : [{self.global_inj_min:.4f}, {self.global_inj_max:.4f}]')
        print(f'Global cum inj   : [{self.global_cum_inj_min:.4f}, {self.global_cum_inj_max:.4f}]')

    # ── Case list / split ─────────────────────────────────────────────────────
    def _build_case_list(self):
        self.cases = [
            {'case_idx': ci, 'timestep': t}
            for ci in range(self.max_cases)
            for t  in self.timesteps
        ]
        print(f'Total samples: {len(self.cases)}')

    def _split(self):
        unique = sorted({c['case_idx'] for c in self.cases})
        np.random.seed(42)
        np.random.shuffle(unique)
        n  = len(unique)
        t1 = int(n * 0.8)
        t2 = int(n * 0.9)
        if self.split == 'train':
            sel = set(unique[:t1])
        elif self.split == 'valid':
            sel = set(unique[t1:t2])
        else:
            sel = set(unique[t2:])
        self.cases = [c for c in self.cases if c['case_idx'] in sel]
        print(f'{self.split}: {len(sel)} cases, {len(self.cases)} samples')

    # ── Normalisation helpers ─────────────────────────────────────────────────
    @staticmethod
    def _minmax(data, lo=None, hi=None):
        if lo is None: lo = data.min()
        if hi is None: hi = data.max()
        return (data - lo) / (hi - lo + 1e-8), lo, hi

    @staticmethod
    def _log_minmax(data, log_lo=None, log_hi=None, eps=1e-8):
        ld = np.log(data + eps)
        if log_lo is None: log_lo = ld.min()
        if log_hi is None: log_hi = ld.max()
        return (ld - log_lo) / (log_hi - log_lo + 1e-8), log_lo, log_hi

    # ── Well masks ────────────────────────────────────────────────────────────
    def _well_mask(self, case_idx):
        fine_mask   = np.zeros(self.fine_shape)
        coarse_mask = np.zeros(self.coarse_shape)

        def _set(mask, shape, src, idx):
            x, y = src[case_idx, idx]
            if x > 0 and y > 0:
                r, c = int(y) - 1, int(x) - 1
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    mask[r, c] = 1.0

        for i in range(2):
            _set(fine_mask,   self.fine_shape,   self.fine_data['inj_uba'],   i)
            _set(fine_mask,   self.fine_shape,   self.fine_data['prd_uba'],   i)
            _set(coarse_mask, self.coarse_shape, self.coarse_data['inj_uba'], i)
            _set(coarse_mask, self.coarse_shape, self.coarse_data['prd_uba'], i)

        # fallback: downsample fine mask
        if coarse_mask.sum() == 0 and fine_mask.sum() > 0:
            s = self.upscale_factor
            for i in range(self.coarse_shape[0]):
                for j in range(self.coarse_shape[1]):
                    if fine_mask[i*s:(i+1)*s, j*s:(j+1)*s].any():
                        coarse_mask[i, j] = 1.0

        return coarse_mask, fine_mask

    # ── Broadcast scalar helpers ──────────────────────────────────────────────
    def _broadcast(self, value: float) -> np.ndarray:
        """Fill the entire coarse grid with a single scalar value."""
        return np.full(self.coarse_shape, value, dtype=np.float32)

    def _norm_scalar(self, value: float, lo: float, hi: float) -> float:
        return (value - lo) / (hi - lo + 1e-8)

    def _cum_inj(self, case_idx: int, timestep: int, well: int) -> float:
        """Cumulative injection of well (1-indexed) from t=0 to t=timestep."""
        return float(self.fine_data[f'rateinj_ts_{well}'][case_idx, :timestep + 1].sum())

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        info  = self.cases[idx]
        ci, t = info['case_idx'], info['timestep']

        try:
            # ── Static per-case fields (cached across timesteps) ──────────────
            sc = self._get_static(ci)
            c_perm_n    = sc['c_perm_n']
            f_perm_n    = sc['f_perm_n']
            f_well_mask = sc['f_well_mask']

            # ── Time-varying spatial fields ───────────────────────────────────
            c_pres = self.coarse_data['pres_grid'][ci, t].squeeze(-1)
            c_sw   = self.coarse_data[self._coarse_sw_key][ci, t].squeeze(-1)
            f_sw   = self.fine_data['sw_grid'][ci, t].squeeze(-1)
            f_pres = self.fine_data['pres_grid'][ci, t].squeeze(-1)

            # ── Spatial normalisation ─────────────────────────────────────────
            c_pres_n, plo, phi = self._minmax(c_pres,
                                              self.global_pres_min, self.global_pres_max)
            c_sw_n,   slo, shi = self._minmax(c_sw)

            # ── Target (fine grid) ────────────────────────────────────────────
            if self.target_field == 'pressure':
                f_target_n, tlo, thi = self._minmax(f_pres, plo, phi)
            else:
                f_target_n, tlo, thi = self._minmax(f_sw, slo, shi)

            # ── Scalar well controls → broadcast to full coarse grid ──────────
            # Cumulative injection I1, I2
            cum_I1 = self._cum_inj(ci, t, 1)
            cum_I2 = self._cum_inj(ci, t, 2)
            ch_cum_I1 = self._broadcast(self._norm_scalar(
                cum_I1, self.global_cum_inj_min, self.global_cum_inj_max))
            ch_cum_I2 = self._broadcast(self._norm_scalar(
                cum_I2, self.global_cum_inj_min, self.global_cum_inj_max))

            # Current injection rate I1, I2
            rate_I1 = float(self.fine_data['rateinj_ts_1'][ci, t])
            rate_I2 = float(self.fine_data['rateinj_ts_2'][ci, t])
            ch_rate_I1 = self._broadcast(self._norm_scalar(
                rate_I1, self.global_inj_min, self.global_inj_max))
            ch_rate_I2 = self._broadcast(self._norm_scalar(
                rate_I2, self.global_inj_min, self.global_inj_max))

            # Current BHP P1, P2  (normalised with global pressure range)
            bhp_P1 = float(self.fine_data['bhpprd_ts_1'][ci, t])
            bhp_P2 = float(self.fine_data['bhpprd_ts_2'][ci, t])
            ch_bhp_P1 = self._broadcast(self._norm_scalar(
                bhp_P1, self.global_pres_min, self.global_pres_max))
            ch_bhp_P2 = self._broadcast(self._norm_scalar(
                bhp_P2, self.global_pres_min, self.global_pres_max))

            # ── 9-channel coarse input ────────────────────────────────────────
            coarse_inputs = np.stack([
                c_pres_n,    # ch0: coarse pressure
                c_sw_n,      # ch1: coarse saturation
                c_perm_n,    # ch2: permeability (log)
                ch_cum_I1,   # ch3: I1 cumulative injection
                ch_cum_I2,   # ch4: I2 cumulative injection
                ch_rate_I1,  # ch5: I1 current rate
                ch_rate_I2,  # ch6: I2 current rate
                ch_bhp_P1,   # ch7: P1 current BHP
                ch_bhp_P2,   # ch8: P2 current BHP
            ])

            # ── MBE physical fields (ReservoirSRDataset only) ────────────────
            # Static fields from cache; only Sw_cur and rates are time-varying
            t_mbe_prev  = max(t - 1, 0)
            Sw_cur_phys = self.fine_data['sw_grid'][ci, t_mbe_prev].squeeze(-1).astype(np.float32)
            poro_ref    = sc['poro_ref']
            inj_maps    = sc['inj_maps']
            prod_maps   = sc['prod_maps']

            rates_inj = np.array([
                float(self.fine_data['rateinj_ts_1'][ci, t]),
                float(self.fine_data['rateinj_ts_2'][ci, t]),
            ], dtype=np.float32)
            rates_wtr = np.array([
                float(self.fine_data['ratewtr_ts_1'][ci, t]),
                float(self.fine_data['ratewtr_ts_2'][ci, t]),
            ], dtype=np.float32)
            rates_oil = np.array([
                float(self.fine_data['rateoil_ts_1'][ci, t]),
                float(self.fine_data['rateoil_ts_2'][ci, t]),
            ], dtype=np.float32)

            return {
                'coarse_inputs':    torch.FloatTensor(coarse_inputs),
                'fine_target':      torch.FloatTensor(f_target_n),
                'fine_well_mask':   torch.FloatTensor(f_well_mask),
                'fine_perm':        torch.FloatTensor(f_perm_n).unsqueeze(0),
                'case_info':        info,
                'norm_params': {
                    'target_min': float(tlo),
                    'target_max': float(thi),
                },
                # MBE physical fields
                'Sw_cur_phys': torch.FloatTensor(Sw_cur_phys),  # (H, W)
                'poro_ref':    torch.FloatTensor(poro_ref),      # (H, W)
                'inj_maps':    inj_maps,                         # (N_inj, H, W)
                'prod_maps':   prod_maps,                        # (N_prod, H, W)
                'rates_inj':   torch.FloatTensor(rates_inj),    # (N_inj,)
                'rates_wtr':   torch.FloatTensor(rates_wtr),    # (N_prod,)
                'rates_oil':   torch.FloatTensor(rates_oil),    # (N_prod,)
            }

        except Exception as e:
            print(f'[Dataset] Error case {ci} t={t}: {e}')
            return {
                'coarse_inputs':    torch.zeros(9, *self.coarse_shape),
                'fine_target':      torch.zeros(*self.fine_shape),
                'fine_well_mask':   torch.zeros(*self.fine_shape),
                'fine_perm':        torch.zeros(1, *self.fine_shape),
                'case_info':        info,
                'norm_params':      {'target_min': 0.0, 'target_max': 1.0},
                'Sw_cur_phys':      torch.zeros(*self.fine_shape),
                'poro_ref':         torch.zeros(*self.fine_shape),
                'inj_maps':         torch.zeros(2, *self.fine_shape),
                'prod_maps':        torch.zeros(2, *self.fine_shape),
                'rates_inj':        torch.zeros(2),
                'rates_wtr':        torch.zeros(2),
                'rates_oil':        torch.zeros(2),
            }

    def __len__(self):
        return len(self.cases)
