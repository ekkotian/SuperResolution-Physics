"""
mbe_physics.py
==============
Global material balance verification for the 2-phase (oil-water)
channelised reservoir SR project.

Physical setup (Table 1 in paper)
----------------------------------
Domain      : 10,000 × 10,000 × 10  ft³
Fine grid   : 100 × 100  →  dx = dy = 100 ft,  dz = 10 ft
Coarse grid :  20 ×  20  →  dx = dy = 500 ft,  dz = 10 ft
Porosity    : from poro_grid per case (≈ 0.15 constant)
ρ_w = 62.2 lbm/ft³  (constant)
ρ_o = 46.2 lbm/ft³  (constant)
dt  = 365 days per timestep

Global balance equation (water phase)
--------------------------------------
  ΔStorage  =  Injection  −  Production

  ΔStorage  = Σ_cells  φ · (Sw_next − Sw_cur) · ρ_w · V_cell    [lbm]
  Injection = Σ_wells  rate_inj · COEF_Q · ρ_w                   [lbm]
  Production= Σ_wells  rate_wtr · COEF_Q · ρ_w                   [lbm]

  Residual  = |ΔStorage − (Injection − Production)|
  Relative  = Residual / (Injection + ε)                          [fraction]

This formulation avoids all numerical differentiation and transmissibility
calculations, so it is insensitive to time-step size and discretisation
scheme differences between our code and the simulator.
"""

import torch
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

FINE_DX   = 100.0   # ft
FINE_DY   = 100.0   # ft
FINE_DZ   = 10.0    # ft
COARSE_DX = 500.0   # ft
COARSE_DY = 500.0   # ft
COARSE_DZ = 10.0    # ft

RHO_W  = 62.2    # lbm/ft³
RHO_O  = 46.2    # lbm/ft³
COEF_Q = 5.614583  # ft³/STB


# ── Global balance function ───────────────────────────────────────────────────

def global_balance(
        Sw_next:   torch.Tensor,   # (B, H, W)  Sw at t+1  (predicted OR true)
        Sw_cur:    torch.Tensor,   # (B, H, W)  Sw at t    (always true)
        poro:      torch.Tensor,   # (B, H, W)  porosity (reference, constant ok)
        inj_maps:  torch.Tensor,   # (B, N_inj,  H, W)  binary per injector
        prod_maps: torch.Tensor,   # (B, N_prod, H, W)  binary per producer
        rates_inj: torch.Tensor,   # (B, N_inj)   water injection rate, STB/day
        rates_wtr: torch.Tensor,   # (B, N_prod)  water production rate, STB/day
        rates_oil: torch.Tensor,   # (B, N_prod)  oil production rate,   STB/day
        dt:        float = 365.0,  # timestep length, days
        grid: str = 'fine',
):
    """
    Compute global (field-scale) material balance error for water and oil.

    Parameters
    ----------
    Sw_next : predicted or true Sw at t+1 — swap this to compare scenarios
    Sw_cur  : always use true Sw at t (the starting state is known)
    rates_* : total volumes per timestep from dataset  [STB]

    Returns
    -------
    water_rel : (B,)  |ΔStorage_w − NetInjection_w| / Injection_w   [fraction]
    oil_rel   : (B,)  |ΔStorage_o − NetProduction_o| / Production_o [fraction]

    Interpretation
    --------------
    0.01  →  1%  imbalance  (excellent)
    0.05  →  5%  imbalance  (acceptable)
    0.10  →  10% imbalance  (marginal)
    >0.20 →  poor physical consistency
    """
    if grid == 'fine':
        dx, dy, dz = FINE_DX, FINE_DY, FINE_DZ
    else:
        dx, dy, dz = COARSE_DX, COARSE_DY, COARSE_DZ

    V_cell = dx * dy * dz   # ft³ per grid cell

    # ── Storage change  [lbm] ─────────────────────────────────────────────────
    # Sum over all cells: φ × ΔSw × ρ × V
    delta_Sw  = Sw_next - Sw_cur                               # (B, H, W)
    delta_w   = (poro * delta_Sw          * RHO_W * V_cell).sum(dim=(1, 2))  # (B,)
    delta_o   = (poro * (-delta_Sw)       * RHO_O * V_cell).sum(dim=(1, 2))  # (B,)
    # Note: Δ(1-Sw) = -ΔSw, so oil storage change is just -delta_w scaled by ρ_o/ρ_w

    # ── Well volumes  [lbm] ───────────────────────────────────────────────────
    # rates_* are STB/day (daily rates).
    # Total volume per timestep = rate × dt days × COEF_Q ft³/STB × ρ lbm/ft³
    total_inj_w  = (rates_inj.sum(dim=1)) * dt * COEF_Q * RHO_W    # (B,)
    total_prod_w = (rates_wtr.sum(dim=1)) * dt * COEF_Q * RHO_W    # (B,)
    total_prod_o = (rates_oil.sum(dim=1)) * dt * COEF_Q * RHO_O    # (B,)

    # ── Residuals  [lbm] ─────────────────────────────────────────────────────
    # Water: what went in minus what came out should equal storage change
    res_w = delta_w - (total_inj_w - total_prod_w)
    # Oil: what was produced should equal reduction in storage
    res_o = delta_o + total_prod_o   # delta_o is negative (oil leaving), prod_o positive

    # ── Relative error  [fraction] ────────────────────────────────────────────
    eps = 1.0   # lbm, prevents div-by-zero
    water_rel = res_w.abs() / (total_inj_w.abs() + eps)
    oil_rel   = res_o.abs() / (total_prod_o.abs() + eps)

    return water_rel, oil_rel


# ── Build per-well binary maps ────────────────────────────────────────────────

def build_well_maps(inj_uba: np.ndarray, prd_uba: np.ndarray,
                    H: int, W: int, device=None):
    """
    inj_uba / prd_uba : (N_wells, 2)  1-indexed (x=col, y=row)
    Returns inj_maps (N_inj, H, W), prod_maps (N_prod, H, W).
    """
    def _make(uba):
        maps = []
        for (x, y) in uba:
            m = np.zeros((H, W), dtype=np.float32)
            r, c = int(y) - 1, int(x) - 1
            if 0 <= r < H and 0 <= c < W:
                m[r, c] = 1.0
            maps.append(m)
        return torch.tensor(np.stack(maps))

    it = _make(inj_uba)
    pt = _make(prd_uba)
    if device:
        it, pt = it.to(device), pt.to(device)
    return it, pt
