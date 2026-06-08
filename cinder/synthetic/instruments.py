"""instruments.py - Layer B: map standardized structural states to instrument units (§5.1).

Takes the standardized structural trajectory from `latent.py`, maps each channel to native
instrument units via the §2 state-conditioned mean/SD, imposes the §3 ICC measurement-noise
floor, clips to valid ranges, and rounds to the C8 precision. RAPID3 is COMPUTED here from the
three OBSERVED scores: FN (observed HAQ-II) + PN (observed pain) + PtGA (observed global) -
never drawn independently (§4 construct rule; R2 closed). Every component carries measurement
noise, so RAPID3 has no noise-free component.

Per-instrument measurement noise is scaled to the patient's state true-SD so the emergent
test-retest ICC matches §3 regardless of disease-activity mixture:
    var_meas = var_true * (1 - ICC) / ICC   ->   sigma_meas = SD_true * sqrt((1-ICC)/ICC)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cinder.synthetic.latent import LatentTrajectory
from cinder.synthetic.params import MeasurementICC, StateBaseline

__all__ = [
    "InstrumentSeries",
    "WaveScores",
    "affine_true",
    "map_to_instruments",
    "noise_sd_for_icc",
]


def affine_true(struct: np.ndarray, mean: float, sd: float, lo: float, hi: float) -> np.ndarray:
    """Layer-B affine map of a standardized structural state to clipped native units.

    Shared by `map_to_instruments` and `flares` so the ground-truth crossings the flare
    planter computes use the exact same true values the instruments emit.
    """
    return np.clip(mean + sd * struct, lo, hi)


@dataclass(slots=True)
class WaveScores:
    """Observed, rounded instrument scores for a single wave."""

    haq_ii: float
    pain_vas: float
    pga_vas: float
    rapid3: float
    rapid3_fn: float
    rapid3_pn: float
    rapid3_ptga: float


@dataclass(slots=True)
class InstrumentSeries:
    """Per-wave observed scores plus the clean (pre-noise) true values for ground truth.

    The answer sheet's MCID arithmetic (Sprint 4) re-derives crossings from ``true_*`` so it
    is not fooled by measurement noise; the CSVs emit the ``obs`` (rounded) values.
    """

    waves: list[WaveScores]
    true_haq: np.ndarray
    true_pain: np.ndarray
    true_pga: np.ndarray


def noise_sd_for_icc(true_sd: float, icc: float) -> float:
    """Measurement-noise SD that yields the target test-retest ICC for a given true SD."""
    icc = min(max(icc, 1e-6), 1.0 - 1e-6)
    return float(true_sd) * float(np.sqrt((1.0 - icc) / icc))


def _fn_from_observed_haq(haq_obs: np.ndarray) -> np.ndarray:
    """RAPID3 FN (0-10), a calibrated monotone function of the OBSERVED HAQ-II.

    HAQ-II is 0-3; FN is 0-10. We scale the observed (measurement-noised) HAQ-II by 10/3 so FN
    co-moves with HAQ-II AND carries the same measurement noise as the other RAPID3 components
    (PN from observed Pain, PtGA from observed PGA). RAPID3 is therefore a pure function of the
    three observed instrument scores, as in the real MDHAQ-derived composite - no component is
    noise-free by construction. Calibrated synthetic proxy (cite Pincus 2008 for structure),
    NOT the exact MDHAQ transform (R2).
    """
    return np.clip(haq_obs * (10.0 / 3.0), 0.0, 10.0)


def map_to_instruments(
    rng: np.random.Generator,
    traj: LatentTrajectory,
    baseline: StateBaseline,
    icc: MeasurementICC,
) -> InstrumentSeries:
    """Map a structural trajectory to observed instrument scores for one patient."""
    n = traj.latent.shape[0]

    # --- Layer B affine map: standardized struct -> native units (clip to valid ranges). ---
    true_haq = affine_true(traj.haq_struct, baseline.haq_mean, baseline.haq_sd, 0.0, 3.0)
    true_pain = affine_true(traj.pain_struct, baseline.pain_mean, baseline.pain_sd, 0.0, 100.0)
    true_pga = affine_true(traj.pga_struct, baseline.pga_mean, baseline.pga_sd, 0.0, 100.0)

    # --- §3 ICC measurement noise (scaled to the patient's state true-SD). ---
    haq_obs = true_haq + rng.normal(0.0, noise_sd_for_icc(baseline.haq_sd, icc.haq), n)
    pain_obs = true_pain + rng.normal(0.0, noise_sd_for_icc(baseline.pain_sd, icc.pain), n)
    pga_obs = true_pga + rng.normal(0.0, noise_sd_for_icc(baseline.pga_sd, icc.pga), n)

    haq_obs = np.clip(haq_obs, 0.0, 3.0)
    pain_obs = np.clip(pain_obs, 0.0, 100.0)
    pga_obs = np.clip(pga_obs, 0.0, 100.0)

    # --- C8 precision: HAQ 1dp; Pain/PGA whole-unit; RAPID3 (+ subscores) 1dp. ---
    haq_r = np.round(haq_obs, 1)
    pain_r = np.round(pain_obs, 0)
    pga_r = np.round(pga_obs, 0)

    # --- RAPID3 computed (never drawn) from the three OBSERVED scores: FN(HAQ)+PN(pain)+PtGA(global).
    fn = np.round(_fn_from_observed_haq(haq_r), 1)
    pn = np.round(np.clip(pain_r / 10.0, 0.0, 10.0), 1)
    ptga = np.round(np.clip(pga_r / 10.0, 0.0, 10.0), 1)
    rapid3 = np.round(fn + pn + ptga, 1)

    waves = [
        WaveScores(
            haq_ii=float(haq_r[w]),
            pain_vas=float(pain_r[w]),
            pga_vas=float(pga_r[w]),
            rapid3=float(rapid3[w]),
            rapid3_fn=float(fn[w]),
            rapid3_pn=float(pn[w]),
            rapid3_ptga=float(ptga[w]),
        )
        for w in range(n)
    ]
    return InstrumentSeries(waves=waves, true_haq=true_haq, true_pain=true_pain, true_pga=true_pga)
