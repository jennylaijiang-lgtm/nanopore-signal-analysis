"""Signal -> pA conversion and normalization to the LUT level distribution.

1. To pA:  pA = (raw + offset) * rng / digitisation
2. Scale to LUT:
   - Initial (robust): shift = median(pA), scale = 1.4826 * MAD(pA);
     x[t] = (pA[t] - shift) / scale. Normalize LUT the same way, pooling ALL
     pre and post means when computing the LUT median/MAD.
   - Refine (short reads): after first alignment, regress observed x against
     expected level along the path to re-estimate shift/scale, then re-align
     once or twice. Converges because the input molecule is known.

Public API:
    to_pa(read)                       -- raw ADC -> pA
    robust_params(levels)             -- (shift, scale) = (median, 1.4826*MAD)
    normalize_signal(pa)              -- raw pA signal -> robust z-scored signal
    lut_params(lut)                   -- robust (shift, scale) over pooled pre+post means
    to_lut_units(xm, xs, lut)         -- map observed levels onto the LUT scale
    refine_scaling(x, path, profile)  -- regression-based shift/scale refinement
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from .lut import get_default_lut


def robust_params(levels: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Robust center/scale: shift = median, scale = 1.4826 * MAD."""
    levels = np.asarray(levels, float)
    shift = float(np.median(levels))
    scale = float(1.4826 * np.median(np.abs(levels - shift)))
    return shift, scale


def normalize_signal(
    pa: npt.NDArray[np.float64],
    shift: float | None = None,
    scale: float | None = None,
) -> npt.NDArray[np.float64]:
    """Robust z-score a raw pA signal so it sits on a scale-free level axis.

    `x[t] = (pa[t] - shift) / scale`, with `(shift, scale) = robust_params(pa)`
    by default (per-read median / 1.4826*MAD). This is the normalization the
    step finder needs before `find_steps`: its CPIC variance floor is
    scale-dependent, so raw LUT-scale pA (tiny within-level variance) is
    undersegmented while a robust z-scored signal is not.

    Pass explicit `shift`/`scale` to reuse a defined open-state normalization
    (e.g. the poreFlow IOS open-pore level) instead of per-read median/MAD.
    NaNs are ignored when estimating the params.
    """
    pa = np.asarray(pa, float)
    if shift is None or scale is None:
        s, c = robust_params(pa[~np.isnan(pa)])
        shift = s if shift is None else shift
        scale = c if scale is None else scale
    if scale == 0.0:
        scale = 1.0
    return (pa - shift) / scale


def lut_params(lut: pd.DataFrame | None = None) -> tuple[float, float]:
    """Robust (shift, scale) of the LUT, pooling ALL pre + post means."""
    lut = get_default_lut() if lut is None else lut
    pooled = np.concatenate([
        lut["pre_mean"].to_numpy(float), lut["post_mean"].to_numpy(float),
    ])
    return robust_params(pooled)


def to_lut_units(
    xm: npt.NDArray[np.float64],
    xs: npt.NDArray[np.float64],
    lut: pd.DataFrame | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map observed step mean/std onto the LUT level scale (robust, per read).

    z-score the observed means, then re-express in LUT units so emission can be
    compared directly against LUT levels. stds are rescaled by the same factor.
    On data already in LUT units this is ~identity.
    """
    xm = np.asarray(xm, float)
    xs = np.asarray(xs, float)
    o_shift, o_scale = robust_params(xm)
    l_shift, l_scale = lut_params(lut)
    factor = l_scale / o_scale
    return (xm - o_shift) * factor + l_shift, xs * factor


# TODO: implement to_pa(read) and refine_scaling once the Read dataclass + a
# Component 1 alignment path are available (not needed for already-pA synthetic).
