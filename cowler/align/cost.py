"""Local cost matrices shared by the HMM aligner (`dp.py`) and DTW (`dtw.py`).

Both aligners are dynamic programs over the SAME object: a local cost/score grid
`C[Na, Nb]` scoring observation `i` against target `j`. They differ only in the
recursion applied to it (probabilistic transitions + logsumexp vs fixed step
patterns + min). Factoring the grid out here keeps that shared and makes the cost
choice an explicit parameter instead of an assumption baked into each aligner.

`dp.emission` is already generic in exactly this way -- it takes `(xm, xs, Em, Es)`
and knows nothing about molecules, profiles, or the `L = 2M` half-step axis. So the
same call serves both the reference case (target = a known molecule's half-step
profile) and the pairwise case (target = ANOTHER READ's steps). This module is the
thin, named layer over that.

Two target kinds
----------------
  reference : target is a molecule/peptide profile from `io` (Em, Es per half-step).
              Es = reference level spread. Used by `dp.viterbi` / `forward_backward`.
  pairwise  : target is a second read's steps (bm, bsd). Now BOTH sides are noisy
              observations; `v = xs**2 + bsd**2` reads as "combined observation
              noise" rather than "step + reference uncertainty". Used by
              `dtw.dtw_pairwise` for reference-free grouping + fragment analysis.
The math is identical; only the interpretation of the second (mean, std) pair
changes. No separate code path.

Cost choice (this is a real decision, not a style preference)
------------------------------------------------------------
cost_gaussian -- negated `dp.emission`. Folds both uncertainties into
    `v = xs**2 + Es**2`, so noisy / short steps are automatically down-weighted --
    the same property that makes the HMM emission work. DEFAULT for pairwise
    distance and fragment-local alignment.
cost_l2 -- plain squared euclidean on means, stds ignored. Required inside DBA
    (`consensus/dba.py`): barycenter averaging is only provably a barycenter under
    squared euclidean, because the arithmetic mean is that metric's Frechet mean.
    Gaussian cost breaks the guarantee and the DBA update stops being a descent
    step. Also the honest choice on iteration 1, where the barycenter has no
    meaningful std yet.
cost_abs -- |mean difference|. Cheapest; a baseline / debug cost, and what most
    published DTW squiggle aligners actually use. Keep it to quantify what the
    uncertainty-weighted costs buy.

Normalization precondition
--------------------------
For PAIRWISE costs both reads must be on a common scale first
(`io/normalize.py:normalize_signal`, or `to_lut_units`). Otherwise the distance
measures per-read scale/offset drift, not signal shape -- two reads of the same
molecule with different open-pore levels look maximally distant. The reference
case already normalizes to LUT units upstream, so this bites only the new pairwise
path. Callers must not rely on the cost builders to do it.

Sign convention
---------------
`dp` maximizes a SCORE (log-likelihood); DTW minimizes a COST. Everything here
returns a COST (lower = better match): `cost_gaussian = -emission`. `dp` callers
negate back, or keep using `dp.emission` directly.

Public API
----------
    cost_gaussian(am, asd, bm, bsd) -> C[Na, Nb]   # -emission, both-sided noise
    cost_l2(am, bm)                 -> C[Na, Nb]   # (am_i - bm_j)**2   [DBA]
    cost_abs(am, bm)                -> C[Na, Nb]   # |am_i - bm_j|      [baseline]

All return float64 `C[Na, Nb]`, NaN-free (caller drops NaN steps first, as the
profile builder already does for boundary half-steps).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .dp import emission

FloatArr = npt.NDArray[np.float64]


def cost_gaussian(am: FloatArr, asd: FloatArr, bm: FloatArr, bsd: FloatArr) -> FloatArr:
    """Uncertainty-weighted cost `C[Na, Nb] = -emission(am, asd, bm, bsd)`.

    Both sides may be noisy observations (pairwise) or the second may be a
    reference profile; the math does not care. Lower = better match.
    """
    return -emission(am, asd, bm, bsd)


def cost_l2(am: FloatArr, bm: FloatArr) -> FloatArr:
    """Squared euclidean on means, `C[i, j] = (am[i] - bm[j])**2`.  [DBA]"""
    a = np.asarray(am, float)[:, None]
    b = np.asarray(bm, float)[None, :]
    return (a - b) ** 2


def cost_abs(am: FloatArr, bm: FloatArr) -> FloatArr:
    """Absolute level difference, `C[i, j] = |am[i] - bm[j]|`.  [baseline]"""
    a = np.asarray(am, float)[:, None]
    b = np.asarray(bm, float)[None, :]
    return np.abs(a - b)
