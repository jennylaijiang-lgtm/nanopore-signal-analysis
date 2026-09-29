"""Dynamic time warping over observed steps -- pairwise + fragment-local alignment.

DTW is the second aligner in this repo. It is NOT a replacement for `dp.viterbi`;
the two answer different questions and the boundary between them is sharp:

    dp.viterbi   observation vs a KNOWN molecule profile. Has an emission model, a
                 calibrated physical transition matrix (pstep/phold/pmiss), free
                 ends, and returns (s, e) + logL. Strictly more informed.
    dtw          observation vs ANOTHER OBSERVATION, or vs a fragment, when no
                 profile / likelihood model exists. Returns a warping path and a
                 distance -- no (s,e), no likelihood.

Why not use DTW for Component 1
-------------------------------
`dp.viterbi` already IS a DTW, with better parts: the local cost carries step
uncertainty (`v = xs**2 + Es**2`) instead of a bare level difference, and the step
pattern is a calibrated `logT` instead of hand-weighted {(0,1),(1,1),(1,0)} moves
-- including real back-steps, which monotone DTW cannot express at all. Swapping
DTW in for a known molecule throws both away. (Tools like uncalled4 use banded DTW
because the ONT stack has no per-read likelihood model and needs speed on long
reads; cowler reads are <300 nt and full O(N*L) is already cheap.) Keep DTW as an
`eval/` baseline for exactly this comparison -- it quantifies what the physical
transition model buys.

Where DTW earns its place
-------------------------
1. **Fragment-local analysis.** Compare a short defined signal fragment against a
   read or another fragment. Ends are DEFINED here, so this is the closed
   (corner-to-corner) case -- see the ends section below.
2. **Reference-free grouping.** Pairwise distance matrix over reads -> clustering
   -> same-molecule groups WITHOUT a profile. This is the peptide path
   (`consensus/group.py`): no residue-kmer LUT exists, so there is no emission
   model and no logL, so likelihood demux is impossible. For DNA, keep
   `forward_backward` logL for demux; use DTW distance only to cross-check it.
3. **DBA input.** The alignment step of barycenter averaging (`consensus/dba.py`).

Ends: closed by default
-----------------------
Vanilla DTW forces corner-to-corner (both sequences fully consumed). That is the
RIGHT default here -- the intended use is defined fragments, where both endpoints
are known and forcing them is information, not a constraint to escape.

The open variants exist for the sub-span case (fact 4: reads cover a contiguous
sub-span with distributed start/end), and are configured entirely in two lines of
the recursion:
    open_begin : `D[0] = C[0]` instead of `cumsum(C[0])`  (any target start free)
    open_end   : take `argmin(D[-1])` instead of `D[-1, -1]`  (any target end free)
These are the same free ends `dp.viterbi` implements as uniform init + max over
the final row. If a caller wants (s,e)-like output from DTW, both flags must be
on; a closed DTW forced onto a sub-span read produces garbage endpoints, not a
degraded estimate.

Step pattern
------------
Default `symmetric1`: moves {(1,1) diagonal, (1,0), (0,1)} at unit weight.
`symmetric2` is the same move set with the diagonal weighted 2 -- self-normalizing
(path weights sum to Na + Nb), so `norm_dist` is exact rather than a path-length
approximation. NEITHER pattern bounds run length on its own: consecutive singleton
moves are unbounded, so one step can absorb arbitrarily many partners
(pathological warping). On short fragments a single bad step is a large fraction of
the sequence, so bound it with `max_run` (cap on consecutive same-direction
singleton moves, default 3; `None` = unbounded). This is the DTW-side analogue of
`phold`/`pmiss` -- a bound on how far the warp may depart from 1:1 -- except it is a
hard constraint rather than a prior.

Distance normalization
----------------------
Raw `D[-1, j_end]` scales with path length, so longer fragments always look worse
and distances are NOT comparable across pairs. Always report and cluster on
`norm_dist` (divide by path length for symmetric1; by `Na + Nb` for symmetric2,
which is exact rather than approximate). `dtw_pairwise` returns norm_dist only.

DTW distance is NOT a likelihood
--------------------------------
It has no probabilistic scale, is not comparable across different target lengths
even after normalization, and cannot be summed or turned into a ratio. Never feed
it to Component 2's variant LLR or use it for DNA read->molecule assignment --
`FBResult.logL` owns that. DTW distance is for clustering and for consensus
alignment only.

Public API
----------
    dtw(C, *, open_begin=False, open_end=False, step="symmetric1", max_run=3)
        -> DTWResult(path, dist, norm_dist, j_start, j_end)
        `C` from `align/cost.py`. `path` is an (n_moves, 2) array of (i, j) index
        pairs. `j_start`/`j_end` are the matched target span (== 0, Nb-1 when
        closed; meaningful only with the open flags).

    dtw_pairwise(reads, *, cost=cost_gaussian, normalize=True, **dtw_kw)
        -> D[K, K] symmetric norm_dist matrix, zero diagonal.
        Feeds `scipy.cluster.hierarchy.linkage` for grouping and supplies the
        medoid (argmin row-sum) that initializes DBA.

    warp_to(path, values, n_target) -> values resampled onto the target axis
        Push a per-step quantity (mean, std, dwell, a label) through a warping path
        onto the target index axis. The primitive DBA's averaging step needs, and
        what makes two reads comparable position-by-position.

Notes
-----
- Reads are short; no banding (same reasoning as `dp`). Add a Sakoe-Chiba band only
  if fragment lengths ever reach thousands of steps.
- Inner loop is the same shape as `dp.viterbi`'s and gets `numba @njit` at the same
  time (deferred for both).
- Z-normalize per read before pairwise costs -- see the precondition in
  `align/cost.py`. `dtw_pairwise(normalize=True)` does it; direct `dtw` callers own it.
- `open_end` takes `argmin` of the RAW last row, the standard convention. With a
  cost that can go negative (`cost_gaussian` is a log density) that mildly favours
  longer matched spans; with a non-negative cost it favours shorter ones. Closed
  ends -- the default -- have no such bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore[import-untyped]

from .cost import cost_gaussian
from ..io.normalize import robust_params

FloatArr = npt.NDArray[np.float64]

_INF = 1e300
_DIAG, _VERT, _HORIZ = 0, 1, 2
_WEIGHT = {"symmetric1": 1.0, "symmetric2": 2.0}  # diagonal weight


class NoAdmissiblePathError(ValueError):
    """Raised when the declared warp constraint permits no complete DTW path."""


@dataclass
class DTWResult:
    path: npt.NDArray[np.int_]  # (n_moves, 2) array of (i, j) index pairs
    dist: float                 # raw accumulated cost
    norm_dist: float            # dist / normalizer (path length or Na + span)
    j_start: int                # first matched target index (0 unless open_begin)
    j_end: int                  # last matched target index (Nb-1 unless open_end)


@njit(cache=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def _dtw_core(
    C: FloatArr, w_diag: float, open_begin: bool, R: int
) -> tuple[FloatArr, npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Accumulated cost + backpointers.

    State axis tracks singleton-run length so `max_run` can be enforced:
    state 0 = arrived diagonally (or at the start), 1..R = arrived by a run of
    that many (1,0) moves, R+1..2R = same for (0,1). R == 0 disables the cap
    (single state, unbounded runs).
    """
    Na, Nb = C.shape
    nS = 1 if R == 0 else 1 + 2 * R
    D = np.full((Na, Nb, nS), _INF)
    bpm = np.full((Na, Nb, nS), np.int8(-1))  # move taken into this cell
    bps = np.full((Na, Nb, nS), np.int8(-1))  # predecessor state

    for i in range(Na):
        for j in range(Nb):
            if i == 0 and (j == 0 or open_begin):
                if D[i, j, 0] > w_diag * C[i, j]:
                    D[i, j, 0] = w_diag * C[i, j]
                    bpm[i, j, 0] = np.int8(-1)
                continue
            # diagonal -> state 0
            if i > 0 and j > 0:
                best = _INF
                bs = -1
                for s in range(nS):
                    if D[i - 1, j - 1, s] < best:
                        best = D[i - 1, j - 1, s]
                        bs = s
                if best < _INF:
                    cand = best + w_diag * C[i, j]
                    if cand < D[i, j, 0]:
                        D[i, j, 0] = cand
                        bpm[i, j, 0] = np.int8(_DIAG)
                        bps[i, j, 0] = np.int8(bs)
            # singleton moves: vertical (i-1, j) and horizontal (i, j-1)
            for mv in range(2):
                if mv == 0:
                    if i == 0:
                        continue
                    pi, pj, base = i - 1, j, 0
                else:
                    if j == 0:
                        continue
                    pi, pj, base = i, j - 1, R
                for s in range(nS):
                    prev = D[pi, pj, s]
                    if prev >= _INF:
                        continue
                    if R == 0:
                        ns = 0
                    else:
                        run = s - base  # >0 iff same-direction run
                        if 1 <= run <= R:
                            if run == R:
                                continue  # cap reached
                            ns = base + run + 1
                        else:
                            ns = base + 1
                    cand = prev + C[i, j]
                    if cand < D[i, j, ns]:
                        D[i, j, ns] = cand
                        bpm[i, j, ns] = np.int8(_VERT if mv == 0 else _HORIZ)
                        bps[i, j, ns] = np.int8(s)
    return D, bpm, bps


def dtw(
    C: FloatArr,
    *,
    open_begin: bool = False,
    open_end: bool = False,
    step: str = "symmetric1",
    max_run: int | None = 3,
) -> DTWResult:
    """Warp observation axis `i` (fully consumed) onto target axis `j`.

    `C` is a local cost grid from `align/cost.py` (lower = better match). Ends on
    the target axis are closed unless `open_begin` / `open_end` (see module doc).
    """
    costs = np.ascontiguousarray(np.asarray(C, float))
    if costs.ndim != 2 or costs.size == 0:
        raise ValueError("C must be a non-empty 2-D cost matrix")
    if not np.all(np.isfinite(costs)):
        raise ValueError("C contains NaN/inf -- drop NaN steps before costing")
    if step not in _WEIGHT:
        raise ValueError(f"unknown step pattern {step!r}")
    R = 0 if max_run is None else int(max_run)
    if R < 0:
        raise ValueError("max_run must be >= 1 or None")

    Na, Nb = costs.shape
    D, bpm, bps = _dtw_core(costs, _WEIGHT[step], open_begin, R)

    last = D[Na - 1]                                  # [Nb, nS]
    j_end = int(np.argmin(last.min(axis=1))) if open_end else Nb - 1
    s = int(np.argmin(last[j_end]))
    dist = float(last[j_end, s])
    if dist >= _INF:
        raise NoAdmissiblePathError(
            "no admissible warping path -- max_run too tight for these lengths"
        )

    i, j = Na - 1, j_end
    pts = [(i, j)]
    while bpm[i, j, s] >= 0:
        mv, ps = int(bpm[i, j, s]), int(bps[i, j, s])
        if mv == _DIAG:
            i, j = i - 1, j - 1
        elif mv == _VERT:
            i = i - 1
        else:
            j = j - 1
        s = ps
        pts.append((i, j))
    path = np.array(pts[::-1], dtype=np.int_)

    j_start = int(path[0, 1])
    span = j_end - j_start + 1
    denom = float(Na + span) if step == "symmetric2" else float(len(path))
    return DTWResult(path=path, dist=dist, norm_dist=dist / denom,
                     j_start=j_start, j_end=j_end)


def _steps(read: Any) -> tuple[FloatArr, FloatArr]:
    """(mean, std) from a StepRead-like object or (mean, std[, dwell]) tuple."""
    if hasattr(read, "mean") and hasattr(read, "std"):
        return np.asarray(read.mean, float), np.asarray(read.std, float)
    values = tuple(read)
    if len(values) not in (2, 3):
        raise ValueError("a read tuple must contain (mean, std[, dwell])")
    m, sd = values[:2]
    return np.asarray(m, float), np.asarray(sd, float)


def _znorm(m: FloatArr, sd: FloatArr) -> tuple[FloatArr, FloatArr]:
    shift, scale = robust_params(m[~np.isnan(m)])
    if scale == 0.0:
        scale = 1.0
    return (m - shift) / scale, sd / scale


def dtw_pairwise(
    reads: Sequence[Any],
    *,
    cost: Callable[..., FloatArr] = cost_gaussian,
    normalize: bool = True,
    **dtw_kw: Any,
) -> FloatArr:
    """Symmetric `norm_dist` matrix `D[K, K]` (zero diagonal) over K reads.

    Feeds `scipy.cluster.hierarchy.linkage` for grouping and supplies the medoid
    (`argmin` row-sum) that initializes DBA. `normalize=True` robust z-scores each
    read first -- the pairwise precondition in `align/cost.py`.
    """
    steps = [_steps(r) for r in reads]
    if normalize:
        steps = [_znorm(m, sd) for m, sd in steps]
    takes_std = cost is cost_gaussian
    K = len(steps)
    D = np.zeros((K, K), float)
    for a in range(K):
        for b in range(a + 1, K):
            am, asd = steps[a]
            bm, bsd = steps[b]
            C = cost(am, asd, bm, bsd) if takes_std else cost(am, bm)
            d = dtw(C, **dtw_kw).norm_dist
            D[a, b] = D[b, a] = d
    return D


def warp_to(
    path: npt.NDArray[np.int_], values: FloatArr, n_target: int
) -> FloatArr:
    """Average `values` (one per observation step) onto the target index axis.

    Target positions the path never visits are NaN -- open ends make that the
    normal case at the edges, and DBA must not average them in silently.
    """
    values = np.asarray(values, float)
    acc = np.zeros(n_target)
    cnt = np.zeros(n_target)
    for i, j in path:
        acc[j] += values[i]
        cnt[j] += 1
    out = np.full(n_target, np.nan)
    nz = cnt > 0
    out[nz] = acc[nz] / cnt[nz]
    return out
