"""DTW barycenter averaging -- build a consensus signal profile from K reads.

The reference-free consensus path. `posterior.py` combines reads on a KNOWN
peptide's profile axis (the reference IS the alignment); DBA is what to do when no
such profile exists -- it CONSTRUCTS the shared axis from the reads themselves.
That is the peptide situation by default: no residue-kmer LUT, so no expected
profile to align to.

Relation to the other two fallbacks: DBA sits between `posterior.py` (needs a
reference) and `poa.py` (graph-structured, handles indels/variants). DBA assumes a
single linear consensus axis with only warping between reads -- cheaper and far
simpler than POA, and sufficient when the reads are repeat observations of the same
molecule differing by stepping rate, not by content.

Algorithm
---------
    1. init: pairwise DTW (`align.dtw.dtw_pairwise`) over the K reads; barycenter B
       = the MEDOID read (argmin row-sum of the distance matrix). Medoid init, not
       a random read: DBA is a local optimizer (Expectation-Maximization-like, no
       global guarantee) and the medoid is the cheapest defensible starting point.
    2. iterate to convergence (typically 5-10 rounds):
         assign : DTW-align every read r to B -> warping path_r
         update : for each barycenter position j, gather all read steps mapped to j
                  and replace B[j] with their INVERSE-VARIANCE weighted mean
                  (weight 1/std**2 -- reuses the same per-step uncertainty the HMM
                  emission uses; a plain mean throws it away).
       Stop when max |B_new - B_old| < tol, or on the iteration cap.
    3. output: consensus profile + per-position spread + per-position depth.

Cost must be squared euclidean
------------------------------
Use `align.cost.cost_l2` INSIDE the loop, not `cost_gaussian`. DBA's update step
(replace each position by the mean of its assigned observations) is a descent step
only because the arithmetic mean is the Frechet mean under squared euclidean. Under
the Gaussian/emission cost that identity does not hold, the update stops being a
descent step, and the iteration can oscillate instead of converge. The
inverse-variance weighting above is a weighted mean -- still the Frechet mean of
the same metric under those weights, so it is safe; changing the METRIC is not.
(`cost_gaussian` is still the right choice for the step-1 pairwise matrix, which is
a distance computation, not part of the averaging loop.)

Channels stay separate
----------------------
Warp is computed on the LEVEL channel (step mean) only. `dwell` is then carried
through the same path and averaged into its own consensus track. Concatenating
level and dwell into one cost mixes units and lets stepping-rate variation -- the
exact thing DTW is supposed to absorb -- distort the level alignment. Per-step
`std` likewise rides along as a weight, never as a cost dimension.

Partial overlap
---------------
The staged default is CLOSED DTW (`open_begin=False`, `open_end=False`) while
testing fragments whose DNA boundary should make both ends comparable. Every read
therefore consumes the full medoid axis. Once partial-span behavior is being tested,
callers can explicitly enable either open end. Open ends produce unequal depth at
the barycenter edges, which is why depth is always tracked and emitted.

What it yields beyond the profile
---------------------------------
- **Per-position confidence** = weighted spread of the assigned observations,
  Component 6's per-position confidence deliverable, straight out of the update step.
- **Steps per residue**, measurable from the converged barycenter's level-transition
  spacing. Component 6 explicitly refuses to inherit DNA's "two steps per
  nucleotide"; this is the measurement that replaces the assumption.

Public API
----------
    dba(reads, *, medoid_index=None, max_iter=10, tol=..., min_depth=..., **dtw_kw)
        -> Barycenter(mean, std, dwell, depth, n_iter, converged, delta)
        reads : per-read step arrays (mean, std, dwell), pre-normalized to a common
                scale, or pass normalize=True.

Validation order (do NOT go straight to peptide)
------------------------------------------------
Validate on DNA first, where ground truth exists: DBA a group of reads of one known
molecule, compare the converged barycenter against that molecule's LUT profile from
`io`. If the barycenter does not reproduce a known profile, nothing it produces on
peptide data can be trusted. Only then port to peptide chemistry. Same milestone
shape as the rest of Component 6: consensus accuracy must rise with depth K and the
per-position confidence must track true error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from ..align.cost import cost_l2
from ..align.dtw import dtw, dtw_pairwise
from ._signal import prepare_reads, validate_medoid_index
from .cluster import SignalCluster, medoid

FloatArr = npt.NDArray[np.float64]
Int64Arr = npt.NDArray[np.int64]
BoolArr = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Barycenter:
    """A linear signal consensus on the selected medoid's step axis."""

    mean: FloatArr
    std: FloatArr
    dwell: FloatArr
    depth: Int64Arr
    supported: BoolArr
    n_iter: int
    converged: bool
    medoid_index: int
    objective: FloatArr
    delta: FloatArr


def dba(
    reads: Sequence[Any],
    *,
    medoid_index: int | None = None,
    distance_matrix: FloatArr | None = None,
    normalize: bool = True,
    max_iter: int = 10,
    tol: float = 1e-4,
    min_depth: int = 2,
    min_std: float = 1e-3,
    **dtw_kw: Any,
) -> Barycenter:
    """Fit a DTW barycenter initialized by a shared clustering-stage medoid.

    The returned profile remains on the medoid's axis. ``std`` combines weighted
    between-read spread with mean-estimation uncertainty; ``depth`` counts
    distinct covering reads.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol < 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and non-negative")
    if min_depth < 1:
        raise ValueError("min_depth must be >= 1")
    if min_std <= 0.0 or not np.isfinite(min_std):
        raise ValueError("min_std must be finite and positive")

    prepared = prepare_reads(reads, normalize=normalize, min_std=min_std)
    if medoid_index is None:
        if distance_matrix is None:
            pairwise_input = [(r.mean, r.std) for r in prepared]
            distance_matrix = dtw_pairwise(
                pairwise_input, normalize=False, **dtw_kw
            )
        elif np.asarray(distance_matrix).shape != (len(prepared), len(prepared)):
            raise ValueError("distance_matrix shape must match the number of reads")
        medoid_index = medoid(distance_matrix)
    centre = validate_medoid_index(medoid_index, len(prepared))

    initial = prepared[centre]
    mean = initial.mean.copy()
    std = initial.std.copy()
    dwell = initial.dwell.copy()
    length = mean.size
    objective: list[float] = []
    delta_history: list[float] = []
    converged = False
    depth = np.ones(length, dtype=np.int64)
    iteration = 0

    options: dict[str, Any] = {
        "open_begin": False,
        "open_end": False,
        "step": "symmetric1",
        "max_run": 3,
    }
    options.update(dtw_kw)

    for iteration in range(1, max_iter + 1):
        sum_w = np.zeros(length, dtype=float)
        sum_wx = np.zeros(length, dtype=float)
        sum_wx2 = np.zeros(length, dtype=float)
        sum_wdwell = np.zeros(length, dtype=float)
        depth = np.zeros(length, dtype=np.int64)
        round_objective = 0.0

        for read in prepared:
            observation_precision = 1.0 / np.maximum(
                read.std ** 2, min_std ** 2
            )
            local_cost = cost_l2(read.mean, mean) * observation_precision[:, None]
            alignment = dtw(local_cost, **options)
            round_objective += alignment.norm_dist
            obs_idx = alignment.path[:, 0]
            target_idx = alignment.path[:, 1]
            weight = observation_precision[obs_idx]

            np.add.at(sum_w, target_idx, weight)
            np.add.at(sum_wx, target_idx, weight * read.mean[obs_idx])
            np.add.at(sum_wx2, target_idx, weight * read.mean[obs_idx] ** 2)
            np.add.at(sum_wdwell, target_idx, weight * read.dwell[obs_idx])
            depth[np.unique(target_idx)] += 1

        updated = sum_w > 0.0
        new_mean = mean.copy()
        new_std = std.copy()
        new_dwell = dwell.copy()
        new_mean[updated] = sum_wx[updated] / sum_w[updated]
        new_dwell[updated] = sum_wdwell[updated] / sum_w[updated]
        variance = np.zeros(length, dtype=float)
        variance[updated] = np.maximum(
            sum_wx2[updated] / sum_w[updated] - new_mean[updated] ** 2, 0.0
        )
        new_std[updated] = np.sqrt(
            variance[updated] + 1.0 / sum_w[updated]
        )
        new_std = np.maximum(new_std, min_std)

        delta = float(np.max(np.abs(new_mean - mean)))
        mean, std, dwell = new_mean, new_std, new_dwell
        objective.append(round_objective)
        delta_history.append(delta)
        if delta <= tol:
            converged = True
            break

    return Barycenter(
        mean=mean,
        std=std,
        dwell=dwell,
        depth=depth,
        supported=depth >= min_depth,
        n_iter=iteration,
        converged=converged,
        medoid_index=centre,
        objective=np.asarray(objective, dtype=float),
        delta=np.asarray(delta_history, dtype=float),
    )


def dba_cluster(
    reads: Sequence[Any],
    cluster: SignalCluster,
    **kwargs: Any,
) -> Barycenter:
    """Run DBA on one shared clustering result without manual index remapping."""
    selected, local_medoid = cluster.select(reads)
    return dba(selected, medoid_index=local_medoid, **kwargs)
