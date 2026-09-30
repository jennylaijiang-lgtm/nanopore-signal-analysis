"""Estimate a consensus signal with uncertainty-weighted DTW barycentre averaging.

Reads are repeated observations of the same signal with different stepping rates.
The output is a shared signal profile, not an amino-acid sequence.

Algorithm
---------
1. Initialise the profile from a medoid: the read with the smallest sum of
   pairwise DTW distances, unless an explicit medoid index is supplied.
2. Align each read to the current profile using squared-error DTW costs weighted
   by the inverse observation variance.
3. At each profile position, replace the mean with the inverse-variance weighted
   mean of aligned observations. Repeat until the maximum absolute mean change
   is below ``tol`` or ``max_iter`` is reached.

Squared-error alignment and the weighted-mean update use the same observation
precisions. Gaussian-DTW distances select the initial medoid; they are not the
cost used in the iterative averaging step. Initialisation can affect the result,
and reaching the iteration limit does not establish convergence.

Signal channels and support
---------------------------
Alignment uses signal levels and their uncertainty. Dwell times follow the same
warping paths and are averaged separately. The returned ``std`` combines weighted
between-observation spread with mean-estimation uncertainty; it is not a calibrated
confidence interval. ``depth`` counts distinct reads covering each position, and
``supported`` marks positions meeting ``min_depth``.

Closed-end DTW is the default for fragments with comparable endpoints. Callers
may explicitly enable open ends for partial overlap, which can reduce edge depth.
Reads must share a current scale, or callers may use ``normalize=True`` for
per-read robust normalisation. The output axis follows the initial medoid; its
positions should not be interpreted as residue identities or steps per residue.
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
