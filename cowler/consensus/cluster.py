"""Reference-free DTW clustering and medoid selection for signal consensus.

This is the shared first stage for both signal-consensus workflows:

    normalized step reads
        -> pairwise DTW distances
        -> hierarchical clusters
        -> one medoid per cluster
        -> {DBA, profile-HMM}

DTW distances are used only for grouping and initialization.  They are not
likelihoods and must not be passed to Component 2's likelihood ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
from scipy.spatial.distance import squareform

from ..align.cost import cost_gaussian
from ..align.dtw import dtw_pairwise

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int_]

_LINKAGE_METHODS = {"single", "complete", "average", "weighted"}


@dataclass(frozen=True)
class SignalCluster:
    """One DTW cluster.

    ``members`` and ``medoid_index`` index the original input sequence.  Use
    :meth:`select` to obtain the cluster reads and the medoid's local index for
    either consensus engine.
    """

    label: int
    members: IntArr
    medoid_index: int
    medoid_cost: float

    @property
    def n_reads(self) -> int:
        return int(self.members.size)

    @property
    def medoid_position(self) -> int:
        positions = np.flatnonzero(self.members == self.medoid_index)
        if positions.size != 1:
            raise ValueError("cluster medoid is not present exactly once in members")
        return int(positions[0])

    def select(self, reads: Sequence[Any]) -> tuple[list[Any], int]:
        """Return this cluster's reads plus the medoid index within that list."""
        if self.members.size and (
            np.any(self.members < 0) or np.any(self.members >= len(reads))
        ):
            raise ValueError("cluster contains an invalid member index")
        return [reads[int(i)] for i in self.members], self.medoid_position


@dataclass(frozen=True)
class ClusteringResult:
    """Pairwise distances, linkage tree, assignments, and selected medoids."""

    distance_matrix: FloatArr
    linkage_matrix: FloatArr
    labels: IntArr
    clusters: tuple[SignalCluster, ...]
    distance_shift: float


@dataclass(frozen=True)
class MedoidCandidate:
    """One candidate in deterministic ascending within-group row-sum order."""

    index: int
    row_sum: float


def rank_medoid_candidates(
    distance_matrix: FloatArr,
    members: npt.ArrayLike | None = None,
    *,
    top_k: int | None = 5,
) -> tuple[MedoidCandidate, ...]:
    """Rank medoid candidates using the production row-sum criterion.

    ``index`` always refers to the original distance-matrix axis.  Scores use only
    the selected ``members``.  Equal row sums are resolved by the lowest original
    matrix index, matching :func:`medoid` and ``np.argmin`` on input-index order.
    Pass ``top_k=None`` to return the complete ranking.
    """
    D = _validate_distances(distance_matrix)
    idx = (
        np.arange(D.shape[0], dtype=np.int_)
        if members is None
        else np.asarray(members, dtype=np.int_)
    )
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("members must be a non-empty one-dimensional index array")
    if np.any(idx < 0) or np.any(idx >= D.shape[0]) or np.unique(idx).size != idx.size:
        raise ValueError("members contains invalid or duplicate indices")
    if top_k is not None and (isinstance(top_k, bool) or int(top_k) != top_k or top_k < 1):
        raise ValueError("top_k must be a positive integer or None")

    idx = np.sort(idx)
    costs = D[np.ix_(idx, idx)].sum(axis=1)
    order = np.lexsort((idx, costs))
    if top_k is not None:
        order = order[: min(int(top_k), order.size)]
    return tuple(
        MedoidCandidate(index=int(idx[position]), row_sum=float(costs[position]))
        for position in order
    )


def medoid(distance_matrix: FloatArr, members: npt.ArrayLike | None = None) -> int:
    """Return the original index with minimum within-group distance sum.

    Ties are deterministic: the lowest original index wins.
    """
    return rank_medoid_candidates(
        distance_matrix, members, top_k=1
    )[0].index


def cluster_from_distances(
    distance_matrix: FloatArr,
    *,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    method: str = "average",
    min_cluster_size: int = 1,
) -> ClusteringResult:
    """Cluster a precomputed symmetric DTW distance matrix.

    Exactly one of ``n_clusters`` or ``distance_threshold`` is required.  Gaussian
    DTW costs may be negative, so condensed distances are shifted to non-negative
    values before SciPy linkage.  The constant shift preserves ordering for the
    supported linkage methods and is recorded in the result.
    """
    D = _validate_distances(distance_matrix)
    K = D.shape[0]
    if (n_clusters is None) == (distance_threshold is None):
        raise ValueError("provide exactly one of n_clusters or distance_threshold")
    if method not in _LINKAGE_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_LINKAGE_METHODS)} for DTW distances"
        )
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")

    if n_clusters is not None:
        if not 1 <= n_clusters <= K:
            raise ValueError("n_clusters must be between 1 and the number of reads")
    elif distance_threshold is not None and not np.isfinite(distance_threshold):
        raise ValueError("distance_threshold must be finite")

    if K == 1:
        raw_labels = np.zeros(1, dtype=np.int_)
        linkage_matrix = np.empty((0, 4), dtype=float)
        shift = 0.0
    else:
        condensed = np.asarray(squareform(D, checks=False), dtype=float)
        shift = max(0.0, -float(np.min(condensed)))
        shifted = condensed + shift
        linkage_matrix = np.asarray(linkage(shifted, method=method), dtype=float)
        if n_clusters is not None:
            raw_labels = np.asarray(
                cut_tree(linkage_matrix, n_clusters=[n_clusters])[:, 0],
                dtype=np.int_,
            )
        else:
            assert distance_threshold is not None
            raw_labels = np.asarray(
                fcluster(
                    linkage_matrix,
                    distance_threshold + shift,
                    criterion="distance",
                ),
                dtype=np.int_,
            )

    labels = np.full(K, -1, dtype=np.int_)
    member_groups = [
        np.flatnonzero(raw_labels == label).astype(np.int_)
        for label in np.unique(raw_labels)
    ]
    member_groups = [
        members for members in member_groups if members.size >= min_cluster_size
    ]
    member_groups.sort(key=lambda members: int(members[0]))

    clusters: list[SignalCluster] = []
    for canonical_label, members in enumerate(member_groups):
        labels[members] = canonical_label
        centre = medoid(D, members)
        local = D[np.ix_(members, members)]
        centre_pos = int(np.flatnonzero(members == centre)[0])
        clusters.append(
            SignalCluster(
                label=canonical_label,
                members=members,
                medoid_index=centre,
                medoid_cost=float(local[centre_pos].sum()),
            )
        )

    return ClusteringResult(
        distance_matrix=D.copy(),
        linkage_matrix=linkage_matrix,
        labels=labels,
        clusters=tuple(clusters),
        distance_shift=shift,
    )


def cluster_reads_dtw(
    reads: Sequence[Any],
    *,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    method: str = "average",
    min_cluster_size: int = 1,
    cost: Callable[..., FloatArr] = cost_gaussian,
    normalize: bool = True,
    **dtw_kw: Any,
) -> ClusteringResult:
    """Compute pairwise DTW distances, cluster reads, and select each medoid."""
    if len(reads) == 0:
        raise ValueError("reads must be non-empty")
    D = dtw_pairwise(reads, cost=cost, normalize=normalize, **dtw_kw)
    return cluster_from_distances(
        D,
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        method=method,
        min_cluster_size=min_cluster_size,
    )


def _validate_distances(distance_matrix: FloatArr) -> FloatArr:
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix must be a non-empty square matrix")
    if not np.all(np.isfinite(D)):
        raise ValueError("distance_matrix contains NaN or infinity")
    if not np.allclose(D, D.T, rtol=1e-10, atol=1e-12):
        raise ValueError("distance_matrix must be symmetric")
    if not np.allclose(np.diag(D), 0.0, atol=1e-12):
        raise ValueError("distance_matrix diagonal must be zero")
    return D
