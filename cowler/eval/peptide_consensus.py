"""Reusable metrics for peptide signal-consensus and outlier evaluation.

This module deliberately stops at the signal-profile level.  It aligns fitted
profiles, scores held-out step traces against fixed profiles with Gaussian DTW,
and computes classification/bootstrap summaries.  It does not decode residues
and it never treats a DTW distance as a likelihood.

Dataset loading, annotation-assisted interval selection, split construction,
and plotting belong in ``scripts/evaluate_psk_dba_consensus.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from ..align.cost import cost_gaussian, cost_l2
from ..align.dtw import NoAdmissiblePathError, dtw
from ..consensus.cluster import ClusteringResult, cluster_from_distances, medoid

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
ScoreDirection = Literal["lower", "higher"]
MAX_SUPPORTED_DWELL_RATIO = 100.0
DEFAULT_DBA_UPSTREAM_OUTLIER_Z = 3.5
DEFAULT_HMM_UPSTREAM_OUTLIER_Z = 3.0


@dataclass(frozen=True)
class ProfileAlignmentMetrics:
    """Metrics after bounded L2-DTW alignment of two supported profiles."""

    rmse: float
    mae: float
    correlation: float
    normalized_l2_dtw: float
    n_a: int
    n_b: int
    path_length: int
    mean_warp_deviation: float
    max_warp_deviation: float
    path: IntArr
    difference: FloatArr


@dataclass(frozen=True)
class ClassificationMetrics:
    """Multi-class metrics for a fixed-profile score grid."""

    labels: tuple[str, ...]
    confusion: IntArr
    precision: FloatArr
    recall: FloatArr
    f1: FloatArr
    counts: IntArr
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    predicted_index: IntArr
    margin: FloatArr
    score_direction: ScoreDirection


@dataclass(frozen=True)
class HMMFitDiagnostics:
    """Predeclared technical acceptance summary for one fitted profile HMM."""

    quality_pass: bool
    failure_reasons: tuple[str, ...]
    finite_pass: bool
    convergence_pass: bool
    parameter_delta_pass: bool
    likelihood_pass: bool
    transition_probability_pass: bool
    transition_row_pass: bool
    support_pass: bool
    support_mask_pass: bool
    support_fraction_pass: bool
    internal_support_pass: bool
    terminal_support_pass: bool
    std_floor_pass: bool
    dwell_pass: bool
    n_states: int
    n_iter: int
    max_iter: int
    stop_reason: str
    cap_hit: bool
    fit_tolerance: float
    final_parameter_delta: float
    n_likelihood_decreases: int
    max_relative_likelihood_decrease: float
    transition_probability_min: float
    transition_probability_max: float
    max_transition_row_sum_error: float
    required_support_depth: int
    supported_fraction: float
    max_internal_unsupported_run: int
    max_terminal_unsupported_run: int
    terminal_unsupported_fraction: float
    std_floor_fraction: float
    dwell_min: float
    dwell_max: float
    dwell_ratio: float


@dataclass(frozen=True)
class DistanceOutlierMetrics:
    """Trace-level summaries from a precomputed within-dataset DTW matrix.

    Distance-to-medoid z-scores are calculated within the supplied clusters.
    kNN distances can be calculated either across the complete dataset (the
    exploratory subgroup diagnostic) or within each supplied cluster (the
    cluster-aware automatic-filtering mode).
    """

    cluster_size: IntArr
    medoid_index: IntArr
    medoid_flag: npt.NDArray[np.bool_]
    normalized_dtw_to_medoid: FloatArr
    robust_z_score: FloatArr
    knn_distance: FloatArr
    knn_robust_z_score: FloatArr


@dataclass(frozen=True)
class DistanceClusterSelection:
    """Cluster selection used to calibrate distance outliers in the supplied cohort.

    Candidate average-linkage cuts are accepted only when every cluster meets
    the minimum-size rule and the best shifted-distance silhouette reaches the
    declared threshold.  Otherwise all reads remain in one cluster.
    """

    clustering: ClusteringResult
    selected_clusters: int
    best_valid_silhouette: float
    required_cluster_size: int
    silhouette_threshold: float


def _precomputed_silhouette_score(
    distance_matrix: FloatArr, labels: IntArr
) -> float:
    """Return the mean silhouette for a non-negative precomputed distance grid."""

    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= labels.size:
        raise ValueError("silhouette requires between 2 and n_events - 1 clusters")
    sample_scores = np.zeros(labels.size, dtype=float)
    for index, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        same = same[same != index]
        if same.size == 0:
            continue
        within = float(np.mean(distance_matrix[index, same]))
        nearest_other = min(
            float(np.mean(distance_matrix[index, labels == other]))
            for other in unique
            if other != label
        )
        denominator = max(within, nearest_other)
        if denominator > 0.0:
            sample_scores[index] = (nearest_other - within) / denominator
    return float(np.mean(sample_scores))


def select_distance_outlier_clusters(
    distance_matrix: npt.ArrayLike,
    *,
    min_cluster_fraction: float = 0.08,
    min_cluster_size: int = 5,
    max_clusters: int = 4,
    silhouette_threshold: float = 0.25,
) -> DistanceClusterSelection:
    """Select cluster structure for robust within-species outlier scoring.

    Gaussian-DTW distances can be negative.  The same constant off-diagonal
    shift used by :func:`cluster_from_distances` is therefore applied only for
    silhouette calculation; clustering and all returned distances retain their
    original path-normalized values.
    """

    matrix = np.asarray(distance_matrix, dtype=float)
    if not 0.0 <= min_cluster_fraction <= 1.0:
        raise ValueError("min_cluster_fraction must be between zero and one")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    if max_clusters < 1:
        raise ValueError("max_clusters must be >= 1")
    if not np.isfinite(silhouette_threshold):
        raise ValueError("silhouette_threshold must be finite")

    one_cluster = cluster_from_distances(matrix, n_clusters=1, method="average")
    n_events = int(one_cluster.labels.size)
    required_size = max(
        min_cluster_size, int(np.ceil(min_cluster_fraction * n_events))
    )
    shifted = matrix.copy()
    off_diagonal = ~np.eye(n_events, dtype=bool)
    shifted[off_diagonal] += one_cluster.distance_shift
    np.fill_diagonal(shifted, 0.0)

    candidates: list[tuple[float, int, ClusteringResult]] = []
    upper = min(max_clusters, n_events - 1)
    for n_clusters in range(2, upper + 1):
        clustering = cluster_from_distances(
            matrix, n_clusters=n_clusters, method="average"
        )
        counts = np.bincount(clustering.labels)
        if counts.size != n_clusters or int(np.min(counts)) < required_size:
            continue
        score = _precomputed_silhouette_score(shifted, clustering.labels)
        candidates.append((score, n_clusters, clustering))

    if candidates:
        best_score, best_count, best = max(
            candidates, key=lambda item: (item[0], -item[1])
        )
    else:
        best_score, best_count, best = float("nan"), 1, one_cluster
    if not np.isfinite(best_score) or best_score < silhouette_threshold:
        selected_count = 1
        selected = one_cluster
    else:
        selected_count = best_count
        selected = best
    return DistanceClusterSelection(
        clustering=selected,
        selected_clusters=selected_count,
        best_valid_silhouette=best_score,
        required_cluster_size=required_size,
        silhouette_threshold=silhouette_threshold,
    )


def robust_z_scores(values: npt.ArrayLike) -> FloatArr:
    """Return median/MAD z-scores with a finite standard-deviation fallback.

    The robust scale is ``1.4826 * MAD``. A standard-deviation fallback is used
    only for tied samples whose MAD is numerically zero; a constant sample maps
    to all-zero scores.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains NaN or infinity")
    centre = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - centre)))
    tolerance = np.finfo(float).eps * max(1.0, abs(centre))
    if scale <= tolerance:
        scale = float(np.std(array))
    if scale <= tolerance:
        return np.zeros(array.size, dtype=float)
    return np.asarray((array - centre) / scale, dtype=float)


def distance_outlier_metrics(
    distance_matrix: npt.ArrayLike,
    cluster_labels: npt.ArrayLike,
    *,
    n_neighbors: int = 5,
    knn_within_clusters: bool = False,
) -> DistanceOutlierMetrics:
    """Summarize medoid and kNN distances without recomputing DTW.

    ``distance_matrix`` must contain path-normalized, symmetric DTW distances.
    Cluster labels may be any non-negative integers. By default, the kNN
    distance is the mean distance to the nearest
    ``min(n_neighbors, n_events - 1)`` traces in the complete dataset. Set
    ``knn_within_clusters=True`` when the metric will drive exclusions: each
    trace is then compared only with its cluster peers, so a coherent minority
    cluster is not labelled anomalous merely because it is separated from the
    majority. Singleton clusters receive zero kNN distance and z-score.
    """

    matrix = np.asarray(distance_matrix, dtype=float)
    labels = np.asarray(cluster_labels, dtype=np.int64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError("distance_matrix must be a non-empty square matrix")
    if labels.shape != (matrix.shape[0],) or np.any(labels < 0):
        raise ValueError("cluster_labels must be one non-negative label per trace")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance_matrix contains NaN or infinity")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("distance_matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12):
        raise ValueError("distance_matrix diagonal must be zero")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")

    n_traces = matrix.shape[0]
    cluster_size = np.empty(n_traces, dtype=np.int64)
    medoid_index = np.empty(n_traces, dtype=np.int64)
    normalized_distance = np.empty(n_traces, dtype=float)
    robust_z = np.empty(n_traces, dtype=float)
    medoid_flag = np.zeros(n_traces, dtype=bool)
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label).astype(np.int64)
        centre = medoid(matrix, members)
        values = matrix[members, centre]
        cluster_size[members] = members.size
        medoid_index[members] = centre
        normalized_distance[members] = values
        non_medoid = members != centre
        robust_z[members] = 0.0
        if np.any(non_medoid):
            robust_z[members[non_medoid]] = robust_z_scores(values[non_medoid])
        medoid_flag[centre] = True

    knn_distance = np.zeros(n_traces, dtype=float)
    knn_robust_z = np.zeros(n_traces, dtype=float)
    if knn_within_clusters:
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label).astype(np.int64)
            if members.size == 1:
                continue
            k = min(n_neighbors, members.size - 1)
            within = matrix[np.ix_(members, members)].copy()
            np.fill_diagonal(within, np.inf)
            nearest = np.partition(within, kth=k - 1, axis=1)[:, :k]
            values = np.mean(nearest, axis=1)
            knn_distance[members] = values
            knn_robust_z[members] = robust_z_scores(values)
    elif n_traces > 1:
        k = min(n_neighbors, n_traces - 1)
        without_self = matrix.copy()
        np.fill_diagonal(without_self, np.inf)
        nearest = np.partition(without_self, kth=k - 1, axis=1)[:, :k]
        knn_distance = np.mean(nearest, axis=1)
        knn_robust_z = robust_z_scores(knn_distance)
    return DistanceOutlierMetrics(
        cluster_size=cluster_size,
        medoid_index=medoid_index,
        medoid_flag=medoid_flag,
        normalized_dtw_to_medoid=normalized_distance,
        robust_z_score=robust_z,
        knn_distance=np.asarray(knn_distance, dtype=float),
        knn_robust_z_score=np.asarray(knn_robust_z, dtype=float),
    )


def _supported_values(
    values: npt.ArrayLike, supported: npt.ArrayLike | None
) -> FloatArr:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("profile values must be a non-empty one-dimensional array")
    mask = np.isfinite(array)
    if supported is not None:
        support = np.asarray(supported, dtype=bool)
        if support.shape != array.shape:
            raise ValueError("supported mask must match profile values")
        mask &= support
    selected = array[mask]
    if selected.size == 0:
        raise ValueError("profile has no finite supported positions")
    return selected


def align_profiles(
    mean_a: npt.ArrayLike,
    mean_b: npt.ArrayLike,
    *,
    supported_a: npt.ArrayLike | None = None,
    supported_b: npt.ArrayLike | None = None,
    max_run: int | None = 3,
) -> ProfileAlignmentMetrics:
    """Align two profiles with closed, bounded, normalized L2-DTW.

    ``normalized_l2_dtw`` is the square root of DTW's path-normalized squared
    Euclidean cost.  The aligned RMSE/MAE use the returned correspondence path.
    Profile positions are not treated as independent replicates; confidence
    intervals must come from whole-event resampling and profile refitting.
    """

    a = _supported_values(mean_a, supported_a)
    b = _supported_values(mean_b, supported_b)
    result = dtw(cost_l2(a, b), max_run=max_run)
    ai = result.path[:, 0]
    bi = result.path[:, 1]
    difference = a[ai] - b[bi]
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    mae = float(np.mean(np.abs(difference)))
    if np.std(a[ai]) == 0.0 or np.std(b[bi]) == 0.0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(a[ai], b[bi])[0, 1])

    if a.size == 1 or b.size == 1:
        deviation = np.zeros(result.path.shape[0], dtype=float)
    else:
        deviation = np.abs(ai / (a.size - 1) - bi / (b.size - 1))
    return ProfileAlignmentMetrics(
        rmse=rmse,
        mae=mae,
        correlation=correlation,
        normalized_l2_dtw=float(np.sqrt(max(result.norm_dist, 0.0))),
        n_a=int(a.size),
        n_b=int(b.size),
        path_length=int(result.path.shape[0]),
        mean_warp_deviation=float(np.mean(deviation)),
        max_warp_deviation=float(np.max(deviation)),
        path=result.path.astype(np.int64, copy=False),
        difference=difference,
    )


def _mean_std(read: Any) -> tuple[FloatArr, FloatArr]:
    if hasattr(read, "mean") and hasattr(read, "std"):
        mean = np.asarray(read.mean, dtype=float)
        std = np.asarray(read.std, dtype=float)
    else:
        values = tuple(read)
        if len(values) not in (2, 3):
            raise ValueError("a signal trace must contain (mean, std[, dwell])")
        mean = np.asarray(values[0], dtype=float)
        std = np.asarray(values[1], dtype=float)
    if mean.ndim != 1 or mean.size == 0 or std.shape != mean.shape:
        raise ValueError("signal mean/std must be matching non-empty 1-D arrays")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("signal mean/std contains NaN or infinity")
    return mean, std


def uncertainty_dtw_score(
    read: Any,
    profile: Any,
    *,
    max_run: int | None = 3,
) -> float:
    """Return lower-is-better normalized Gaussian-DTW distance.

    This can be negative because it is a path-normalized negative log density.
    It is a classification score, not a likelihood or likelihood ratio.
    """

    read_mean, read_std = _mean_std(read)
    profile_mean, profile_std = _mean_std(profile)
    supported_value = getattr(profile, "supported", None)
    if supported_value is not None:
        supported = np.asarray(supported_value, dtype=bool)
        if supported.shape != profile_mean.shape:
            raise ValueError("profile supported mask must match its mean")
        profile_mean = profile_mean[supported]
        profile_std = profile_std[supported]
    if profile_mean.size == 0:
        raise ValueError("profile has no supported positions")
    return float(
        dtw(
            cost_gaussian(read_mean, read_std, profile_mean, profile_std),
            max_run=max_run,
        ).norm_dist
    )


def fixed_profile_scores(
    reads: Sequence[Any],
    profiles: Mapping[str, Any],
    *,
    max_run: int | None = 3,
) -> tuple[tuple[str, ...], FloatArr]:
    """Score each held-out trace against each already-fitted profile.

    A profile that cannot be reached under the declared hard ``max_run`` bound
    receives positive-infinite distance.  This preserves the constraint instead
    of silently loosening it; classification remains valid when every read has
    at least one admissible profile.
    """

    labels = tuple(profiles)
    if not labels:
        raise ValueError("profiles must be non-empty")
    scores = np.empty((len(reads), len(labels)), dtype=float)
    for row, read in enumerate(reads):
        for column, label in enumerate(labels):
            try:
                scores[row, column] = uncertainty_dtw_score(
                    read, profiles[label], max_run=max_run
                )
            except NoAdmissiblePathError:
                scores[row, column] = np.inf
    return labels, scores


def fixed_hmm_profile_scores(
    reads: Sequence[Any],
    profiles: Mapping[str, Any],
) -> tuple[tuple[str, ...], FloatArr]:
    """Return raw higher-is-better likelihoods from immutable HMM profiles.

    The public fixed-profile scorer owns the HMM probability semantics.  This
    helper only constructs the held-out event-by-profile grid; it never refits,
    negates, or otherwise transforms the returned log likelihoods.
    """

    from ..consensus.hmm import score_hmm_profile

    labels = tuple(profiles)
    if not labels:
        raise ValueError("profiles must be non-empty")
    scores = np.empty((len(reads), len(labels)), dtype=float)
    for row, read in enumerate(reads):
        for column, label in enumerate(labels):
            scores[row, column] = float(
                score_hmm_profile(read, profiles[label]).log_likelihood
            )
    return labels, scores


def _validate_score_direction(score_direction: str) -> ScoreDirection:
    if score_direction == "lower":
        return "lower"
    if score_direction == "higher":
        return "higher"
    raise ValueError("score_direction must be 'lower' or 'higher'")


def classification_metrics(
    true_index: npt.ArrayLike,
    scores: npt.ArrayLike,
    labels: Sequence[str],
    *,
    score_direction: ScoreDirection = "lower",
) -> ClassificationMetrics:
    """Summarize a score matrix using deterministic direction-aware calls.

    Confidence margins are always non-negative: runner-up minus winner for a
    lower-is-better distance, and winner minus runner-up for a
    higher-is-better likelihood.
    """

    truth = np.asarray(true_index, dtype=np.int64)
    score_grid = np.asarray(scores, dtype=float)
    names = tuple(str(label) for label in labels)
    direction = _validate_score_direction(score_direction)
    if truth.ndim != 1 or score_grid.shape != (truth.size, len(names)):
        raise ValueError("scores must have shape (n_events, n_labels)")
    if truth.size == 0 or len(names) < 2:
        raise ValueError("classification requires events and at least two labels")
    if np.any((truth < 0) | (truth >= len(names))):
        raise ValueError("true_index contains an unknown class")
    if np.any(np.isnan(score_grid)):
        raise ValueError("scores contains NaN")

    if direction == "lower":
        if np.any(np.isneginf(score_grid)):
            raise ValueError("lower-is-better scores cannot contain -infinity")
        if np.any(~np.any(np.isfinite(score_grid), axis=1)):
            raise ValueError("each event requires at least one admissible finite score")
        predicted = np.argmin(score_grid, axis=1).astype(np.int64)
        sorted_scores = np.sort(score_grid, axis=1)
        margin = sorted_scores[:, 1] - sorted_scores[:, 0]
    else:
        if np.any(np.isposinf(score_grid)):
            raise ValueError("higher-is-better scores cannot contain +infinity")
        if np.any(~np.any(np.isfinite(score_grid), axis=1)):
            raise ValueError("each event requires at least one admissible finite score")
        predicted = np.argmax(score_grid, axis=1).astype(np.int64)
        sorted_scores = np.sort(score_grid, axis=1)
        margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    counts = confusion.sum(axis=1)
    called = confusion.sum(axis=0)
    diagonal = np.diag(confusion).astype(float)
    precision = np.divide(
        diagonal, called, out=np.zeros_like(diagonal), where=called > 0
    )
    recall = np.divide(
        diagonal, counts, out=np.zeros_like(diagonal), where=counts > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0.0,
    )
    return ClassificationMetrics(
        labels=names,
        confusion=confusion,
        precision=precision,
        recall=recall,
        f1=f1,
        counts=counts.astype(np.int64),
        accuracy=float(np.mean(predicted == truth)),
        balanced_accuracy=float(np.mean(recall[counts > 0])),
        macro_f1=float(np.mean(f1[counts > 0])),
        predicted_index=predicted,
        margin=margin,
        score_direction=direction,
    )


def _max_false_run(mask: npt.NDArray[np.bool_]) -> int:
    longest = 0
    current = 0
    for value in mask:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def hmm_fit_diagnostics(
    profile: Any,
    *,
    training_depth: int,
    max_iter: int | None = None,
    min_std: float | None = None,
    tol: float | None = None,
) -> HMMFitDiagnostics:
    """Apply the locked profile-HMM technical acceptance gates.

    The gates come from ``docs/hmm_consensus_validation.md``: finite returned
    diagnostics, convergence before the declared cap, no raw-likelihood
    relative decrease above ``1e-3``, normalized transition rows and global
    probabilities in ``[1e-3, 0.995]``, at least 80% supported states, internal
    unsupported runs of at most three states, terminal unsupported runs of at
    most 10% of the profile, at most 25% of spreads at the numerical floor, and
    positive supported-state dwell values spanning no more than 100-fold.

    Support is recomputed at ``max(2, ceil(0.5 * training_depth))`` so the gate
    does not silently trust a profile fitted with a weaker support threshold.
    ``max_iter``, ``min_std``, and ``tol`` default to the immutable fit settings
    stored on the profile; explicit values support diagnostics for older saved
    profiles.
    """

    if training_depth < 1:
        raise ValueError("training_depth must be >= 1")
    settings = getattr(profile, "settings", None)
    configured_max_iter = (
        max_iter if max_iter is not None else getattr(settings, "max_iter", None)
    )
    configured_min_std = (
        min_std if min_std is not None else getattr(settings, "min_std", None)
    )
    configured_tol = tol if tol is not None else getattr(settings, "tol", None)
    if configured_max_iter is None:
        raise ValueError("max_iter is required when profile settings are unavailable")
    if configured_min_std is None:
        raise ValueError("min_std is required when profile settings are unavailable")
    if configured_tol is None:
        raise ValueError("tol is required when profile settings are unavailable")
    max_iter_value = int(configured_max_iter)
    min_std_value = float(configured_min_std)
    tol_value = float(configured_tol)
    if max_iter_value < 1:
        raise ValueError("max_iter must be >= 1")
    if min_std_value <= 0.0 or not np.isfinite(min_std_value):
        raise ValueError("min_std must be finite and positive")
    if tol_value < 0.0 or not np.isfinite(tol_value):
        raise ValueError("tol must be finite and non-negative")

    mean = np.asarray(profile.mean, dtype=float)
    std = np.asarray(profile.std, dtype=float)
    stderr = np.asarray(profile.stderr, dtype=float)
    dwell = np.asarray(profile.dwell, dtype=float)
    depth = np.asarray(profile.depth, dtype=float)
    declared_supported = np.asarray(profile.supported, dtype=bool)
    if mean.ndim != 1 or mean.size == 0:
        raise ValueError("profile mean must be a non-empty one-dimensional array")
    n_states = int(mean.size)
    for name, values in (
        ("std", std),
        ("stderr", stderr),
        ("dwell", dwell),
        ("depth", depth),
        ("supported", declared_supported),
    ):
        if values.shape != mean.shape:
            raise ValueError(f"profile {name} must match profile mean")

    offsets = np.asarray(profile.transition_offsets, dtype=np.int64)
    probabilities = np.asarray(profile.transition_prob, dtype=float)
    if offsets.ndim != 1 or offsets.size == 0 or probabilities.ndim != 1:
        raise ValueError("transition parameters must be non-empty one-dimensional arrays")
    likelihood = np.asarray(profile.log_likelihood, dtype=float)
    if likelihood.ndim != 1 or likelihood.size == 0:
        raise ValueError("log_likelihood must be a non-empty one-dimensional trace")

    n_iter = int(profile.n_iter)
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    stop_reason = str(getattr(profile, "stop_reason", ""))
    converged = bool(profile.converged)
    cap_hit = stop_reason == "max_iter" or n_iter >= max_iter_value
    convergence_pass = converged and stop_reason == "converged" and not cap_hit
    parameter_delta = np.asarray(
        getattr(profile, "parameter_delta", np.empty(0)), dtype=float
    )
    parameter_trace_pass = parameter_delta.shape == (n_iter,)
    parameter_trace_finite = bool(
        parameter_trace_pass and np.all(np.isfinite(parameter_delta))
    )
    final_parameter_delta = (
        float(parameter_delta[-1])
        if parameter_trace_finite and parameter_delta.size
        else float("inf")
    )
    parameter_delta_pass = bool(
        parameter_trace_finite and final_parameter_delta <= tol_value
    )

    required_support_depth = max(2, int(np.ceil(0.5 * training_depth)))
    supported = np.isfinite(depth) & (depth >= required_support_depth)
    support_mask_pass = bool(np.array_equal(declared_supported, supported))
    supported_fraction = float(np.mean(supported))
    support_fraction_pass = supported_fraction >= 0.80
    supported_index = np.flatnonzero(supported)
    if supported_index.size == 0:
        leading_unsupported = n_states
        trailing_unsupported = n_states
        max_internal_unsupported_run = 0
    else:
        first_supported = int(supported_index[0])
        last_supported = int(supported_index[-1])
        leading_unsupported = first_supported
        trailing_unsupported = n_states - last_supported - 1
        internal = supported[first_supported : last_supported + 1]
        max_internal_unsupported_run = _max_false_run(internal)
    max_terminal_unsupported_run = max(
        leading_unsupported, trailing_unsupported
    )
    terminal_unsupported_fraction = max_terminal_unsupported_run / n_states
    internal_support_pass = max_internal_unsupported_run <= 3
    terminal_support_pass = terminal_unsupported_fraction <= 0.10

    supported_values_finite = bool(
        np.all(np.isfinite(mean[supported]))
        and np.all(np.isfinite(std[supported]))
        and np.all(np.isfinite(stderr[supported]))
        and np.all(np.isfinite(dwell[supported]))
    )
    supported_dwell = dwell[supported]
    dwell_min = (
        float(np.min(supported_dwell))
        if supported_dwell.size and np.all(np.isfinite(supported_dwell))
        else float("nan")
    )
    dwell_max = (
        float(np.max(supported_dwell))
        if supported_dwell.size and np.all(np.isfinite(supported_dwell))
        else float("nan")
    )
    dwell_ratio = (
        dwell_max / dwell_min
        if np.isfinite(dwell_min) and dwell_min > 0.0
        else float("inf")
    )
    dwell_pass = bool(
        supported_dwell.size
        and np.all(np.isfinite(dwell))
        and np.all(dwell > 0.0)
        and dwell_ratio <= MAX_SUPPORTED_DWELL_RATIO
    )
    finite_pass = bool(
        np.all(np.isfinite(mean))
        and np.all(np.isfinite(std))
        and np.all(std > 0.0)
        and np.all(np.isfinite(dwell))
        and np.all(np.isfinite(depth))
        and np.all(np.isfinite(probabilities))
        and np.all(np.isfinite(likelihood))
        and parameter_trace_finite
        and supported_values_finite
    )

    likelihood_trace_pass = likelihood.size == n_iter + 1
    likelihood_finite = bool(np.all(np.isfinite(likelihood)))
    if likelihood_finite and likelihood.size > 1:
        changes = np.diff(likelihood)
        decreases = changes < 0.0
        denominator = np.maximum(1.0, np.abs(likelihood[:-1]))
        relative_decrease = np.where(decreases, -changes / denominator, 0.0)
        n_likelihood_decreases = int(np.count_nonzero(decreases))
        max_relative_likelihood_decrease = float(np.max(relative_decrease))
    elif likelihood_finite:
        n_likelihood_decreases = 0
        max_relative_likelihood_decrease = 0.0
    else:
        n_likelihood_decreases = 0
        max_relative_likelihood_decrease = float("inf")
    likelihood_pass = bool(
        likelihood_finite
        and likelihood_trace_pass
        and max_relative_likelihood_decrease <= 1e-3
    )

    probability_shape_pass = probabilities.shape == offsets.shape
    transition_probability_min = (
        float(np.min(probabilities)) if probabilities.size else float("nan")
    )
    transition_probability_max = (
        float(np.max(probabilities)) if probabilities.size else float("nan")
    )
    transition_probability_pass = bool(
        probability_shape_pass
        and np.unique(offsets).size == offsets.size
        and np.all(np.isfinite(probabilities))
        and transition_probability_min >= 1e-3
        and transition_probability_max <= 0.995
        and abs(float(np.sum(probabilities)) - 1.0) <= 1e-10
    )

    max_transition_row_sum_error = float("inf")
    transition_row_pass = False
    if probability_shape_pass:
        from ..consensus.hmm import hmm_transition_matrix

        try:
            transition = np.asarray(
                hmm_transition_matrix(
                    n_states,
                    tuple(int(offset) for offset in offsets),
                    probabilities,
                ),
                dtype=float,
            )
            if transition.shape == (n_states, n_states):
                row_sums = np.sum(transition, axis=1)
                if np.all(np.isfinite(row_sums)):
                    max_transition_row_sum_error = float(
                        np.max(np.abs(row_sums - 1.0))
                    )
                    transition_row_pass = max_transition_row_sum_error <= 1e-10
        except (FloatingPointError, OverflowError, ValueError):
            pass

    floor_tolerance = max(
        np.finfo(float).eps, abs(min_std_value) * 1e-10
    )
    at_floor = np.isfinite(std) & (std <= min_std_value + floor_tolerance)
    std_floor_fraction = float(np.mean(at_floor))
    std_floor_pass = std_floor_fraction <= 0.25
    support_pass = bool(
        support_mask_pass
        and support_fraction_pass
        and internal_support_pass
        and terminal_support_pass
        and supported_values_finite
    )

    failure_reasons: list[str] = []
    if not finite_pass:
        failure_reasons.append("returned-model diagnostics are non-finite or invalid")
    if not convergence_pass:
        failure_reasons.append("fit did not converge before the declared iteration cap")
    if not parameter_trace_pass:
        failure_reasons.append(
            "parameter-delta trace does not contain one value per completed update"
        )
    elif not parameter_trace_finite:
        failure_reasons.append("parameter-delta trace contains NaN or infinity")
    elif not parameter_delta_pass:
        failure_reasons.append(
            "final parameter delta exceeds the declared tolerance "
            f"({final_parameter_delta:.6g} > {tol_value:.6g})"
        )
    if not likelihood_finite:
        failure_reasons.append("likelihood trace contains NaN or infinity")
    if not likelihood_trace_pass:
        failure_reasons.append(
            "likelihood trace does not contain the initial and every returned-model value"
        )
    if max_relative_likelihood_decrease > 1e-3:
        failure_reasons.append(
            "maximum relative likelihood decrease exceeds 1e-3 "
            f"({max_relative_likelihood_decrease:.6g})"
        )
    if not transition_probability_pass:
        failure_reasons.append(
            "global transition probabilities violate normalization or [1e-3, 0.995] bounds"
        )
    if not transition_row_pass:
        failure_reasons.append(
            "transition rows are invalid or exceed the 1e-10 row-sum tolerance"
        )
    if not support_mask_pass:
        failure_reasons.append(
            "stored support mask does not match max(2, ceil(0.5 * training_depth))"
        )
    if not supported_values_finite:
        failure_reasons.append("supported states contain non-finite diagnostics")
    if not support_fraction_pass:
        failure_reasons.append(
            f"supported fraction is below 0.80 ({supported_fraction:.6g})"
        )
    if not internal_support_pass:
        failure_reasons.append(
            "internal unsupported run exceeds three states "
            f"({max_internal_unsupported_run})"
        )
    if not terminal_support_pass:
        failure_reasons.append(
            "terminal unsupported fraction exceeds 0.10 "
            f"({terminal_unsupported_fraction:.6g})"
        )
    if not std_floor_pass:
        failure_reasons.append(
            f"spread-floor fraction exceeds 0.25 ({std_floor_fraction:.6g})"
        )
    if not dwell_pass:
        failure_reasons.append(
            "dwell values are non-positive or supported-state dwell range exceeds "
            f"{MAX_SUPPORTED_DWELL_RATIO:g}-fold ({dwell_ratio:.6g})"
        )

    quality_pass = bool(
        finite_pass
        and convergence_pass
        and parameter_delta_pass
        and likelihood_pass
        and transition_probability_pass
        and transition_row_pass
        and support_pass
        and std_floor_pass
        and dwell_pass
    )
    return HMMFitDiagnostics(
        quality_pass=quality_pass,
        failure_reasons=tuple(failure_reasons),
        finite_pass=finite_pass,
        convergence_pass=convergence_pass,
        parameter_delta_pass=parameter_delta_pass,
        likelihood_pass=likelihood_pass,
        transition_probability_pass=transition_probability_pass,
        transition_row_pass=transition_row_pass,
        support_pass=support_pass,
        support_mask_pass=support_mask_pass,
        support_fraction_pass=support_fraction_pass,
        internal_support_pass=internal_support_pass,
        terminal_support_pass=terminal_support_pass,
        std_floor_pass=std_floor_pass,
        dwell_pass=dwell_pass,
        n_states=n_states,
        n_iter=n_iter,
        max_iter=max_iter_value,
        stop_reason=stop_reason,
        cap_hit=cap_hit,
        fit_tolerance=tol_value,
        final_parameter_delta=final_parameter_delta,
        n_likelihood_decreases=n_likelihood_decreases,
        max_relative_likelihood_decrease=max_relative_likelihood_decrease,
        transition_probability_min=transition_probability_min,
        transition_probability_max=transition_probability_max,
        max_transition_row_sum_error=max_transition_row_sum_error,
        required_support_depth=required_support_depth,
        supported_fraction=supported_fraction,
        max_internal_unsupported_run=max_internal_unsupported_run,
        max_terminal_unsupported_run=max_terminal_unsupported_run,
        terminal_unsupported_fraction=terminal_unsupported_fraction,
        std_floor_fraction=std_floor_fraction,
        dwell_min=dwell_min,
        dwell_max=dwell_max,
        dwell_ratio=dwell_ratio,
    )


def percentile_interval(
    values: npt.ArrayLike, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided percentile interval over finite bootstrap replicates."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(array, [tail, 1.0 - tail])
    return float(low), float(high)


def bootstrap_classification(
    true_index: npt.ArrayLike,
    scores: npt.ArrayLike,
    labels: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    score_direction: ScoreDirection = "lower",
) -> dict[str, FloatArr]:
    """Stratified paired bootstrap using the declared fixed-score direction."""

    truth = np.asarray(true_index, dtype=np.int64)
    score_grid = np.asarray(scores, dtype=float)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    # Validate once and establish the expected grid/label contract.
    direction = _validate_score_direction(score_direction)
    classification_metrics(
        truth, score_grid, labels, score_direction=direction
    )
    rng = np.random.default_rng(seed)
    class_rows = [np.flatnonzero(truth == k) for k in range(len(labels))]
    accuracy = np.empty(n_bootstrap, dtype=float)
    balanced = np.empty(n_bootstrap, dtype=float)
    macro_f1 = np.empty(n_bootstrap, dtype=float)
    for repeat in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(rows, size=rows.size, replace=True) for rows in class_rows]
        )
        metrics = classification_metrics(
            truth[sampled],
            score_grid[sampled],
            labels,
            score_direction=direction,
        )
        accuracy[repeat] = metrics.accuracy
        balanced[repeat] = metrics.balanced_accuracy
        macro_f1[repeat] = metrics.macro_f1
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
    }


def binary_rank_metrics(
    binary_truth: npt.ArrayLike, decision_score: npt.ArrayLike
) -> tuple[float, float]:
    """Return AUROC and average precision for a higher-is-positive score."""

    truth = np.asarray(binary_truth, dtype=np.int64)
    score = np.asarray(decision_score, dtype=float)
    if truth.ndim != 1 or score.shape != truth.shape or truth.size == 0:
        raise ValueError("binary truth/score must be matching non-empty 1-D arrays")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isfinite(score)):
        raise ValueError("binary truth must be 0/1 and scores must be finite")
    positive = truth == 1
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(truth.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("both binary classes must be represented")

    ranks = rankdata(score, method="average")
    rank_sum = float(np.sum(ranks[positive]))
    auroc = (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)

    order = np.argsort(-score, kind="mergesort")
    ordered_truth = truth[order]
    cumulative_positive = np.cumsum(ordered_truth)
    positive_ranks = np.flatnonzero(ordered_truth == 1)
    average_precision = float(
        np.mean(cumulative_positive[positive_ranks] / (positive_ranks + 1))
    )
    return float(auroc), average_precision


def separation_ratio(
    between_distance: float,
    within_a_distance: float,
    within_b_distance: float,
) -> float:
    """Variant-profile distance relative to the two split-profile noise floors."""

    denominator = 0.5 * (within_a_distance + within_b_distance)
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(between_distance / denominator)
