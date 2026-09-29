"""Profile-HMM signal consensus initialized from a cluster medoid.

The HMM and DBA workflows share DTW clustering and the same medoid axis.  DBA
uses one hard warping path per read; this module instead uses forward-backward
posteriors and refines the profile with soft posterior updates.

The states are observed-signal profile positions, not peptide residues.  No DNA
alphabet, k-mer size, or steps-per-residue assumption appears here.  Residue
decoding remains a separate step requiring a measured peptide level model.

The probability model has a normalized uniform entry distribution, a free exit,
and boundary-renormalized transition rows.  Consequently a fixed-profile score is
a raw, higher-is-better log likelihood and does not acquire a ``+log(L)`` bonus
merely because a profile contains more states.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from ..align.dp import emission
from ._signal import SignalSteps, prepare_reads, validate_medoid_index
from .cluster import SignalCluster, medoid

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]
Int64Arr = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ProfileHMMSettings:
    """Settings needed to reproduce a profile-HMM fit and score new reads."""

    normalize: bool
    max_iter: int
    tol: float
    min_depth: int
    min_std: float
    transition_pseudocount: float
    transition_offsets: tuple[int, ...]
    initial_transition_prob: tuple[float, ...]
    transition_optimizer: str
    emission_optimizer: str


@dataclass(frozen=True)
class ProfileHMMConsensus:
    """A soft signal consensus on the medoid's profile-state axis.

    ``log_likelihood`` contains the initial-model likelihood followed by the
    likelihood after every completed parameter update.  It therefore has
    ``n_iter + 1`` entries, and its final value describes the returned model.
    Parameter-delta arrays contain one entry per completed update.
    """

    mean: FloatArr
    std: FloatArr
    stderr: FloatArr
    dwell: FloatArr
    depth: FloatArr
    supported: BoolArr
    transition_offsets: Int64Arr
    transition_prob: FloatArr
    mean_history: FloatArr
    std_history: FloatArr
    dwell_history: FloatArr
    transition_prob_history: FloatArr
    log_likelihood: FloatArr
    n_iter: int
    converged: bool
    medoid_index: int
    mean_delta: FloatArr
    std_delta: FloatArr
    transition_delta: FloatArr
    parameter_delta: FloatArr
    transition_data_objective_before: FloatArr
    transition_data_objective_after: FloatArr
    transition_prior_objective_before: FloatArr
    transition_prior_objective_after: FloatArr
    transition_penalized_objective_before: FloatArr
    transition_penalized_objective_after: FloatArr
    transition_gradient_norm: FloatArr
    transition_optimizer_success: BoolArr
    emission_objective_before: FloatArr
    emission_objective_after: FloatArr
    stop_reason: str
    settings: ProfileHMMSettings

    @property
    def likelihood_trace(self) -> FloatArr:
        """Alias making the trace semantics of ``log_likelihood`` explicit."""
        return self.log_likelihood

    @property
    def final_log_likelihood(self) -> float:
        """Raw likelihood recomputed from the returned parameters."""
        return float(self.log_likelihood[-1])

    @property
    def combined_delta(self) -> FloatArr:
        """Maximum parameter-component delta for every update."""
        return self.parameter_delta

    @property
    def delta(self) -> FloatArr:
        """Short compatibility alias for :attr:`parameter_delta`."""
        return self.parameter_delta

    @property
    def spread_delta(self) -> FloatArr:
        """Domain-language alias for the latent-profile ``std`` delta."""
        return self.std_delta

    @property
    def cap_hit(self) -> bool:
        """Whether fitting stopped at ``max_iter`` rather than tolerance."""
        return self.stop_reason == "max_iter"


@dataclass(frozen=True)
class ProfileHMMScore:
    """Immutable result of scoring one read against one fixed profile HMM."""

    log_likelihood: float
    posterior: FloatArr | None = None

    @property
    def logL(self) -> float:
        """Compatibility spelling used by the alignment forward-backward API."""
        return self.log_likelihood


@dataclass(frozen=True)
class _Expectation:
    log_likelihood: float
    gamma: FloatArr
    source_offset_counts: FloatArr


@dataclass(frozen=True)
class _FitPass:
    """One E-pass and the sufficient statistics used by the profile update."""

    log_likelihood: float
    expectations: tuple[tuple[SignalSteps, FloatArr], ...]
    sum_precision: FloatArr
    responsibility: FloatArr
    sum_dwell: FloatArr
    depth: FloatArr
    source_offset_counts: FloatArr


@dataclass(frozen=True)
class _TransitionUpdate:
    """One boundary-aware update of the shared displacement weights."""

    probabilities: FloatArr
    data_objective_before: float
    data_objective_after: float
    prior_objective_before: float
    prior_objective_after: float
    penalized_objective_before: float
    penalized_objective_after: float
    gradient_norm: float
    success: bool


@dataclass(frozen=True)
class _Maximization:
    """Candidate parameters and fixed-posterior objective diagnostics."""

    mean: FloatArr
    std: FloatArr
    dwell: FloatArr
    transition_prob: FloatArr
    transition: _TransitionUpdate
    emission_objective_before: float
    emission_objective_after: float


@dataclass(frozen=True)
class _EmissionStateUpdate:
    """Exact fixed-posterior update for one heteroscedastic profile state."""

    mean: float
    variance: float
    objective: float
    success: bool


def hmm_consensus(
    reads: Sequence[Any],
    *,
    medoid_index: int | None = None,
    distance_matrix: FloatArr | None = None,
    normalize: bool = True,
    max_iter: int = 20,
    tol: float = 1e-4,
    min_depth: int = 2,
    min_std: float = 1e-3,
    transition_offsets: npt.ArrayLike = (-1, 0, 1, 2),
    transition_prob: npt.ArrayLike | None = None,
    transition_pseudocount: float = 1.0,
) -> ProfileHMMConsensus:
    """Fit a medoid-initialized profile HMM by generalized-EM posterior updates.

    The default transition topology allows backsteps, holds, ordinary advances,
    and skips.  Its global displacement weights are learned from the cluster;
    each state's legal moves are then normalized to form a proper probability
    row.  When no initial weights are supplied, the medoid's one-state-per-step
    axis contributes ``L-1`` ordinary advances and a pseudocount keeps every
    other move possible.  This initializes direction without importing DNA
    chemistry rates.

    Shared displacement weights are fitted with the same source-dependent legal
    move normalization used by scoring.  The emission update remains an
    instrumented generalized-EM step; fixed-posterior transition, emission, and
    raw-likelihood traces let callers verify its behavior.
    """
    _validate_fit_options(
        max_iter=max_iter,
        tol=tol,
        min_depth=min_depth,
        min_std=min_std,
        transition_pseudocount=transition_pseudocount,
    )

    offsets = _validate_offsets(transition_offsets)
    prepared = prepare_reads(reads, normalize=normalize, min_std=min_std)
    if medoid_index is None:
        if distance_matrix is None:
            from ..align.dtw import dtw_pairwise

            distance_matrix = dtw_pairwise(
                [(read.mean, read.std) for read in prepared],
                normalize=False,
            )
        else:
            candidate_distances = np.asarray(distance_matrix, dtype=float)
            if candidate_distances.shape != (len(prepared), len(prepared)):
                raise ValueError(
                    "distance_matrix shape must match the number of reads"
                )
            if not np.all(np.isfinite(candidate_distances)):
                raise ValueError("distance_matrix contains NaN or infinity")
            distance_matrix = candidate_distances
        medoid_index = medoid(distance_matrix)
    centre = validate_medoid_index(medoid_index, len(prepared))

    initial = prepared[centre]
    mean = initial.mean.copy()
    std = np.maximum(initial.std.copy(), min_std)
    dwell = initial.dwell.copy()
    length = mean.size

    probabilities = _initial_transition_prob(
        offsets,
        length,
        transition_prob=transition_prob,
        pseudocount=transition_pseudocount,
    )
    settings = ProfileHMMSettings(
        normalize=bool(normalize),
        max_iter=int(max_iter),
        tol=float(tol),
        min_depth=int(min_depth),
        min_std=float(min_std),
        transition_pseudocount=float(transition_pseudocount),
        transition_offsets=tuple(int(value) for value in offsets),
        initial_transition_prob=tuple(float(value) for value in probabilities),
        transition_optimizer="boundary_aware_logit",
        emission_optimizer="heteroscedastic_profiled_scalar",
    )

    fit_pass = _fit_expectation_pass(
        prepared,
        mean,
        std,
        _log_transition_matrix(length, offsets, probabilities),
        offsets,
    )
    mean_history = [mean.copy()]
    std_history = [std.copy()]
    dwell_history = [dwell.copy()]
    transition_prob_history = [probabilities.copy()]
    likelihood_trace = [fit_pass.log_likelihood]
    mean_deltas: list[float] = []
    std_deltas: list[float] = []
    transition_deltas: list[float] = []
    parameter_deltas: list[float] = []
    transition_data_before: list[float] = []
    transition_data_after: list[float] = []
    transition_prior_before: list[float] = []
    transition_prior_after: list[float] = []
    transition_penalized_before: list[float] = []
    transition_penalized_after: list[float] = []
    transition_gradient_norms: list[float] = []
    transition_optimizer_successes: list[bool] = []
    emission_before: list[float] = []
    emission_after: list[float] = []
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        update = _maximization(
            fit_pass,
            mean,
            std,
            dwell,
            offsets,
            probabilities,
            min_std=min_std,
            transition_pseudocount=transition_pseudocount,
        )
        new_mean = update.mean
        new_std = update.std
        new_dwell = update.dwell
        new_probabilities = update.transition_prob

        mean_delta = float(np.max(np.abs(new_mean - mean)))
        std_delta = float(np.max(np.abs(new_std - std)))
        transition_delta = float(
            np.max(np.abs(new_probabilities - probabilities))
        )
        parameter_delta = max(mean_delta, std_delta, transition_delta)
        if not np.all(
            np.isfinite(
                np.asarray(
                    [mean_delta, std_delta, transition_delta, parameter_delta]
                )
            )
        ):
            raise FloatingPointError("profile-HMM parameter update is non-finite")

        mean_deltas.append(mean_delta)
        std_deltas.append(std_delta)
        transition_deltas.append(transition_delta)
        parameter_deltas.append(parameter_delta)
        transition_data_before.append(
            update.transition.data_objective_before
        )
        transition_data_after.append(update.transition.data_objective_after)
        transition_prior_before.append(
            update.transition.prior_objective_before
        )
        transition_prior_after.append(update.transition.prior_objective_after)
        transition_penalized_before.append(
            update.transition.penalized_objective_before
        )
        transition_penalized_after.append(
            update.transition.penalized_objective_after
        )
        transition_gradient_norms.append(update.transition.gradient_norm)
        transition_optimizer_successes.append(update.transition.success)
        emission_before.append(update.emission_objective_before)
        emission_after.append(update.emission_objective_after)
        mean, std, dwell = new_mean, new_std, new_dwell
        probabilities = new_probabilities
        mean_history.append(mean.copy())
        std_history.append(std.copy())
        dwell_history.append(dwell.copy())
        transition_prob_history.append(probabilities.copy())

        # This is both the next iteration's E-pass and, if fitting stops here,
        # the mandatory final pass for the exact parameters returned below.
        fit_pass = _fit_expectation_pass(
            prepared,
            mean,
            std,
            _log_transition_matrix(length, offsets, probabilities),
            offsets,
        )
        likelihood_trace.append(fit_pass.log_likelihood)
        if parameter_delta <= tol:
            converged = True
            break

    final_updated = fit_pass.sum_precision > 0.0
    stderr = np.full(length, np.inf, dtype=float)
    stderr[final_updated] = np.sqrt(1.0 / fit_pass.sum_precision[final_updated])
    if np.any(np.isnan(stderr)) or np.any(stderr < 0.0):
        raise FloatingPointError("profile-HMM standard error is invalid")

    trace = np.asarray(likelihood_trace, dtype=float)
    parameter_history = (
        np.asarray(mean_history, dtype=float),
        np.asarray(std_history, dtype=float),
        np.asarray(dwell_history, dtype=float),
        np.asarray(transition_prob_history, dtype=float),
    )
    deltas = (
        np.asarray(mean_deltas, dtype=float),
        np.asarray(std_deltas, dtype=float),
        np.asarray(transition_deltas, dtype=float),
        np.asarray(parameter_deltas, dtype=float),
    )
    objective_diagnostics = (
        np.asarray(transition_data_before, dtype=float),
        np.asarray(transition_data_after, dtype=float),
        np.asarray(transition_prior_before, dtype=float),
        np.asarray(transition_prior_after, dtype=float),
        np.asarray(transition_penalized_before, dtype=float),
        np.asarray(transition_penalized_after, dtype=float),
        np.asarray(transition_gradient_norms, dtype=float),
        np.asarray(emission_before, dtype=float),
        np.asarray(emission_after, dtype=float),
    )
    optimizer_success = np.asarray(
        transition_optimizer_successes, dtype=bool
    )
    if not np.all(np.isfinite(trace)) or any(
        not np.all(np.isfinite(values)) for values in deltas
    ) or any(
        not np.all(np.isfinite(values)) for values in parameter_history
    ) or any(
        not np.all(np.isfinite(values)) for values in objective_diagnostics
    ):
        raise FloatingPointError("profile-HMM diagnostics contain non-finite values")

    # Guard against an accidental change to the trace contract.
    if trace.size != iteration + 1 or any(
        values.size != iteration for values in deltas
    ) or any(
        values.size != iteration for values in objective_diagnostics
    ) or optimizer_success.size != iteration:
        raise RuntimeError("profile-HMM diagnostic trace lengths are inconsistent")
    if (
        parameter_history[0].shape != (iteration + 1, length)
        or parameter_history[1].shape != (iteration + 1, length)
        or parameter_history[2].shape != (iteration + 1, length)
        or parameter_history[3].shape != (iteration + 1, offsets.size)
    ):
        raise RuntimeError("profile-HMM parameter-history shapes are inconsistent")

    return ProfileHMMConsensus(
        mean=mean,
        std=std,
        stderr=stderr,
        dwell=dwell,
        depth=fit_pass.depth,
        supported=fit_pass.depth >= min_depth,
        transition_offsets=offsets,
        transition_prob=probabilities,
        mean_history=parameter_history[0],
        std_history=parameter_history[1],
        dwell_history=parameter_history[2],
        transition_prob_history=parameter_history[3],
        log_likelihood=trace,
        n_iter=iteration,
        converged=converged,
        medoid_index=centre,
        mean_delta=deltas[0],
        std_delta=deltas[1],
        transition_delta=deltas[2],
        parameter_delta=deltas[3],
        transition_data_objective_before=objective_diagnostics[0],
        transition_data_objective_after=objective_diagnostics[1],
        transition_prior_objective_before=objective_diagnostics[2],
        transition_prior_objective_after=objective_diagnostics[3],
        transition_penalized_objective_before=objective_diagnostics[4],
        transition_penalized_objective_after=objective_diagnostics[5],
        transition_gradient_norm=objective_diagnostics[6],
        transition_optimizer_success=optimizer_success,
        emission_objective_before=objective_diagnostics[7],
        emission_objective_after=objective_diagnostics[8],
        stop_reason="converged" if converged else "max_iter",
        settings=settings,
    )


def hmm_cluster(
    reads: Sequence[Any],
    cluster: SignalCluster,
    **kwargs: Any,
) -> ProfileHMMConsensus:
    """Fit one HMM consensus from the shared clustering result."""
    selected, local_medoid = cluster.select(reads)
    return hmm_consensus(selected, medoid_index=local_medoid, **kwargs)


def hmm_transition_matrix(
    length: int,
    transition_offsets: npt.ArrayLike,
    transition_prob: npt.ArrayLike,
) -> FloatArr:
    """Return the row-stochastic profile-state transition matrix.

    ``transition_prob`` supplies global displacement weights.  At a boundary,
    weights for illegal targets are removed and the remaining legal moves are
    renormalized.  Thus every returned row sums to one instead of silently losing
    probability mass at the profile ends.
    """
    profile_length = _positive_integer(length, name="length")
    if profile_length < 1:
        raise ValueError("length must be a positive integer")

    offsets = _coerce_offsets(transition_offsets)
    probabilities = np.asarray(transition_prob, dtype=float)
    if probabilities.shape != offsets.shape:
        raise ValueError("transition_prob must match transition_offsets")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("transition_prob must contain finite positive values")
    total_probability = float(probabilities.sum())
    if not np.isfinite(total_probability) or total_probability <= 0.0:
        raise ValueError("transition_prob must have a finite positive sum")

    transition = np.zeros((profile_length, profile_length), dtype=float)
    for offset, probability in zip(offsets, probabilities):
        source, target = _legal_edges(profile_length, int(offset))
        transition[source, target] = float(probability)

    row_sum = transition.sum(axis=1)
    if not np.all(np.isfinite(row_sum)) or np.any(row_sum <= 0.0):
        raise ValueError("transition offsets leave a profile state with no legal move")
    transition /= row_sum[:, None]
    if not np.all(np.isfinite(transition)) or not np.allclose(
        transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise FloatingPointError("failed to construct a row-stochastic transition")
    return transition


def score_hmm_profile(
    read: Any,
    profile: ProfileHMMConsensus,
    *,
    normalize: bool | None = None,
    return_posterior: bool = False,
) -> ProfileHMMScore:
    """Score one read against an immutable, already-fitted profile HMM.

    This operation performs forward-backward only: it never selects a medoid,
    updates a profile parameter, or appends to the training likelihood trace.
    The returned log likelihood is raw and higher is better.  By default the
    read uses the fit's recorded normalization choice; callers may override it
    when a frozen upstream transform has already been applied.
    """
    mean, std, offsets, probabilities = _validated_profile_parameters(profile)
    settings = profile.settings
    use_normalization = settings.normalize if normalize is None else bool(normalize)
    prepared = prepare_reads(
        [read], normalize=use_normalization, min_std=settings.min_std
    )[0]
    expectation = _expectation(
        prepared,
        mean,
        std,
        _log_transition_matrix(mean.size, offsets, probabilities),
        offsets,
    )
    posterior: FloatArr | None = None
    if return_posterior:
        posterior = expectation.gamma.copy()
        posterior.setflags(write=False)
    return ProfileHMMScore(
        log_likelihood=expectation.log_likelihood,
        posterior=posterior,
    )


def snapshot_hmm_profile(
    profile: ProfileHMMConsensus,
    iteration: int,
    training_reads: Sequence[Any],
) -> ProfileHMMConsensus:
    """Reconstruct the model after exactly ``iteration`` completed updates.

    History index zero is the initial model before any update.  History index
    ``k`` and ``log_likelihood[k]`` describe the model after exactly ``k``
    completed M-step updates, while update-diagnostic index ``k - 1`` describes
    the change that produced it.  Consequently an iteration-100 checkpoint uses
    parameter-history index 100, likelihood index 100, and delta index 99.

    The requested iteration must actually exist; a shorter converged trajectory
    is never clamped and relabelled as a later checkpoint.  Posterior-derived
    depth, standard error, support, and aggregate likelihood are recomputed from
    ``training_reads`` so they describe the checkpoint parameters rather than
    the final model.
    """

    checkpoint = int(iteration)
    if checkpoint < 0 or checkpoint > profile.n_iter:
        raise ValueError(
            "iteration checkpoint is unavailable; expected "
            f"0 <= iteration <= {profile.n_iter}"
        )
    if len(training_reads) == 0:
        raise ValueError("training_reads must be non-empty")

    expected_history = profile.n_iter + 1
    histories = (
        profile.mean_history,
        profile.std_history,
        profile.dwell_history,
        profile.transition_prob_history,
        profile.log_likelihood,
    )
    if any(values.shape[0] != expected_history for values in histories):
        raise ValueError(
            "profile parameter and likelihood histories must contain the initial "
            "model plus every completed update"
        )
    update_diagnostics = (
        profile.mean_delta,
        profile.std_delta,
        profile.transition_delta,
        profile.parameter_delta,
        profile.transition_data_objective_before,
        profile.transition_data_objective_after,
        profile.transition_prior_objective_before,
        profile.transition_prior_objective_after,
        profile.transition_penalized_objective_before,
        profile.transition_penalized_objective_after,
        profile.transition_gradient_norm,
        profile.transition_optimizer_success,
        profile.emission_objective_before,
        profile.emission_objective_after,
    )
    if any(values.shape != (profile.n_iter,) for values in update_diagnostics):
        raise ValueError(
            "profile update diagnostics must contain one value per completed update"
        )

    is_final = checkpoint == profile.n_iter
    snapshot = replace(
        profile,
        mean=profile.mean_history[checkpoint].copy(),
        std=profile.std_history[checkpoint].copy(),
        stderr=np.full(profile.mean.size, np.inf, dtype=float),
        dwell=profile.dwell_history[checkpoint].copy(),
        depth=np.zeros(profile.mean.size, dtype=float),
        supported=np.zeros(profile.mean.size, dtype=bool),
        transition_offsets=profile.transition_offsets.copy(),
        transition_prob=profile.transition_prob_history[checkpoint].copy(),
        mean_history=profile.mean_history[: checkpoint + 1].copy(),
        std_history=profile.std_history[: checkpoint + 1].copy(),
        dwell_history=profile.dwell_history[: checkpoint + 1].copy(),
        transition_prob_history=profile.transition_prob_history[
            : checkpoint + 1
        ].copy(),
        log_likelihood=profile.log_likelihood[: checkpoint + 1].copy(),
        n_iter=checkpoint,
        converged=profile.converged if is_final else False,
        mean_delta=profile.mean_delta[:checkpoint].copy(),
        std_delta=profile.std_delta[:checkpoint].copy(),
        transition_delta=profile.transition_delta[:checkpoint].copy(),
        parameter_delta=profile.parameter_delta[:checkpoint].copy(),
        transition_data_objective_before=profile.transition_data_objective_before[
            :checkpoint
        ].copy(),
        transition_data_objective_after=profile.transition_data_objective_after[
            :checkpoint
        ].copy(),
        transition_prior_objective_before=profile.transition_prior_objective_before[
            :checkpoint
        ].copy(),
        transition_prior_objective_after=profile.transition_prior_objective_after[
            :checkpoint
        ].copy(),
        transition_penalized_objective_before=(
            profile.transition_penalized_objective_before[:checkpoint].copy()
        ),
        transition_penalized_objective_after=(
            profile.transition_penalized_objective_after[:checkpoint].copy()
        ),
        transition_gradient_norm=profile.transition_gradient_norm[:checkpoint].copy(),
        transition_optimizer_success=profile.transition_optimizer_success[
            :checkpoint
        ].copy(),
        emission_objective_before=profile.emission_objective_before[:checkpoint].copy(),
        emission_objective_after=profile.emission_objective_after[:checkpoint].copy(),
        stop_reason=profile.stop_reason if is_final else "checkpoint",
    )

    prepared = prepare_reads(
        training_reads,
        normalize=profile.settings.normalize,
        min_std=profile.settings.min_std,
    )
    log_transition = _log_transition_matrix(
        snapshot.mean.size,
        snapshot.transition_offsets,
        snapshot.transition_prob,
    )
    depth = np.zeros(snapshot.mean.size, dtype=float)
    precision = np.zeros(snapshot.mean.size, dtype=float)
    total_log_likelihood = 0.0
    for read in prepared:
        expectation = _expectation(
            read,
            snapshot.mean,
            snapshot.std,
            log_transition,
            snapshot.transition_offsets,
        )
        total_log_likelihood += expectation.log_likelihood
        occupancy = expectation.gamma.sum(axis=0)
        depth += np.minimum(occupancy, 1.0)
        variance = read.std[:, None] ** 2 + snapshot.std[None, :] ** 2
        precision += (expectation.gamma / variance).sum(axis=0)
    stderr = np.full(snapshot.mean.size, np.inf, dtype=float)
    updated = precision > 0.0
    stderr[updated] = np.sqrt(1.0 / precision[updated])
    supported = depth >= profile.settings.min_depth
    recorded_log_likelihood = float(snapshot.log_likelihood[-1])
    if not np.isclose(
        total_log_likelihood,
        recorded_log_likelihood,
        rtol=1e-10,
        atol=1e-8,
    ):
        raise ValueError(
            "training_reads do not reproduce the checkpoint aggregate likelihood"
        )
    return replace(
        snapshot,
        stderr=stderr,
        depth=depth,
        supported=supported,
    )


def _validate_fit_options(
    *,
    max_iter: int,
    tol: float,
    min_depth: int,
    min_std: float,
    transition_pseudocount: float,
) -> None:
    if _positive_integer(max_iter, name="max_iter") < 1:
        raise ValueError("max_iter must be >= 1")
    if tol < 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and non-negative")
    if _positive_integer(min_depth, name="min_depth") < 1:
        raise ValueError("min_depth must be >= 1")
    if min_std <= 0.0 or not np.isfinite(min_std):
        raise ValueError("min_std must be finite and positive")
    min_variance = float(min_std) * float(min_std)
    if min_variance == 0.0 or not np.isfinite(min_variance):
        raise ValueError("min_std is outside the numerically safe range")
    if transition_pseudocount <= 0.0 or not np.isfinite(
        transition_pseudocount
    ):
        raise ValueError("transition_pseudocount must be finite and positive")


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _coerce_offsets(offsets: npt.ArrayLike) -> Int64Arr:
    raw = np.asarray(offsets)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("transition_offsets must be a non-empty sequence")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("transition_offsets must contain integers") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
        raise ValueError("transition_offsets must contain finite integers")
    int64_info = np.iinfo(np.int64)
    if np.any(numeric < int64_info.min) or np.any(numeric > int64_info.max):
        raise ValueError("transition_offsets are outside the int64 range")
    values = np.asarray(numeric, dtype=np.int64)
    if np.unique(values).size != values.size:
        raise ValueError("transition_offsets must be unique")
    # Preserve caller order because each probability is paired positionally with
    # its displacement. The default is already in increasing offset order.
    return values


def _validate_offsets(offsets: npt.ArrayLike) -> Int64Arr:
    values = _coerce_offsets(offsets)
    if (
        not np.any(values < 0)
        or 0 not in values
        or 1 not in values
        or not np.any(values > 1)
    ):
        raise ValueError(
            "transition_offsets must allow backsteps, holds, +1 steps, and skips"
        )
    return values


def _initial_transition_prob(
    offsets: Int64Arr,
    length: int,
    *,
    transition_prob: npt.ArrayLike | None,
    pseudocount: float,
) -> FloatArr:
    if transition_prob is None:
        counts = np.full(offsets.size, pseudocount, dtype=float)
        counts[int(np.flatnonzero(offsets == 1)[0])] += max(length - 1, 1)
        count_sum = float(counts.sum())
        if not np.all(np.isfinite(counts)) or not np.isfinite(count_sum):
            raise FloatingPointError("initial transition weights are non-finite")
        return counts / count_sum

    probabilities = np.asarray(transition_prob, dtype=float)
    if probabilities.shape != offsets.shape:
        raise ValueError("transition_prob must match transition_offsets")
    probability_sum = float(probabilities.sum())
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or not np.isfinite(probability_sum)
        or probability_sum <= 0.0
    ):
        raise ValueError("transition_prob must contain finite positive values")
    return probabilities / probability_sum


def _legal_edges(length: int, offset: int) -> tuple[Int64Arr, Int64Arr]:
    """Return legal source/target arrays without integer-overflow arithmetic."""
    if offset >= length or offset <= -length:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    if offset >= 0:
        source = np.arange(0, length - offset, dtype=np.int64)
    else:
        source = np.arange(-offset, length, dtype=np.int64)
    return source, source + offset


def _log_transition_matrix(
    length: int,
    offsets: Int64Arr,
    probabilities: FloatArr,
) -> FloatArr:
    transition = hmm_transition_matrix(length, offsets, probabilities)
    with np.errstate(divide="ignore"):
        return np.log(transition)


def _validated_profile_parameters(
    profile: ProfileHMMConsensus,
) -> tuple[FloatArr, FloatArr, Int64Arr, FloatArr]:
    mean = np.asarray(profile.mean, dtype=float)
    std = np.asarray(profile.std, dtype=float)
    if mean.ndim != 1 or mean.size == 0 or std.shape != mean.shape:
        raise ValueError("profile mean/std must be matching non-empty vectors")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("profile mean/std contains NaN or infinity")
    if np.any(std <= 0.0):
        raise ValueError("profile std must be positive")

    offsets = _coerce_offsets(profile.transition_offsets)
    probabilities = np.asarray(profile.transition_prob, dtype=float)
    # The public constructor performs all probability and boundary validation.
    hmm_transition_matrix(mean.size, offsets, probabilities)
    return mean, std, offsets, probabilities


def _fit_expectation_pass(
    reads: Sequence[SignalSteps],
    profile_mean: FloatArr,
    profile_std: FloatArr,
    log_transition: FloatArr,
    offsets: Int64Arr,
) -> _FitPass:
    length = profile_mean.size
    sum_precision = np.zeros(length, dtype=float)
    responsibility = np.zeros(length, dtype=float)
    sum_dwell = np.zeros(length, dtype=float)
    depth = np.zeros(length, dtype=float)
    source_offset_counts = np.zeros((length, offsets.size), dtype=float)
    expectations: list[tuple[SignalSteps, FloatArr]] = []
    total_log_likelihood = 0.0

    for read in reads:
        stats = _expectation(
            read, profile_mean, profile_std, log_transition, offsets
        )
        gamma = stats.gamma
        total_log_likelihood += stats.log_likelihood
        source_offset_counts += stats.source_offset_counts
        expectations.append((read, gamma))

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            variance = read.std[:, None] ** 2 + profile_std[None, :] ** 2
            precision = gamma / variance
        if (
            not np.all(np.isfinite(variance))
            or np.any(variance <= 0.0)
            or not np.all(np.isfinite(precision))
        ):
            raise FloatingPointError("profile-HMM precision is non-finite")
        sum_precision += precision.sum(axis=0)
        occupancy = gamma.sum(axis=0)
        responsibility += occupancy
        sum_dwell += (gamma * read.dwell[:, None]).sum(axis=0)
        depth += np.minimum(occupancy, 1.0)

    accumulated = (
        np.asarray([total_log_likelihood]),
        sum_precision,
        responsibility,
        sum_dwell,
        depth,
        source_offset_counts,
    )
    if any(not np.all(np.isfinite(values)) for values in accumulated):
        raise FloatingPointError("profile-HMM E-pass produced non-finite statistics")
    return _FitPass(
        log_likelihood=float(total_log_likelihood),
        expectations=tuple(expectations),
        sum_precision=sum_precision,
        responsibility=responsibility,
        sum_dwell=sum_dwell,
        depth=depth,
        source_offset_counts=source_offset_counts,
    )


def _legal_move_mask(length: int, offsets: Int64Arr) -> BoolArr:
    """Return whether each source-state/displacement pair is legal."""

    mask = np.zeros((length, offsets.size), dtype=bool)
    for offset_index, offset in enumerate(offsets):
        source, _ = _legal_edges(length, int(offset))
        mask[source, offset_index] = True
    if np.any(~np.any(mask, axis=1)):
        raise ValueError("transition offsets leave a profile state with no legal move")
    return mask


def _transition_objective_and_gradient(
    free_logits: FloatArr,
    source_offset_counts: FloatArr,
    legal_move_mask: BoolArr,
    pseudocount: float,
) -> tuple[float, FloatArr, float, float]:
    """Return penalized boundary-aware transition Q and its free-logit gradient.

    The final move logit is fixed at zero to remove the additive
    non-identifiability.  ``pseudocount`` is a single symmetric global
    pseudo-count per move, equivalent to adding it to aggregate counts when all
    source rows have the same legal moves.
    """

    counts = np.asarray(source_offset_counts, dtype=float)
    legal = np.asarray(legal_move_mask, dtype=bool)
    candidate = np.asarray(free_logits, dtype=np.float64)
    if counts.ndim != 2 or legal.shape != counts.shape:
        raise ValueError("source transition counts and legal mask must match")
    if candidate.shape != (counts.shape[1] - 1,):
        raise ValueError("free transition logits have the wrong shape")
    if (
        not np.all(np.isfinite(counts))
        or np.any(counts < 0.0)
        or np.any(counts[~legal] != 0.0)
        or not np.all(np.isfinite(candidate))
    ):
        raise ValueError("transition objective inputs are invalid")
    if not np.isfinite(pseudocount) or pseudocount < 0.0:
        raise ValueError("pseudocount must be finite and non-negative")
    if np.any(~np.any(legal, axis=1)):
        raise ValueError("every source state needs at least one legal move")

    logits = np.concatenate((candidate, np.zeros(1, dtype=float)))
    legal_logits = np.where(legal, logits[None, :], -np.inf)
    row_log_normalizer = np.asarray(
        logsumexp(legal_logits, axis=1), dtype=np.float64
    )
    source_total = counts.sum(axis=1)
    data_objective = float(
        np.sum(counts * logits[None, :])
        - np.dot(source_total, row_log_normalizer)
    )

    global_log_normalizer = np.asarray(
        logsumexp(logits), dtype=np.float64
    ).item()
    log_weights = logits - global_log_normalizer
    prior_objective = float(pseudocount * np.sum(log_weights))

    row_probability = np.zeros_like(counts)
    row_probability[legal] = np.exp(
        legal_logits[legal]
        - np.repeat(row_log_normalizer, legal.sum(axis=1))
    )
    data_gradient = counts.sum(axis=0) - (
        source_total[:, None] * row_probability
    ).sum(axis=0)
    weights = np.exp(log_weights)
    prior_gradient = pseudocount * (1.0 - counts.shape[1] * weights)
    full_gradient = data_gradient + prior_gradient
    penalized = data_objective + prior_objective
    free_gradient = np.asarray(full_gradient[:-1], dtype=np.float64)
    if not np.isfinite(penalized) or not np.all(np.isfinite(free_gradient)):
        raise FloatingPointError("transition objective is non-finite")
    return penalized, free_gradient, data_objective, prior_objective


def _transition_objective_hessian(
    free_logits: FloatArr,
    source_offset_counts: FloatArr,
    legal_move_mask: BoolArr,
    pseudocount: float,
) -> FloatArr:
    """Return the free-logit Hessian of the concave transition objective."""

    counts = np.asarray(source_offset_counts, dtype=float)
    legal = np.asarray(legal_move_mask, dtype=bool)
    logits = np.concatenate(
        (np.asarray(free_logits, dtype=np.float64), np.zeros(1, dtype=np.float64))
    )
    legal_logits = np.where(legal, logits[None, :], -np.inf)
    row_log_normalizer = np.asarray(
        logsumexp(legal_logits, axis=1), dtype=np.float64
    )
    row_probability = np.zeros_like(counts)
    row_probability[legal] = np.exp(
        legal_logits[legal]
        - np.repeat(row_log_normalizer, legal.sum(axis=1))
    )
    hessian = np.zeros((logits.size, logits.size), dtype=float)
    for source, total in enumerate(counts.sum(axis=1)):
        probability = row_probability[source]
        hessian -= total * (
            np.diag(probability) - np.outer(probability, probability)
        )
    global_log_normalizer = np.asarray(
        logsumexp(logits), dtype=np.float64
    ).item()
    weights = np.exp(logits - global_log_normalizer)
    hessian -= pseudocount * logits.size * (
        np.diag(weights) - np.outer(weights, weights)
    )
    free_hessian = np.asarray(hessian[:-1, :-1], dtype=np.float64)
    if not np.all(np.isfinite(free_hessian)):
        raise FloatingPointError("transition objective Hessian is non-finite")
    return free_hessian


def _refine_transition_logits(
    initial_logits: FloatArr,
    source_offset_counts: FloatArr,
    legal_move_mask: BoolArr,
    pseudocount: float,
) -> FloatArr:
    """Newton-refine a transition solution using monotone backtracking."""

    logits = np.asarray(initial_logits, dtype=np.float64).copy()
    for _ in range(50):
        objective, gradient, _, _ = _transition_objective_and_gradient(
            logits, source_offset_counts, legal_move_mask, pseudocount
        )
        if float(np.max(np.abs(gradient))) <= 1e-9:
            break
        hessian = _transition_objective_hessian(
            logits, source_offset_counts, legal_move_mask, pseudocount
        )
        try:
            direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            direction = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        directional_derivative = float(np.dot(gradient, direction))
        if (
            not np.all(np.isfinite(direction))
            or not np.isfinite(directional_derivative)
            or directional_derivative <= 0.0
        ):
            break
        step = 1.0
        accepted = False
        while step >= 2.0 ** -30:
            candidate = np.asarray(logits + step * direction, dtype=np.float64)
            candidate_objective = _transition_objective_and_gradient(
                candidate,
                source_offset_counts,
                legal_move_mask,
                pseudocount,
            )[0]
            if candidate_objective >= (
                objective + 1e-4 * step * directional_derivative
            ):
                logits = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return np.asarray(logits, dtype=np.float64)


def _optimize_transition_probabilities(
    source_offset_counts: FloatArr,
    legal_move_mask: BoolArr,
    probabilities: FloatArr,
    *,
    pseudocount: float,
) -> _TransitionUpdate:
    """Maximize the boundary-aware transition auxiliary objective."""

    current = np.asarray(probabilities, dtype=float)
    if (
        current.ndim != 1
        or current.size < 2
        or current.size != source_offset_counts.shape[1]
        or not np.all(np.isfinite(current))
        or np.any(current <= 0.0)
    ):
        raise ValueError("current transition probabilities are invalid")
    current = current / float(current.sum())
    initial_logits = np.log(current[:-1]) - np.log(current[-1])
    before, _, data_before, prior_before = _transition_objective_and_gradient(
        initial_logits,
        source_offset_counts,
        legal_move_mask,
        pseudocount,
    )

    def objective(values: FloatArr) -> float:
        value, _, _, _ = _transition_objective_and_gradient(
            np.asarray(values, dtype=np.float64),
            source_offset_counts,
            legal_move_mask,
            pseudocount,
        )
        return -value

    def gradient(values: FloatArr) -> FloatArr:
        _, derivative, _, _ = _transition_objective_and_gradient(
            np.asarray(values, dtype=np.float64),
            source_offset_counts,
            legal_move_mask,
            pseudocount,
        )
        return -derivative

    optimized = minimize(
        objective,
        initial_logits,
        method="L-BFGS-B",
        jac=gradient,
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 500, "maxls": 50},
    )
    optimized_logits = _refine_transition_logits(
        np.asarray(optimized.x, dtype=np.float64),
        source_offset_counts,
        legal_move_mask,
        pseudocount,
    )
    after, derivative, data_after, prior_after = (
        _transition_objective_and_gradient(
            optimized_logits,
            source_offset_counts,
            legal_move_mask,
            pseudocount,
        )
    )
    raw_gradient_norm = float(np.max(np.abs(derivative)))
    gradient_scale = max(
        1.0,
        float(np.sum(source_offset_counts))
        + pseudocount * source_offset_counts.shape[1],
    )
    gradient_norm = raw_gradient_norm / gradient_scale
    success = gradient_norm <= 1e-8 and after >= before - 1e-9
    if not success:
        raise RuntimeError(
            "boundary-aware transition optimization failed: "
            f"status={optimized.status}, scipy_success={optimized.success}, "
            f"objective_change={after - before:.6g}, "
            f"normalized_gradient={gradient_norm:.6g}, "
            f"raw_gradient={raw_gradient_norm:.6g}"
        )

    logits = np.concatenate((optimized_logits, np.zeros(1, dtype=float)))
    global_log_normalizer = np.asarray(
        logsumexp(logits), dtype=np.float64
    ).item()
    new_probabilities = np.exp(logits - global_log_normalizer)
    if (
        not np.all(np.isfinite(new_probabilities))
        or np.any(new_probabilities <= 0.0)
        or not np.isclose(new_probabilities.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise FloatingPointError(
            "boundary-aware transition optimizer returned invalid probabilities"
        )
    return _TransitionUpdate(
        probabilities=np.asarray(new_probabilities, dtype=float),
        data_objective_before=data_before,
        data_objective_after=data_after,
        prior_objective_before=prior_before,
        prior_objective_after=prior_after,
        penalized_objective_before=before,
        penalized_objective_after=after,
        gradient_norm=gradient_norm,
        success=success,
    )


def _emission_auxiliary_objective(
    expectations: Sequence[tuple[SignalSteps, FloatArr]],
    profile_mean: FloatArr,
    profile_std: FloatArr,
) -> float:
    """Evaluate fixed-posterior Gaussian-convolution emission Q."""

    total = 0.0
    for read, gamma in expectations:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            score = emission(read.mean, read.std, profile_mean, profile_std)
        if not np.all(np.isfinite(score)):
            raise FloatingPointError("profile-HMM emission objective is non-finite")
        total += float(np.sum(gamma * score))
    if not np.isfinite(total):
        raise FloatingPointError("profile-HMM emission objective is non-finite")
    return total


def _profiled_emission_state_objective(
    variance: float,
    observed_mean: FloatArr,
    observed_variance: FloatArr,
    responsibility: FloatArr,
) -> tuple[float, float]:
    """Return one state's emission Q and optimal mean at fixed variance."""

    if variance <= 0.0 or not np.isfinite(variance):
        raise ValueError("profile variance must be finite and positive")
    total_variance = observed_variance + variance
    precision_weight = responsibility / total_variance
    precision = float(np.sum(precision_weight))
    if not np.isfinite(precision) or precision <= 0.0:
        raise ValueError("profile state has no finite posterior precision")
    mean = float(np.dot(precision_weight, observed_mean) / precision)
    residual = observed_mean - mean
    objective = -0.5 * float(
        np.sum(
            responsibility
            * (
                np.log(2.0 * np.pi)
                + np.log(total_variance)
                + residual * residual / total_variance
            )
        )
    )
    if not np.isfinite(mean) or not np.isfinite(objective):
        raise FloatingPointError("profile-state emission objective is non-finite")
    return objective, mean


def _profiled_emission_variance_derivative(
    variance: float,
    observed_mean: FloatArr,
    observed_variance: FloatArr,
    responsibility: FloatArr,
) -> float:
    """Derivative of profiled emission Q with respect to latent variance."""

    _, mean = _profiled_emission_state_objective(
        variance, observed_mean, observed_variance, responsibility
    )
    total_variance = observed_variance + variance
    residual = observed_mean - mean
    derivative = 0.5 * float(
        np.sum(
            responsibility
            * (
                residual * residual / (total_variance * total_variance)
                - 1.0 / total_variance
            )
        )
    )
    if not np.isfinite(derivative):
        raise FloatingPointError("profile-state variance derivative is non-finite")
    return derivative


def _optimize_emission_state(
    observed_mean: FloatArr,
    observed_variance: FloatArr,
    responsibility: FloatArr,
    *,
    current_mean: float,
    current_variance: float,
    min_variance: float,
) -> _EmissionStateUpdate:
    """Globally bracket and optimize one state's profiled emission objective."""

    values = np.asarray(observed_mean, dtype=float)
    measurement_variance = np.asarray(observed_variance, dtype=float)
    weights = np.asarray(responsibility, dtype=float)
    if (
        values.ndim != 1
        or measurement_variance.shape != values.shape
        or weights.shape != values.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(measurement_variance))
        or np.any(measurement_variance <= 0.0)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("profile-state emission inputs are invalid")
    occupancy = float(np.sum(weights))
    occupancy_tolerance = np.finfo(float).eps * max(1, values.size)
    if occupancy <= occupancy_tolerance:
        return _EmissionStateUpdate(
            mean=float(current_mean),
            variance=float(current_variance),
            objective=0.0,
            success=True,
        )
    if (
        min_variance <= 0.0
        or current_variance < min_variance
        or not np.isfinite(min_variance)
        or not np.isfinite(current_variance)
    ):
        raise ValueError("profile-state variance bounds are invalid")

    weighted_mean = float(np.dot(weights, values) / occupancy)
    empirical_variance = float(
        np.dot(weights, (values - weighted_mean) ** 2) / occupancy
    )
    upper_variance = max(
        min_variance * 16.0,
        current_variance * 4.0,
        empirical_variance * 4.0,
    )
    for _ in range(64):
        derivative = _profiled_emission_variance_derivative(
            upper_variance, values, measurement_variance, weights
        )
        if derivative <= 0.0:
            break
        upper_variance *= 4.0
        if not np.isfinite(upper_variance):
            raise FloatingPointError("could not bracket profile-state variance")
    else:
        raise RuntimeError("could not bracket profile-state variance optimum")

    lower_log_variance = float(np.log(min_variance))
    upper_log_variance = float(np.log(upper_variance))

    def negative_profiled_objective(log_variance: float) -> float:
        variance = float(np.exp(log_variance))
        objective, _ = _profiled_emission_state_objective(
            variance, values, measurement_variance, weights
        )
        return -objective

    optimized = minimize_scalar(
        negative_profiled_objective,
        bounds=(lower_log_variance, upper_log_variance),
        method="bounded",
        options={"xatol": 1e-11, "maxiter": 500},
    )
    candidates = (
        min_variance,
        current_variance,
        float(np.exp(float(optimized.x))),
    )
    evaluated = [
        (
            *_profiled_emission_state_objective(
                variance, values, measurement_variance, weights
            ),
            variance,
        )
        for variance in candidates
    ]
    objective, mean, variance = max(evaluated, key=lambda item: item[0])
    success = bool(optimized.success and np.isfinite(objective))
    if not success:
        raise RuntimeError("profile-state emission optimization failed")
    return _EmissionStateUpdate(
        mean=float(mean),
        variance=float(variance),
        objective=float(objective),
        success=success,
    )


def _optimize_emission_parameters(
    fit_pass: _FitPass,
    mean: FloatArr,
    std: FloatArr,
    *,
    min_std: float,
) -> tuple[FloatArr, FloatArr]:
    """Optimize all state means/spreads for the fixed E-step posterior."""

    all_mean = np.concatenate(
        [read.mean for read, _ in fit_pass.expectations]
    )
    all_variance = np.concatenate(
        [read.std * read.std for read, _ in fit_pass.expectations]
    )
    all_gamma = np.concatenate(
        [gamma for _, gamma in fit_pass.expectations], axis=0
    )
    new_mean = mean.copy()
    new_std = std.copy()
    min_variance = min_std * min_std
    for state in range(mean.size):
        update = _optimize_emission_state(
            all_mean,
            all_variance,
            all_gamma[:, state],
            current_mean=float(mean[state]),
            current_variance=float(std[state] * std[state]),
            min_variance=min_variance,
        )
        new_mean[state] = update.mean
        new_std[state] = np.sqrt(update.variance)
    return new_mean, new_std


def _maximization(
    fit_pass: _FitPass,
    mean: FloatArr,
    std: FloatArr,
    dwell: FloatArr,
    offsets: Int64Arr,
    probabilities: FloatArr,
    *,
    min_std: float,
    transition_pseudocount: float,
) -> _Maximization:
    length = mean.size
    new_mean, new_std = _optimize_emission_parameters(
        fit_pass, mean, std, min_std=min_std
    )
    new_dwell = dwell.copy()
    occupied = fit_pass.responsibility > 0.0
    new_dwell[occupied] = (
        fit_pass.sum_dwell[occupied] / fit_pass.responsibility[occupied]
    )

    transition = _optimize_transition_probabilities(
        fit_pass.source_offset_counts,
        _legal_move_mask(length, offsets),
        probabilities,
        pseudocount=transition_pseudocount,
    )
    new_probabilities = transition.probabilities
    emission_before = _emission_auxiliary_objective(
        fit_pass.expectations, mean, std
    )
    emission_after = _emission_auxiliary_objective(
        fit_pass.expectations, new_mean, new_std
    )

    parameters = (new_mean, new_std, new_dwell, new_probabilities)
    if any(not np.all(np.isfinite(values)) for values in parameters):
        raise FloatingPointError("profile-HMM parameter update is non-finite")
    if np.any(new_std <= 0.0) or np.any(new_probabilities <= 0.0):
        raise FloatingPointError("profile-HMM update produced invalid parameters")
    return _Maximization(
        mean=new_mean,
        std=new_std,
        dwell=new_dwell,
        transition_prob=new_probabilities,
        transition=transition,
        emission_objective_before=emission_before,
        emission_objective_after=emission_after,
    )


def _expectation(
    read: SignalSteps,
    profile_mean: FloatArr,
    profile_std: FloatArr,
    log_transition: FloatArr,
    offsets: Int64Arr,
) -> _Expectation:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = emission(read.mean, read.std, profile_mean, profile_std)
    if not np.all(np.isfinite(score)):
        raise FloatingPointError("profile-HMM emission score is non-finite")
    n_steps, length = score.shape
    if log_transition.shape != (length, length):
        raise ValueError("log transition shape does not match the profile")
    if np.any(np.isnan(log_transition)) or np.any(np.isposinf(log_transition)):
        raise ValueError("log transition contains an invalid value")

    alpha = np.full((n_steps, length), -np.inf, dtype=float)
    alpha[0] = score[0] - np.log(float(length))
    for step in range(1, n_steps):
        alpha[step] = score[step] + logsumexp(
            alpha[step - 1][:, None] + log_transition, axis=0
        )

    beta = np.full((n_steps, length), -np.inf, dtype=float)
    beta[-1] = 0.0
    for step in range(n_steps - 2, -1, -1):
        beta[step] = logsumexp(
            log_transition
            + (score[step + 1] + beta[step + 1])[None, :],
            axis=1,
        )

    log_likelihood = np.asarray(logsumexp(alpha[-1]), dtype=float).item()
    if not np.isfinite(log_likelihood):
        raise FloatingPointError("profile-HMM log likelihood is non-finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gamma = np.exp(alpha + beta - log_likelihood)
    gamma_sum = gamma.sum(axis=1, keepdims=True)
    if (
        not np.all(np.isfinite(gamma))
        or not np.all(np.isfinite(gamma_sum))
        or np.any(gamma_sum <= 0.0)
    ):
        raise FloatingPointError("profile-HMM posterior is non-finite")
    gamma /= gamma_sum

    source_offset_counts = np.zeros((length, offsets.size), dtype=float)
    for step in range(n_steps - 1):
        for offset_index, offset in enumerate(offsets):
            source, target = _legal_edges(length, int(offset))
            log_xi = (
                alpha[step, source]
                + log_transition[source, target]
                + score[step + 1, target]
                + beta[step + 1, target]
                - log_likelihood
            )
            source_offset_counts[source, offset_index] += np.exp(log_xi)
    if not np.all(np.isfinite(source_offset_counts)):
        raise FloatingPointError("profile-HMM transition posterior is non-finite")

    return _Expectation(
        log_likelihood=log_likelihood,
        gamma=gamma,
        source_offset_counts=source_offset_counts,
    )
