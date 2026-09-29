"""Shared DTW clustering plus DBA/profile-HMM signal consensus."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import cowler.consensus.hmm as hmm_module
from cowler.consensus.cluster import (
    cluster_from_distances,
    cluster_reads_dtw,
    medoid,
    rank_medoid_candidates,
)
from cowler.consensus.dba import dba, dba_cluster
from cowler.consensus.hmm import (
    hmm_cluster,
    hmm_consensus,
    hmm_transition_matrix,
    score_hmm_profile,
    snapshot_hmm_profile,
)


def test_medoid_and_negative_gaussian_distance_clustering():
    distances = np.array(
        [
            [0.0, -2.0, 5.0, 6.0],
            [-2.0, 0.0, 4.0, 5.0],
            [5.0, 4.0, 0.0, -1.0],
            [6.0, 5.0, -1.0, 0.0],
        ]
    )
    result = cluster_from_distances(distances, n_clusters=2)

    assert result.distance_shift == pytest.approx(2.0)
    assert [cluster.members.tolist() for cluster in result.clusters] == [[0, 1], [2, 3]]
    assert [cluster.medoid_index for cluster in result.clusters] == [0, 2]
    assert medoid(distances, [1, 0]) == 0


def test_medoid_candidate_ranking_uses_group_scores_and_original_index_ties():
    distances = np.array(
        [
            [0.0, 1.0, 8.0, 1.0],
            [1.0, 0.0, 7.0, 2.0],
            [8.0, 7.0, 0.0, 7.0],
            [1.0, 2.0, 7.0, 0.0],
        ]
    )

    ranking = rank_medoid_candidates(distances, [3, 1, 0], top_k=None)

    assert [candidate.index for candidate in ranking] == [0, 1, 3]
    assert [candidate.row_sum for candidate in ranking] == pytest.approx([2.0, 3.0, 3.0])
    assert medoid(distances, [3, 1, 0]) == ranking[0].index
    assert rank_medoid_candidates(distances, [3, 1, 0], top_k=2) == ranking[:2]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_medoid_candidate_ranking_rejects_invalid_top_k(top_k: Any) -> None:
    with pytest.raises(ValueError, match="top_k"):
        rank_medoid_candidates(np.zeros((2, 2)), top_k=top_k)


def test_dtw_clustering_recovers_two_signal_shapes():
    rng = np.random.default_rng(4)
    rising = np.array([-2.0, -1.5, -0.5, 0.5, 1.5, 2.0, 1.0, 0.0])
    alternating = np.array([-1.5, 1.5, -1.2, 1.2, -0.9, 0.9, -0.6, 0.6])
    reads = [
        (template + rng.normal(0.0, 0.04, template.size), np.full(template.size, 0.15))
        for template in (rising, rising, rising, alternating, alternating, alternating)
    ]
    result = cluster_reads_dtw(reads, n_clusters=2, max_run=2)
    groups = {frozenset(cluster.members.tolist()) for cluster in result.clusters}
    assert groups == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}


def test_dba_refines_a_noisy_medoid_and_tracks_read_depth():
    rng = np.random.default_rng(10)
    truth = np.array([-2.0, -1.0, 0.5, 1.8, 0.2, -1.5, -0.4, 1.3, 2.1])
    medoid_read = truth + np.array([0.25, -0.2, 0.2, -0.2, 0.25, -0.2, 0.2, -0.2, 0.25])
    means = [medoid_read] + [
        truth + rng.normal(0.0, 0.10, truth.size) for _ in range(7)
    ]
    reads = [
        (mean, np.full(truth.size, 0.12), np.full(truth.size, 5.0))
        for mean in means
    ]

    result = dba(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=10,
        tol=1e-6,
        open_begin=False,
        open_end=False,
    )

    medoid_rmse = float(np.sqrt(np.mean((medoid_read - truth) ** 2)))
    consensus_rmse = float(np.sqrt(np.mean((result.mean - truth) ** 2)))
    assert consensus_rmse < medoid_rmse * 0.5
    assert np.array_equal(result.depth, np.full(truth.size, len(reads)))
    assert np.all(result.supported)
    assert result.n_iter <= 10
    assert result.delta.shape == (result.n_iter,)
    assert result.delta[-1] <= 1e-6


def test_dba_defaults_to_closed_corner_to_corner_dtw():
    profile = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    reads = [
        (profile, np.full(profile.size, 0.1)),
        (profile[1:-1], np.full(profile.size - 2, 0.1)),
    ]

    result = dba(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=1,
    )

    # Closed DTW makes even the shorter read visit both ends of the medoid axis.
    assert np.array_equal(result.depth, np.full(profile.size, len(reads)))


def test_profile_hmm_refines_medoid_and_learns_all_move_classes():
    rng = np.random.default_rng(12)
    truth = np.array([-2.0, -0.8, 0.7, 2.0, 0.3, -1.7, -0.2, 1.5])
    medoid_read = truth + np.array([0.22, -0.18, 0.2, -0.2, 0.18, -0.2, 0.2, -0.18])
    means = [medoid_read] + [
        truth + rng.normal(0.0, 0.08, truth.size) for _ in range(8)
    ]
    reads = [
        (mean, np.full(truth.size, 0.10), np.full(truth.size, 4.0))
        for mean in means
    ]

    result = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=15,
        tol=1e-5,
        min_std=0.02,
    )

    medoid_rmse = float(np.sqrt(np.mean((medoid_read - truth) ** 2)))
    consensus_rmse = float(np.sqrt(np.mean((result.mean - truth) ** 2)))
    assert consensus_rmse < medoid_rmse * 0.6
    assert np.array_equal(result.transition_offsets, np.array([-1, 0, 1, 2]))
    assert np.all(result.transition_prob > 0.0)
    assert result.transition_prob.sum() == pytest.approx(1.0)
    assert np.all(result.depth > len(reads) - 1.0)
    assert np.all(np.isfinite(result.stderr))


def test_one_cluster_can_feed_both_consensus_engines():
    base = np.array([-1.5, -0.5, 1.0, 1.8, 0.4, -1.0])
    reads = [
        (base + delta, np.full(base.size, 0.12), np.full(base.size, 3.0))
        for delta in (0.05, -0.03, 0.0)
    ]
    clustering = cluster_reads_dtw(reads, n_clusters=1)
    cluster = clustering.clusters[0]

    hard = dba_cluster(
        reads,
        cluster,
        normalize=False,
        open_begin=False,
        open_end=False,
    )
    soft = hmm_cluster(reads, cluster, normalize=False, min_std=0.02)

    assert hard.medoid_index == cluster.medoid_position
    assert soft.medoid_index == cluster.medoid_position
    assert hard.mean.shape == soft.mean.shape == base.shape


def test_hmm_handles_partial_spans_holds_skips_and_backsteps():
    profile = np.array([-2.0, -1.2, -0.1, 1.4, 2.1, 0.8, -0.7, -1.8, -0.4, 1.2])
    paths = [
        np.arange(profile.size),
        np.array([2, 3, 3, 4, 6, 5, 6, 7]),  # hold, skip, backstep
        np.array([1, 2, 3, 4, 5, 6, 7]),
    ]
    reads = [
        (
            profile[path],
            np.full(path.size, 0.08),
            np.full(path.size, 3.0),
        )
        for path in paths
    ]

    result = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=12,
        min_std=0.02,
    )

    assert np.all(np.isfinite(result.mean))
    assert np.all(result.transition_prob > 0.0)
    assert result.depth[4] > result.depth[0]
    assert result.depth[5] > result.depth[-1]


def test_hmm_transition_matrix_is_row_stochastic_at_boundaries():
    offsets = np.array([-1, 0, 1, 2])
    probabilities = np.array([0.10, 0.20, 0.60, 0.10])

    transition = hmm_transition_matrix(5, offsets, probabilities)

    assert np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert transition[0, 0] == pytest.approx(0.20 / 0.90)
    assert transition[0, 1] == pytest.approx(0.60 / 0.90)
    assert transition[0, 2] == pytest.approx(0.10 / 0.90)
    assert transition[-1, -2] == pytest.approx(0.10 / 0.30)
    assert transition[-1, -1] == pytest.approx(0.20 / 0.30)
    assert np.all(transition[np.tril_indices(5, -2)] == 0.0)


def test_boundary_transition_objective_gradient_matches_finite_difference():
    counts = np.array(
        [
            [0.0, 8.0, 16.0, 3.0],
            [4.0, 7.0, 18.0, 2.0],
            [6.0, 9.0, 0.0, 0.0],
        ]
    )
    legal = np.array(
        [
            [False, True, True, True],
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    logits = np.array([-0.7, 0.2, 1.1])
    _, analytic, _, _ = hmm_module._transition_objective_and_gradient(
        logits, counts, legal, 0.8
    )
    numerical = np.zeros_like(logits)
    epsilon = 1e-6
    for index in range(logits.size):
        step = np.zeros_like(logits)
        step[index] = epsilon
        above = hmm_module._transition_objective_and_gradient(
            logits + step, counts, legal, 0.8
        )[0]
        below = hmm_module._transition_objective_and_gradient(
            logits - step, counts, legal, 0.8
        )[0]
        numerical[index] = (above - below) / (2.0 * epsilon)

    assert np.allclose(analytic, numerical, rtol=1e-7, atol=1e-7)


def test_boundary_transition_optimizer_reduces_to_count_update_interior():
    counts = np.array(
        [
            [2.0, 7.0, 19.0, 3.0],
            [4.0, 5.0, 13.0, 1.0],
            [3.0, 9.0, 17.0, 2.0],
        ]
    )
    legal = np.ones_like(counts, dtype=bool)
    pseudocount = 0.75
    result = hmm_module._optimize_transition_probabilities(
        counts,
        legal,
        np.array([0.1, 0.2, 0.6, 0.1]),
        pseudocount=pseudocount,
    )
    expected = counts.sum(axis=0) + pseudocount
    expected /= expected.sum()

    assert result.success
    assert (
        result.penalized_objective_after
        >= result.penalized_objective_before - 1e-9
    )
    assert np.allclose(result.probabilities, expected, rtol=1e-8, atol=1e-9)


def test_forced_boundary_hold_exerts_no_data_pressure():
    counts = np.array([[0.0, 120.0, 0.0, 0.0]])
    legal = np.array([[False, True, False, False]])
    result = hmm_module._optimize_transition_probabilities(
        counts,
        legal,
        np.array([0.05, 0.80, 0.10, 0.05]),
        pseudocount=1.0,
    )

    assert result.data_objective_before == pytest.approx(0.0, abs=1e-12)
    assert result.data_objective_after == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(result.probabilities, np.full(4, 0.25), atol=1e-8)


def test_heteroscedastic_emission_optimizer_beats_moment_update():
    observed_mean = np.array([-0.25, -0.10, 0.05, 0.30, 1.20])
    observed_variance = np.array([0.01, 0.01, 0.04, 0.16, 1.00])
    responsibility = np.array([4.0, 6.0, 8.0, 5.0, 3.0])
    current_variance = 0.09
    min_variance = 0.001

    old_precision = responsibility / (observed_variance + current_variance)
    old_mean_update = float(
        np.dot(old_precision, observed_mean) / old_precision.sum()
    )
    moment_variance = max(
        float(
            np.dot(
                responsibility,
                (observed_mean - old_mean_update) ** 2 - observed_variance,
            )
            / responsibility.sum()
        ),
        min_variance,
    )
    moment_objective = hmm_module._profiled_emission_state_objective(
        moment_variance,
        observed_mean,
        observed_variance,
        responsibility,
    )[0]
    optimized = hmm_module._optimize_emission_state(
        observed_mean,
        observed_variance,
        responsibility,
        current_mean=0.2,
        current_variance=current_variance,
        min_variance=min_variance,
    )

    assert optimized.success
    assert optimized.objective > moment_objective
    assert optimized.variance == pytest.approx(min_variance)


def test_hmm_uniform_entry_makes_single_step_score_length_neutral():
    event = (np.array([0.0]), np.array([0.20]), np.array([2.0]))

    def constant_profile(length: int):
        read = (
            np.zeros(length),
            np.full(length, 0.20),
            np.full(length, 2.0),
        )
        return hmm_consensus(
            [read, read],
            medoid_index=0,
            normalize=False,
            max_iter=1,
            min_std=0.05,
        )

    short = constant_profile(4)
    long = constant_profile(11)

    short_score = score_hmm_profile(event, short).log_likelihood
    long_score = score_hmm_profile(event, long).log_likelihood
    assert short_score == pytest.approx(long_score, abs=1e-12)


def _diagnostic_hmm_reads():
    truth = np.array([-2.0, -0.9, 0.4, 1.8, 0.6, -1.5, -0.2, 1.3])
    perturbations = (
        np.array([0.05, -0.03, 0.02, -0.04, 0.01, 0.03, -0.02, 0.04]),
        np.array([-0.04, 0.02, -0.03, 0.05, -0.02, -0.01, 0.03, -0.03]),
        np.array([0.01, 0.04, -0.02, -0.01, 0.03, -0.04, 0.02, 0.01]),
    )
    reads = [
        (
            truth + perturbation,
            np.full(truth.size, 0.12),
            np.full(truth.size, 3.0),
        )
        for perturbation in perturbations
    ]
    return truth, reads


def test_fixed_hmm_scorer_has_correct_direction_and_does_not_mutate():
    truth, reads = _diagnostic_hmm_reads()
    profile = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=5,
        min_std=0.03,
    )
    decoy_reads = [(mean + 5.0, std, dwell) for mean, std, dwell in reads]
    decoy = hmm_consensus(
        decoy_reads,
        medoid_index=0,
        normalize=False,
        max_iter=5,
        min_std=0.03,
    )
    held_out = (
        truth + np.array([0.02, -0.01, 0.03, 0.0, -0.02, 0.01, -0.03, 0.02]),
        np.full(truth.size, 0.12),
        np.full(truth.size, 3.0),
    )
    snapshots = {
        name: getattr(profile, name).copy()
        for name in (
            "mean",
            "std",
            "stderr",
            "dwell",
            "depth",
            "transition_offsets",
            "transition_prob",
            "mean_history",
            "std_history",
            "dwell_history",
            "transition_prob_history",
            "log_likelihood",
            "mean_delta",
            "std_delta",
            "transition_delta",
            "parameter_delta",
            "transition_data_objective_before",
            "transition_data_objective_after",
            "transition_prior_objective_before",
            "transition_prior_objective_after",
            "transition_penalized_objective_before",
            "transition_penalized_objective_after",
            "transition_gradient_norm",
            "transition_optimizer_success",
            "emission_objective_before",
            "emission_objective_after",
        )
    }

    matched = score_hmm_profile(held_out, profile, return_posterior=True)
    mismatched = score_hmm_profile(held_out, decoy)

    assert np.isfinite(matched.log_likelihood)
    assert matched.log_likelihood > mismatched.log_likelihood
    assert matched.logL == matched.log_likelihood
    assert matched.posterior is not None
    assert np.allclose(matched.posterior.sum(axis=1), 1.0, atol=1e-12)
    assert not matched.posterior.flags.writeable
    for name, before in snapshots.items():
        assert np.array_equal(getattr(profile, name), before)


def test_hmm_trace_and_final_diagnostics_describe_returned_model():
    _, reads = _diagnostic_hmm_reads()
    result = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=6,
        tol=1e-8,
        min_depth=2,
        min_std=0.03,
    )

    assert result.log_likelihood.size == result.n_iter + 1
    assert result.likelihood_trace is result.log_likelihood
    assert result.mean_history.shape == (result.n_iter + 1, result.mean.size)
    assert result.std_history.shape == (result.n_iter + 1, result.std.size)
    assert result.dwell_history.shape == (result.n_iter + 1, result.dwell.size)
    assert result.transition_prob_history.shape == (
        result.n_iter + 1,
        result.transition_prob.size,
    )
    assert np.array_equal(result.mean_history[-1], result.mean)
    assert np.array_equal(result.std_history[-1], result.std)
    assert np.array_equal(result.dwell_history[-1], result.dwell)
    assert np.array_equal(
        result.transition_prob_history[-1], result.transition_prob
    )
    for delta in (
        result.mean_delta,
        result.std_delta,
        result.transition_delta,
        result.parameter_delta,
    ):
        assert delta.size == result.n_iter
        assert np.all(np.isfinite(delta))
    for objective in (
        result.transition_data_objective_before,
        result.transition_data_objective_after,
        result.transition_prior_objective_before,
        result.transition_prior_objective_after,
        result.transition_penalized_objective_before,
        result.transition_penalized_objective_after,
        result.transition_gradient_norm,
        result.emission_objective_before,
        result.emission_objective_after,
    ):
        assert objective.size == result.n_iter
        assert np.all(np.isfinite(objective))
    assert np.all(result.transition_optimizer_success)
    assert np.all(
        result.transition_penalized_objective_after
        >= result.transition_penalized_objective_before - 1e-9
    )
    assert np.all(
        result.emission_objective_after
        >= result.emission_objective_before - 1e-9
    )
    assert np.array_equal(result.combined_delta, result.parameter_delta)
    assert result.stop_reason in {"converged", "max_iter"}
    assert result.cap_hit == (result.stop_reason == "max_iter")
    assert result.settings.max_iter == 6
    assert result.settings.min_std == pytest.approx(0.03)

    scores = [
        score_hmm_profile(read, result, return_posterior=True) for read in reads
    ]
    final_likelihood = sum(score.log_likelihood for score in scores)
    assert result.final_log_likelihood == pytest.approx(final_likelihood, abs=1e-10)

    expected_depth = np.zeros(result.mean.size)
    expected_precision = np.zeros(result.mean.size)
    for read, score in zip(reads, scores):
        assert score.posterior is not None
        occupancy = score.posterior.sum(axis=0)
        expected_depth += np.minimum(occupancy, 1.0)
        variance = read[1][:, None] ** 2 + result.std[None, :] ** 2
        expected_precision += (score.posterior / variance).sum(axis=0)
    expected_stderr = np.full(result.mean.size, np.inf)
    occupied = expected_precision > 0.0
    expected_stderr[occupied] = np.sqrt(1.0 / expected_precision[occupied])

    assert np.allclose(result.depth, expected_depth, atol=1e-12)
    assert np.allclose(result.stderr, expected_stderr, atol=1e-12)
    assert np.array_equal(result.supported, expected_depth >= 2)


def test_profile_hmm_fit_and_fixed_scoring_are_deterministic():
    truth, reads = _diagnostic_hmm_reads()
    first = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=4,
        tol=1e-7,
        min_std=0.03,
    )
    second = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=4,
        tol=1e-7,
        min_std=0.03,
    )

    for name in (
        "mean",
        "std",
        "stderr",
        "dwell",
        "depth",
        "supported",
        "transition_offsets",
        "transition_prob",
        "mean_history",
        "std_history",
        "dwell_history",
        "transition_prob_history",
        "log_likelihood",
        "mean_delta",
        "std_delta",
        "transition_delta",
        "parameter_delta",
        "transition_data_objective_before",
        "transition_data_objective_after",
        "transition_prior_objective_before",
        "transition_prior_objective_after",
        "transition_penalized_objective_before",
        "transition_penalized_objective_after",
        "transition_gradient_norm",
        "transition_optimizer_success",
        "emission_objective_before",
        "emission_objective_after",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.n_iter == second.n_iter
    assert first.stop_reason == second.stop_reason
    held_out = (truth, np.full(truth.size, 0.12), np.full(truth.size, 3.0))
    assert score_hmm_profile(held_out, first).log_likelihood == pytest.approx(
        score_hmm_profile(held_out, second).log_likelihood,
        abs=0.0,
    )


def test_hmm_checkpoint_index_means_after_exactly_that_many_updates() -> None:
    _, reads = _diagnostic_hmm_reads()
    profile = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=3,
        tol=0.0,
        min_depth=2,
        min_std=0.03,
    )

    checkpoint = snapshot_hmm_profile(profile, 2, reads)

    assert checkpoint.n_iter == 2
    assert checkpoint.stop_reason == "checkpoint"
    assert not checkpoint.converged
    assert np.array_equal(checkpoint.mean, profile.mean_history[2])
    assert np.array_equal(checkpoint.std, profile.std_history[2])
    assert np.array_equal(checkpoint.dwell, profile.dwell_history[2])
    assert np.array_equal(
        checkpoint.transition_prob, profile.transition_prob_history[2]
    )
    assert checkpoint.mean_history.shape[0] == 3
    assert checkpoint.log_likelihood.shape == (3,)
    assert checkpoint.parameter_delta.shape == (2,)
    assert checkpoint.parameter_delta[-1] == profile.parameter_delta[1]
    rescored = sum(
        score_hmm_profile(read, checkpoint).log_likelihood for read in reads
    )
    assert rescored == pytest.approx(profile.log_likelihood[2], abs=1e-8)


def test_hmm_checkpoint_recomputes_posterior_diagnostics_and_final_scores() -> None:
    _, reads = _diagnostic_hmm_reads()
    profile = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=3,
        tol=0.0,
        min_depth=2,
        min_std=0.03,
    )

    checkpoint = snapshot_hmm_profile(profile, 1, reads)
    final = snapshot_hmm_profile(profile, profile.n_iter, reads)

    assert np.all(np.isfinite(checkpoint.depth))
    assert np.all(np.isfinite(checkpoint.stderr))
    assert np.array_equal(
        checkpoint.supported, checkpoint.depth >= profile.settings.min_depth
    )
    assert np.allclose(final.depth, profile.depth, atol=1e-12)
    assert np.allclose(final.stderr, profile.stderr, atol=1e-12)
    for read in reads:
        assert score_hmm_profile(read, final).log_likelihood == pytest.approx(
            score_hmm_profile(read, profile).log_likelihood,
            abs=0.0,
        )


def test_hmm_checkpoint_never_clamps_an_unavailable_iteration() -> None:
    _, reads = _diagnostic_hmm_reads()
    profile = hmm_consensus(
        reads,
        medoid_index=0,
        normalize=False,
        max_iter=2,
        tol=0.0,
        min_std=0.03,
    )

    with pytest.raises(ValueError, match="checkpoint is unavailable"):
        snapshot_hmm_profile(profile, profile.n_iter + 1, reads)
