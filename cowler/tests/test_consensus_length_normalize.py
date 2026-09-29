"""Tests for pre-DBA relative-position interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from cowler.consensus.dba import dba
from cowler.consensus.length_normalize import (
    resample_signal,
    select_fixed_target,
    select_percentile_max_target,
)
from scripts.evaluate_psk_dba_consensus import PreparedEvent, SignalTrace
from scripts.evaluate_psk_length_normalized_consensus import (
    BCN_DATASETS,
    add_representation_audit_fields,
    prepare_heldout_trace,
    shared_bootstrap_values,
    validate_transformed_score_grid,
)


def test_percentile_target_balances_datasets_and_rounds_up() -> None:
    selection = select_percentile_max_target(
        {
            "small_dataset": [10, 10, 10],
            "large_dataset": [2] * 100 + [12],
            "determining_dataset": [10, 11, 12, 13],
        },
        percentile=95.0,
    )

    expected = np.percentile([10, 11, 12, 13], 95.0, method="linear")
    assert selection.target_length == int(np.ceil(expected)) == 13
    assert selection.per_dataset_percentile["small_dataset"] == pytest.approx(10.0)
    assert selection.determining_datasets == ("determining_dataset",)


def test_fixed_target_is_exact() -> None:
    selection = select_fixed_target(100)
    assert selection.method == "fixed"
    assert selection.target_length == 100
    assert selection.percentile is None


@pytest.mark.parametrize(
    ("lengths", "message"),
    [({}, "non-empty"), ({"JS445": []}, "non-empty"), ({"JS445": [1]}, ">= 2")],
)
def test_percentile_target_rejects_invalid_cohorts(
    lengths: dict[str, list[int]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_percentile_max_target(lengths)


def test_linear_interpolation_creates_intermediate_levels() -> None:
    result = resample_signal(
        [0.30, 0.30, 0.60],
        [0.1, 0.1, 0.1],
        6,
        min_std=1e-6,
    )
    np.testing.assert_allclose(result.mean, [0.30, 0.30, 0.30, 0.36, 0.48, 0.60])
    np.testing.assert_allclose(result.relative_position, np.linspace(0.0, 1.0, 6))
    assert result.source_length == 3
    assert result.target_length == 6


def test_interpolates_variance_not_standard_deviation() -> None:
    result = resample_signal([0.0, 1.0], [1.0, 3.0], 3, min_std=1e-6)
    np.testing.assert_allclose(result.std, [1.0, np.sqrt(5.0), 3.0])


def test_resampling_preserves_endpoints_floor_and_inputs() -> None:
    mean = np.asarray([2.0, 4.0, 3.0, 8.0])
    std = np.asarray([0.0, 0.2, 0.3, 0.4])
    original_mean = mean.copy()
    original_std = std.copy()

    result = resample_signal(mean, std, 2, min_std=0.05)

    np.testing.assert_array_equal(mean, original_mean)
    np.testing.assert_array_equal(std, original_std)
    np.testing.assert_allclose(result.mean, [2.0, 8.0])
    np.testing.assert_allclose(result.std, [0.05, 0.4])


@pytest.mark.parametrize(
    ("mean", "std", "target", "message"),
    [
        ([1.0], [0.1], 3, "at least two"),
        ([1.0, 2.0], [0.1], 3, "matching"),
        ([1.0, np.nan], [0.1, 0.2], 3, "finite"),
        ([1.0, 2.0], [0.1, -0.2], 3, "non-negative"),
        ([1.0, 2.0], [0.1, 0.2], 1, ">= 2"),
    ],
)
def test_resampling_rejects_invalid_inputs(
    mean: list[float], std: list[float], target: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resample_signal(mean, std, target)


def test_equal_length_traces_produce_equal_length_dba_profile() -> None:
    source = [
        (
            np.asarray([0.0, 0.2, 0.8, 0.4, 0.1]),
            np.full(5, 0.1),
        ),
        (
            np.asarray([0.0, 0.1, 0.3, 0.9, 0.5, 0.1]),
            np.full(6, 0.1),
        ),
        (
            np.asarray([0.0, 0.25, 0.7, 0.6, 0.2, 0.1, 0.0]),
            np.full(7, 0.1),
        ),
    ]
    traces = [resample_signal(mean, std, 12) for mean, std in source]

    profile = dba(
        traces,
        medoid_index=0,
        normalize=False,
        max_iter=5,
        min_depth=2,
        max_run=3,
    )

    assert profile.mean.size == 12
    assert profile.std.size == 12
    assert profile.depth.size == 12
    assert profile.supported.size == 12
    assert np.all(np.isfinite(profile.mean))
    assert np.all(profile.std > 0.0)


def test_heldout_representation_uses_the_frozen_method_target() -> None:
    trace = SignalTrace(
        mean=np.asarray([0.2, 0.5, 0.3, 0.7]),
        std=np.asarray([0.1, 0.2, 0.3, 0.4]),
        dwell=np.ones(4),
    )

    native = prepare_heldout_trace(
        trace, "native", p95_target=7, fixed_target=10, min_std=1e-3
    )
    p95 = prepare_heldout_trace(
        trace, "p95_max", p95_target=7, fixed_target=10, min_std=1e-3
    )
    fixed = prepare_heldout_trace(
        trace, "fixed_100", p95_target=7, fixed_target=10, min_std=1e-3
    )

    assert native is trace
    assert p95.mean.size == 7
    assert fixed.mean.size == 10
    np.testing.assert_allclose(p95.mean[[0, -1]], trace.mean[[0, -1]])
    np.testing.assert_allclose(fixed.std[[0, -1]], trace.std[[0, -1]])


def test_transformed_score_grid_requires_exact_finite_shape() -> None:
    valid = np.zeros((4, len(BCN_DATASETS)))
    returned = validate_transformed_score_grid(
        "p95_max", valid, expected_events=4
    )
    assert returned.shape == (4, 9)

    with pytest.raises(RuntimeError, match="shape"):
        validate_transformed_score_grid(
            "fixed_100", np.zeros((3, 9)), expected_events=4
        )
    invalid = valid.copy()
    invalid[0, 0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_transformed_score_grid(
            "fixed_100", invalid, expected_events=4
        )


def test_native_score_grid_may_retain_inadmissible_candidate_scores() -> None:
    scores = np.zeros((2, len(BCN_DATASETS)))
    scores[0, 0] = np.inf
    returned = validate_transformed_score_grid(
        "native", scores, expected_events=2
    )
    assert np.isposinf(returned[0, 0])


def test_paired_bootstrap_reuses_draws_across_methods() -> None:
    truth = np.repeat(np.arange(len(BCN_DATASETS), dtype=np.int64), 2)
    scores = np.ones((truth.size, len(BCN_DATASETS)))
    scores[np.arange(truth.size), truth] = 0.0
    score_grids = {
        method: scores.copy()
        for method in ("native", "p95_max", "fixed_100")
    }

    values = shared_bootstrap_values(
        truth, score_grids, n_bootstrap=5, seed=17
    )

    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        np.testing.assert_array_equal(
            values["native"][metric], values["p95_max"][metric]
        )
        np.testing.assert_array_equal(
            values["native"][metric], values["fixed_100"][metric]
        )


def test_event_score_rows_audit_source_and_scoring_lengths() -> None:
    events: list[PreparedEvent] = []
    native: list[SignalTrace] = []
    for index, dataset in enumerate(BCN_DATASETS):
        trace = SignalTrace(
            mean=np.linspace(0.2, 0.7, 4 + index, dtype=np.float64),
            std=np.full(4 + index, 0.1, dtype=np.float64),
            dwell=np.ones(4 + index, dtype=np.float64),
        )
        native.append(trace)
        events.append(
            PreparedEvent(
                dataset=dataset,
                event_id=index + 1,
                event_order=index,
                stored_is_consensus=False,
                is_training=False,
                peptide=trace,
                dna_window=trace,
                tail=None,
                dna_shift=0.0,
                dna_scale=1.0,
                nuisance=np.zeros(5),
                tail_steps=0,
                raw_peptide_steps=trace.mean.size,
            )
        )
    fixed = [
        prepare_heldout_trace(
            trace, "fixed_100", p95_target=7, fixed_target=10, min_std=1e-3
        )
        for trace in native
    ]
    event_rows = [
        {
            "dataset": event.dataset,
            "event_id": event.event_id,
            "true_class": event.dataset,
            "predicted_class": event.dataset,
        }
        for event in events
    ]
    add_representation_audit_fields(
        event_rows, events, fixed, method="fixed_100"
    )

    fixed_first = event_rows[0]
    assert fixed_first["dataset"] == "JS445"
    assert fixed_first["true_label"] == "JS445"
    assert fixed_first["source_length"] == 4
    assert fixed_first["scoring_length"] == 10
    assert fixed_first["predicted_label"] == "JS445"
