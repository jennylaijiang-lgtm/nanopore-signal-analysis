"""Focused tests for the gated Stage 2 medoid-robustness runner."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import numpy as np

from cowler.consensus.dba import Barycenter
from scripts.evaluate_medoid_robustness_stage2 import (
    _aligned_rmse,
    _cache_profile,
    _rank1_gate_row,
)


def _profile() -> Barycenter:
    return Barycenter(
        mean=np.asarray([0.2, 0.3, 0.4]),
        std=np.asarray([0.01, 0.02, 0.01]),
        dwell=np.asarray([0.1, 0.1, 0.2]),
        depth=np.asarray([4, 4, 4], dtype=np.int64),
        supported=np.asarray([True, True, True]),
        n_iter=2,
        converged=True,
        medoid_index=1,
        objective=np.asarray([4.0, 2.0]),
        delta=np.asarray([0.1, 0.0]),
    )


def _candidate() -> dict[str, str]:
    return {
        "construct": "JS447",
        "candidate_rank": "1",
        "local_training_index": "1",
        "original_read_id": "57",
        "original_input_index": "56",
    }


def test_rank1_gate_requires_exact_serialized_profile_reproduction() -> None:
    profile = _profile()
    baseline = {
        "mean": profile.mean.copy(),
        "std": profile.std.copy(),
        "depth": profile.depth.copy(),
        "supported": profile.supported.copy(),
    }
    diagnostic = {
        "medoid_event_id": "57",
        "profile_length": "3",
        "n_iter": "2",
        "converged": "1",
        "objective_start": "4.0",
        "objective_end": "2.0",
    }

    passed = _rank1_gate_row(
        "JS447", _candidate(), profile, baseline, diagnostic
    )
    altered = {**baseline, "mean": baseline["mean"] + np.asarray([0.0, 1e-12, 0.0])}
    failed = _rank1_gate_row(
        "JS447", _candidate(), profile, altered, diagnostic
    )

    assert passed["passed"] == 1
    assert failed["passed"] == 0
    assert float(failed["max_abs_mean_difference"]) > 0.0


def test_consensus_cache_records_explicit_candidate_and_convergence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rank1.npz"
    profile = _profile()

    _cache_profile(path, profile, _candidate(), input_fingerprint="abc123")

    with np.load(path, allow_pickle=False) as cached:
        assert cached["construct"].item() == "JS447"
        assert cached["candidate_rank"].item() == 1
        assert cached["candidate_event_id"].item() == 57
        assert cached["medoid_index"].item() == 1
        assert cached["converged"].item()
        assert cached["n_iter"].item() == 2
        assert cached["consensus_length"].item() == 3
        assert cached["final_delta"].item() == 0.0
        assert cached["input_fingerprint"].item() == "abc123"


def test_aligned_rmse_uses_stage2_closed_dtw_definition() -> None:
    baseline = np.asarray([0.0, 1.0, 2.0, 3.0])
    alternative = np.asarray([0.0, 1.0, 1.0, 2.0, 3.0])

    rmse, path_length = _aligned_rmse(baseline, alternative)

    assert rmse == 0.0
    assert path_length == 5
