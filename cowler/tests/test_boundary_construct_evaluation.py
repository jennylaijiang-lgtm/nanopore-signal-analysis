"""Focused tests for multi-construct DNA/peptide boundary evaluation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pandas as pd
import pytest

from cowler.align.hybrid_segment import (
    GaussianLevelModel,
    HybridRegionModels,
    HybridSegmentParams,
)
from cowler.align.segment import Step
from scripts import evaluate_js445_segmentation as js445
from scripts.evaluate_boundary_constructs import (
    CONSTRUCTS,
    JS445_FROZEN_TRANSFER,
    TRANSFER_TARGETS,
    WITHIN_CONSTRUCT_OOF,
    build_parser,
    dataset_paths,
    metric_row,
)


def _event(event_id: int) -> js445.EventSteps:
    truth = js445.EventTruth(
        event_id=event_id,
        start_sample=0,
        end_sample=300,
        dna_start_sample=50,
        dna_end_sample=199,
        non_dna_start_sample=200,
        non_dna_end_sample=280,
        aligned_dna_length=70,
        is_consensus=1,
        is_hand_pick=0,
    )
    steps = (
        Step(1.0, 0.1, 100, 0, 100),
        Step(2.0, 0.1, 100, 100, 200),
        Step(3.0, 0.1, 100, 200, 300),
    )
    return js445.EventSteps(truth=truth, steps=steps)


def _models() -> HybridRegionModels:
    return HybridRegionModels(
        pre_dna=GaussianLevelModel(1.0, 0.2, "pre_dna"),
        dna=GaussianLevelModel(2.0, 0.2, "dna"),
        non_dna=GaussianLevelModel(3.0, 0.2, "non_dna"),
    )


def test_fixed_model_predictor_reuses_models_without_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _models()
    seen: list[HybridRegionModels] = []

    def fake_segment(
        steps: Sequence[Step],
        profile: pd.DataFrame,
        supplied_models: HybridRegionModels,
        params: HybridSegmentParams,
    ) -> Any:
        del steps, profile, params
        seen.append(supplied_models)
        return SimpleNamespace(
            status="pass",
            anchor_start_step=1,
            anchor_end_step=2,
            anchor_start_sample=100,
            anchor_end_sample=200,
            boundary_step=2,
            boundary_sample=200,
            anchor_ref_start=1,
            anchor_ref_end=2,
            anchor_ref_coverage=1.0,
            anchor_observed_fraction=1.0,
            total_score=10.0,
            null_score=0.0,
            score_margin=10.0,
            anchor_emission_margin=2.0,
        )

    # This bookkeeping test uses a fake decoder; no measured LUT is needed.
    monkeypatch.setattr(js445, "predict_DNA_6mer_5_3", lambda _: pd.DataFrame())
    monkeypatch.setattr(js445, "segment_dna_non_dna", fake_segment)
    rows = js445.predict_with_models(
        [_event(1), _event(2)],
        models=models,
        params=HybridSegmentParams(),
    )

    assert seen == [models, models]
    assert [row["fold"] for row in rows] == [-1, -1]
    assert [row["boundary_error_steps"] for row in rows] == [0, 0]


def test_metric_row_reports_percentages_over_accepted_events() -> None:
    summary = {
        "n_events": 10,
        "n_pass": 8,
        "n_rejected": 2,
        "n_error": 0,
        "acceptance_rate": 0.8,
        "median_error_steps": -1.0,
        "mean_error_steps": -0.5,
        "median_absolute_error_steps": 1.0,
        "mean_absolute_error_steps": 1.5,
        "p90_absolute_error_steps": 3.0,
        "max_absolute_error_steps": 5.0,
        "exact_step_fraction": 0.25,
        "within_1_step_fraction": 0.625,
        "within_2_steps_fraction": 0.75,
    }
    row = metric_row("JS446", JS445_FROZEN_TRANSFER, summary)

    assert row["n_accepted"] == 8
    assert row["exact_percent"] == pytest.approx(25.0)
    assert row["within_1_step_percent"] == pytest.approx(62.5)
    assert row["within_2_steps_percent"] == pytest.approx(75.0)


def test_dataset_paths_and_defaults_cover_the_prespecified_constructs(
    tmp_path: Path,
) -> None:
    js445_dir = tmp_path / "js445"
    data_dir = tmp_path / "other"
    js445_dir.mkdir()
    data_dir.mkdir()
    for construct, directory in (("JS445", js445_dir), ("JS446", data_dir)):
        (directory / f"{construct}_synthetic.fast5").touch()
        (directory / f"{construct}_synthetic.annot.fast5").touch()

    assert dataset_paths(
        "JS445", data_dir=data_dir, js445_dir=js445_dir
    )[0].parent == js445_dir
    assert dataset_paths(
        "JS446", data_dir=data_dir, js445_dir=js445_dir
    )[0].parent == data_dir

    defaults = build_parser().parse_args([])
    assert tuple(defaults.constructs) == CONSTRUCTS
    assert TRANSFER_TARGETS == tuple(f"JS{number}" for number in range(446, 454))
    assert defaults.folds == 8
    assert defaults.boundary_trim_steps == 4
    assert {WITHIN_CONSTRUCT_OOF, JS445_FROZEN_TRANSFER} == {
        "within_construct_oof",
        "js445_frozen_transfer",
    }
