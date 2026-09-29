"""Predict and evaluate DNA-to-non-DNA boundaries for all JS445 events.

The raw JS445 FAST5 is one continuous trace and does not contain the 96 event
spans.  This script therefore uses only ``start_idx``/``end_idx`` from the
manual ``Events`` table to isolate reads. It does not expose an event's semantic
boundaries to that event's decoder.

Region emission models are fitted out-of-fold from the other events' manual
regions.  This is useful within-run validation, not independent biological
validation: all folds still come from the same JS445 construct and run.
Exact/within-one/within-two fractions are calculated over accepted events only;
rejected and errored events are excluded from those accuracy denominators.

The reported manual boundary is the source table's ``p_start_idx``. In JS445 it is exactly
one sample after ``d_end_idx`` for every event. The automatic boundary is the
three-region decoder's own ``semantic_boundary`` with no post-hoc correction; the
earlier out-of-fold offset calibration has been removed as a dataset-specific fit
that did not generalise. The source field is interpreted only as the start of
non-DNA signal; the dataset provides no target for an intermediate region or a
more specific downstream identity.

The evaluator inherits the reusable segmenter's QC defaults. Region emission
models are still fitted out-of-fold from manual regions (they are the decoder's
emission model), so "uncalibrated" means without a boundary correction, not fully
unsupervised.

Example:

    uv run python -m scripts.evaluate_js445_segmentation \
        --raw ../2026.07.15-dataset/JS445_synthetic.fast5 \
        --annotations ../2026.07.15-dataset/JS445_synthetic.annot.fast5 \
        --out-dir tmp/js445_segmentation

Outputs:

``js445_predicted_boundaries.csv``
    Prediction-only table. It contains event spans but no manual semantic boundaries.
``js445_boundary_comparison.csv``
    Predictions joined to manual boundaries and observed-step errors.
``js445_summary.json``
    Aggregate acceptance and boundary-error metrics.
``js445_calibration_models.csv``
    Out-of-fold Gaussian parameters and training counts.
``js445_boundary_trim_sweep.csv`` and ``js445_boundary_trim_predictions.csv``
    Written with ``--sweep-boundary-trim``; aggregate and per-event held-out results
    for each tested full-profile anchor trim.
``js445_boundary_comparison.png`` and ``js445_boundary_overview.png``
    Aggregate comparison and per-event trace panels.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import h5py
import numpy as np
import numpy.typing as npt

from cowler.align.hybrid_segment import (
    DEFAULT_BOUNDARY_TRIM_STEPS,
    GaussianLevelModel,
    HybridRegionModels,
    HybridSegmentParams,
    segment_dna_non_dna,
    select_trusted_anchor,
)
from cowler.align.segment import Step, find_steps
from cowler.io.lut import predict_DNA_6mer_5_3
from cowler.io.normalize import normalize_signal

TEMPLATE_DNA = (
    "TTACTGAAGTCTCACGTGCCTGGTATATTAGCGTCCACTCTCACTATCGGATTCTACATCGGTCGTAGCC"
)
BOUNDARY_TRIM_SWEEP_STEPS = tuple(range(10))


@dataclass(frozen=True)
class EventTruth:
    """Event-isolation metadata plus boundaries used only for fitting/evaluation."""

    event_id: int
    start_sample: int
    end_sample: int
    dna_start_sample: int
    dna_end_sample: int
    non_dna_start_sample: int
    non_dna_end_sample: int
    aligned_dna_length: int
    is_consensus: int
    is_hand_pick: int


@dataclass(frozen=True)
class EventSteps:
    """One event's pA steps with global raw-sample bounds."""

    truth: EventTruth
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class FittedLevel:
    """Auditable parameters for one fold/semantic region."""

    fold: int
    region: str
    mean: float
    std: float
    n_steps: int


def load_raw_pa(raw_path: Path) -> tuple[npt.NDArray[np.float64], float]:
    """Load the continuous raw signal and apply the FAST5 pA calibration."""

    with h5py.File(raw_path, "r") as handle:
        dataset = cast(h5py.Dataset, handle["Raw/Channel_1/Signal"])
        raw = np.asarray(dataset[:], dtype=np.float64)
        calibration = dataset.parent.attrs
        offset = float(cast(Any, calibration["offset"]))
        signal_range = float(cast(Any, calibration["range"]))
        digitisation = float(cast(Any, calibration["digitisation"]))
        sample_rate = float(cast(Any, calibration["sample_rate"]))
    pa = (raw + offset) * signal_range / digitisation
    return pa, sample_rate


def load_event_truth(annotation_path: Path) -> list[EventTruth]:
    """Read and validate event spans and manual evaluation boundaries."""

    with h5py.File(annotation_path, "r") as handle:
        dataset = cast(h5py.Dataset, handle["Events"])
        events = np.asarray(dataset[:])
    events = np.sort(events, order="index")

    result: list[EventTruth] = []
    for event in events:
        truth = EventTruth(
            event_id=int(event["index"]),
            start_sample=int(event["start_idx"]),
            end_sample=int(event["end_idx"]),
            dna_start_sample=int(event["d_start_idx"]),
            dna_end_sample=int(event["d_end_idx"]),
            non_dna_start_sample=int(event["p_start_idx"]),
            non_dna_end_sample=int(event["p_end_idx"]),
            aligned_dna_length=int(event["aligned_DNA_length"]),
            is_consensus=int(event["isConsensus"]),
            is_hand_pick=int(event["isHandPick"]),
        )
        if not (
            truth.start_sample
            <= truth.dna_start_sample
            < truth.dna_end_sample
            <= truth.non_dna_start_sample
            < truth.non_dna_end_sample
            <= truth.end_sample
        ):
            raise ValueError(f"invalid boundary ordering for event {truth.event_id}")
        result.append(truth)

    if not result:
        raise ValueError("annotation file contains no events")
    return result


def segment_event_steps(
    pa: npt.NDArray[np.float64],
    truth: EventTruth,
    *,
    sensitivity: float,
    min_level_length: int,
) -> EventSteps:
    """Run CPIC on a normalized event, then recompute step statistics in pA."""

    event_pa = pa[truth.start_sample : truth.end_sample]
    local_steps = find_steps(
        normalize_signal(event_pa),
        sensitivity=sensitivity,
        min_level_length=min_level_length,
    )

    global_steps: list[Step] = []
    for local in local_steps:
        local_segment = event_pa[local.start_sample : local.end_sample]
        finite = local_segment[np.isfinite(local_segment)]
        if not finite.size:
            continue
        global_steps.append(
            Step(
                mean=float(np.median(finite)),
                std=float(np.std(finite)),
                dwell=int(local.end_sample - local.start_sample),
                start_sample=truth.start_sample + int(local.start_sample),
                end_sample=truth.start_sample + int(local.end_sample),
            )
        )
    if not global_steps:
        raise ValueError(f"step finder returned no finite steps for event {truth.event_id}")
    return EventSteps(truth=truth, steps=tuple(global_steps))


def _step_midpoint(step: Step) -> int:
    return (step.start_sample + step.end_sample) // 2


def _training_values(
    events: Iterable[EventSteps],
) -> tuple[list[float], list[float], list[float]]:
    """Collect pre-DNA, DNA, and downstream non-DNA training levels."""

    pre_dna: list[float] = []
    dna: list[float] = []
    non_dna: list[float] = []
    for event in events:
        truth = event.truth
        for step in event.steps:
            midpoint = _step_midpoint(step)
            if midpoint < truth.dna_start_sample:
                pre_dna.append(step.mean)
            elif midpoint < truth.non_dna_start_sample:
                dna.append(step.mean)
            else:
                non_dna.append(step.mean)
    return pre_dna, dna, non_dna


def _fit_level(
    values: Sequence[float],
    *,
    label: str,
    fold: int,
    min_std: float = 1e-3,
) -> tuple[GaussianLevelModel, FittedLevel]:
    """Fit a robust single-Gaussian prototype emission."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError(f"fold {fold} has too few {label!r} training steps")
    center = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - center)))
    if not np.isfinite(scale) or scale < min_std:
        scale = max(float(np.std(array)), min_std)
    return (
        GaussianLevelModel(center, scale, label),
        FittedLevel(fold, label, center, scale, int(array.size)),
    )


def fit_fold_models(
    training_events: Sequence[EventSteps], fold: int
) -> tuple[HybridRegionModels, list[FittedLevel]]:
    """Fit the pre-DNA, DNA, and non-DNA region models."""

    pre_dna_values, dna_values, non_dna_values = _training_values(training_events)
    pre_dna, pre_dna_fit = _fit_level(
        pre_dna_values, label="pre_dna", fold=fold
    )
    dna, dna_fit = _fit_level(dna_values, label="dna", fold=fold)
    non_dna, non_dna_fit = _fit_level(
        non_dna_values, label="non_dna", fold=fold
    )
    return HybridRegionModels(pre_dna, dna, non_dna), [
        pre_dna_fit,
        dna_fit,
        non_dna_fit,
    ]


def _fold_for(event_id: int, n_folds: int) -> int:
    return (event_id - 1) % n_folds


def _nearest_truth_step(steps: Sequence[Step], truth_sample: int) -> int:
    starts = np.asarray([step.start_sample for step in steps], dtype=np.int64)
    return int(np.argmin(np.abs(starts - truth_sample)))


def predict_with_models(
    events: Sequence[EventSteps],
    *,
    models: HybridRegionModels,
    params: HybridSegmentParams,
    fold: int = -1,
) -> list[dict[str, Any]]:
    """Predict events with one already fitted, unchanged set of region models.

    This is the fixed-model counterpart to :func:`predict_cross_validated`. It
    deliberately performs no fitting or boundary calibration, which allows a
    model trained on one construct/run to be evaluated on another. Manual
    boundaries are joined only after decoding to calculate held-out errors.
    """

    profile = predict_DNA_6mer_5_3(TEMPLATE_DNA)
    rows: list[dict[str, Any]] = []
    for event in events:
        truth = event.truth
        base: dict[str, Any] = {
            "event_id": truth.event_id,
            "fold": fold,
            "event_start_sample": truth.start_sample,
            "event_end_sample": truth.end_sample,
            "n_steps": len(event.steps),
            "is_consensus": truth.is_consensus,
            "is_hand_pick": truth.is_hand_pick,
        }
        try:
            result = segment_dna_non_dna(event.steps, profile, models, params)
            base.update(
                {
                    "status": result.status,
                    "predicted_anchor_start_step": result.anchor_start_step,
                    "predicted_anchor_end_step": result.anchor_end_step,
                    "predicted_anchor_start_sample": result.anchor_start_sample,
                    "predicted_anchor_end_sample": result.anchor_end_sample,
                    "predicted_boundary_step": result.boundary_step,
                    "predicted_boundary_sample": result.boundary_sample,
                    "semantic_boundary_step": result.boundary_step,
                    "semantic_boundary_sample": result.boundary_sample,
                    "anchor_ref_start": result.anchor_ref_start,
                    "anchor_ref_end": result.anchor_ref_end,
                    "anchor_ref_coverage": result.anchor_ref_coverage,
                    "anchor_observed_fraction": result.anchor_observed_fraction,
                    "total_score": result.total_score,
                    "null_score": result.null_score,
                    "score_margin": result.score_margin,
                    "anchor_emission_margin": result.anchor_emission_margin,
                    "error_message": "",
                }
            )
        except (ValueError, RuntimeError) as exc:
            base.update(
                {
                    "status": "error",
                    "predicted_anchor_start_step": -1,
                    "predicted_anchor_end_step": -1,
                    "predicted_anchor_start_sample": -1,
                    "predicted_anchor_end_sample": -1,
                    "predicted_boundary_step": -1,
                    "predicted_boundary_sample": -1,
                    "semantic_boundary_step": -1,
                    "semantic_boundary_sample": -1,
                    "anchor_ref_start": -1,
                    "anchor_ref_end": -1,
                    "anchor_ref_coverage": 0.0,
                    "anchor_observed_fraction": 0.0,
                    "total_score": float("nan"),
                    "null_score": float("nan"),
                    "score_margin": float("nan"),
                    "anchor_emission_margin": float("nan"),
                    "error_message": str(exc),
                }
            )

        truth_step = _nearest_truth_step(event.steps, truth.non_dna_start_sample)
        predicted_step = int(base["predicted_boundary_step"])
        has_prediction = base["status"] == "pass" and predicted_step >= 0
        base.update(
            {
                "manual_dna_start_sample": truth.dna_start_sample,
                "manual_dna_end_sample": truth.dna_end_sample,
                "manual_boundary_sample": truth.non_dna_start_sample,
                "manual_non_dna_end_sample": truth.non_dna_end_sample,
                "manual_boundary_nearest_step": truth_step,
                "aligned_dna_length": truth.aligned_dna_length,
                "boundary_error_steps": (
                    predicted_step - truth_step if has_prediction else ""
                ),
            }
        )
        rows.append(base)

    rows.sort(key=lambda row: int(row["event_id"]))
    return rows


def predict_cross_validated(
    events: Sequence[EventSteps],
    *,
    n_folds: int,
    params: HybridSegmentParams,
) -> tuple[list[dict[str, Any]], list[FittedLevel]]:
    """Predict each event with models fitted without that event's fold.

    The reported boundary is the decoder's own uncalibrated ``semantic_boundary``;
    no post-hoc offset correction is applied.
    """

    if not 2 <= n_folds <= len(events):
        raise ValueError("n_folds must be between 2 and the number of events")
    rows: list[dict[str, Any]] = []
    all_fits: list[FittedLevel] = []

    for fold in range(n_folds):
        training = [
            event
            for event in events
            if _fold_for(event.truth.event_id, n_folds) != fold
        ]
        held_out = [
            event
            for event in events
            if _fold_for(event.truth.event_id, n_folds) == fold
        ]
        models, fits = fit_fold_models(training, fold)
        all_fits.extend(fits)
        rows.extend(
            predict_with_models(
                held_out,
                models=models,
                params=params,
                fold=fold,
            )
        )

    rows.sort(key=lambda row: int(row["event_id"]))
    return rows, all_fits


PREDICTION_FIELDS = [
    "event_id",
    "fold",
    "event_start_sample",
    "event_end_sample",
    "n_steps",
    "is_consensus",
    "is_hand_pick",
    "status",
    "predicted_anchor_start_step",
    "predicted_anchor_end_step",
    "predicted_anchor_start_sample",
    "predicted_anchor_end_sample",
    "predicted_boundary_step",
    "predicted_boundary_sample",
    "semantic_boundary_step",
    "semantic_boundary_sample",
    "anchor_ref_start",
    "anchor_ref_end",
    "anchor_ref_coverage",
    "anchor_observed_fraction",
    "total_score",
    "null_score",
    "score_margin",
    "anchor_emission_margin",
    "error_message",
]

COMPARISON_FIELDS = PREDICTION_FIELDS + [
    "manual_dna_start_sample",
    "manual_dna_end_sample",
    "manual_boundary_sample",
    "manual_non_dna_end_sample",
    "manual_boundary_nearest_step",
    "aligned_dna_length",
    "boundary_error_steps",
]

TRIM_SWEEP_FIELDS = [
    "boundary_trim_steps",
    "approx_trimmed_nt",
    "n_anchor_states",
    "anchor_first_profile_row",
    "anchor_last_profile_row",
    "anchor_profile_rows",
    "trimmed_profile_rows",
    "selected",
    "selection_min_acceptance_rate",
    "n_events",
    "n_pass",
    "n_rejected",
    "n_error",
    "acceptance_rate",
    "median_absolute_error_steps",
    "mean_absolute_error_steps",
    "p90_absolute_error_steps",
    "max_absolute_error_steps",
    "mean_error_steps",
    "median_error_steps",
    "exact_step_fraction",
    "within_1_step_fraction",
    "within_2_steps_fraction",
    "sample_rate_hz",
    "manual_boundary_field",
    "automatic_boundary_source",
    "event_isolation_fields",
    "calibration",
    "scope",
]

def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: Sequence[dict[str, Any]], sample_rate: float
) -> dict[str, Any]:
    """Calculate aggregate boundary metrics over accepted events only."""

    accepted = [row for row in rows if row["status"] == "pass"]
    errors = np.asarray(
        [float(row["boundary_error_steps"]) for row in accepted], dtype=float
    )
    absolute = np.abs(errors)
    summary: dict[str, Any] = {
        "n_events": len(rows),
        "n_pass": len(accepted),
        "n_rejected": sum(row["status"] == "rejected" for row in rows),
        "n_error": sum(row["status"] == "error" for row in rows),
        "acceptance_rate": len(accepted) / len(rows),
        "sample_rate_hz": sample_rate,
        "manual_boundary_field": "p_start_idx",
        "automatic_boundary_source": "semantic_boundary",
        "event_isolation_fields": ["start_idx", "end_idx"],
        "calibration": "none (uncalibrated decoder boundary); "
        "region models fitted out-of-fold from manual regions",
        "scope": "within-run JS445 validation; not independent held-out biology",
    }
    if errors.size:
        summary.update(
            {
                "mean_error_steps": float(np.mean(errors)),
                "median_error_steps": float(np.median(errors)),
                "mean_absolute_error_steps": float(np.mean(absolute)),
                "median_absolute_error_steps": float(np.median(absolute)),
                "p90_absolute_error_steps": float(np.quantile(absolute, 0.90)),
                "max_absolute_error_steps": float(np.max(absolute)),
                "exact_step_fraction": float(np.mean(absolute == 0.0)),
                "within_1_step_fraction": float(np.mean(absolute <= 1.0)),
                "within_2_steps_fraction": float(np.mean(absolute <= 2.0)),
            }
        )
    return summary


def select_optimal_boundary_trim(
    summaries: Sequence[dict[str, Any]],
    *,
    min_acceptance_rate: float = 0.90,
) -> int:
    """Choose the lowest-error trim among candidates with adequate acceptance.

    The criterion is fixed before inspecting JS445 results: require the configured
    acceptance rate, then minimize median absolute observed-step error, followed
    by p90, mean absolute error, and finally the smaller trim. If no candidate
    reaches the threshold, restrict selection to candidates with the highest
    acceptance rate.
    """

    if not 0.0 <= min_acceptance_rate <= 1.0:
        raise ValueError("min_acceptance_rate must be in [0, 1]")
    usable = [
        row
        for row in summaries
        if "median_absolute_error_steps" in row
        and np.isfinite(float(row["median_absolute_error_steps"]))
    ]
    if not usable:
        raise ValueError("trim sweep contains no accepted predictions")
    eligible = [
        row
        for row in usable
        if float(row["acceptance_rate"]) >= min_acceptance_rate
    ]
    if not eligible:
        best_acceptance = max(float(row["acceptance_rate"]) for row in usable)
        eligible = [
            row
            for row in usable
            if float(row["acceptance_rate"]) == best_acceptance
        ]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["median_absolute_error_steps"]),
            float(row["p90_absolute_error_steps"]),
            float(row["mean_absolute_error_steps"]),
            int(row["boundary_trim_steps"]),
        ),
    )
    return int(selected["boundary_trim_steps"])


def sweep_boundary_trim_steps(
    events: Sequence[EventSteps],
    *,
    n_folds: int,
    base_params: HybridSegmentParams,
    sample_rate: float,
    trim_steps: Sequence[int] = BOUNDARY_TRIM_SWEEP_STEPS,
    min_acceptance_rate: float = 0.90,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate full-DNA anchors with 0--9 boundary-proximal states removed.

    Each candidate uses the complete post-trim profile. Region models are fitted
    out-of-fold independently for that candidate, and selection uses the decoder's
    uncalibrated boundary. Returned summaries record the exact source-profile rows
    retained and removed.
    """

    candidates = tuple(int(value) for value in trim_steps)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("trim_steps must be a non-empty sequence of unique values")
    if any(value < 0 for value in candidates):
        raise ValueError("trim_steps values must be >= 0")

    profile = predict_DNA_6mer_5_3(TEMPLATE_DNA)
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for trim in candidates:
        params = replace(
            base_params,
            boundary_trim_steps=trim,
        )
        anchor = select_trusted_anchor(
            profile,
            boundary_edge=params.boundary_edge,
            boundary_trim_steps=trim,
        )
        rows, _ = predict_cross_validated(
            events,
            n_folds=n_folds,
            params=params,
        )
        for row in rows:
            predictions.append({"boundary_trim_steps": trim, **row})

        summary = summarize(rows, sample_rate)
        summary.update(
            {
                "boundary_trim_steps": trim,
                "approx_trimmed_nt": trim / 2.0,
                "n_anchor_states": len(anchor),
                "anchor_first_profile_row": anchor.anchor_profile_rows[0],
                "anchor_last_profile_row": anchor.anchor_profile_rows[-1],
                "anchor_profile_rows": json.dumps(anchor.anchor_profile_rows),
                "trimmed_profile_rows": json.dumps(anchor.trimmed_profile_rows),
            }
        )
        summaries.append(summary)

    selected_trim = select_optimal_boundary_trim(
        summaries, min_acceptance_rate=min_acceptance_rate
    )
    for summary in summaries:
        summary["selected"] = int(summary["boundary_trim_steps"]) == selected_trim
        summary["selection_min_acceptance_rate"] = min_acceptance_rate
    return summaries, predictions


def make_plots(
    pa: npt.NDArray[np.float64],
    events: Sequence[EventSteps],
    rows: Sequence[dict[str, Any]],
    sample_rate: float,
    out_dir: Path,
) -> None:
    """Write an aggregate comparison and a trace panel for every event."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accepted = [row for row in rows if row["status"] == "pass"]
    manual_steps = np.asarray(
        [int(row["manual_boundary_nearest_step"]) for row in accepted], dtype=int
    )
    predicted_steps = np.asarray(
        [int(row["predicted_boundary_step"]) for row in accepted], dtype=int
    )
    errors_steps = predicted_steps - manual_steps
    consensus = np.asarray([int(row["is_consensus"]) for row in accepted])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for value, label, color in ((0, "isConsensus=0", "C0"), (1, "isConsensus=1", "C1")):
        selected = consensus == value
        axes[0].scatter(
            manual_steps[selected],
            predicted_steps[selected],
            s=24,
            alpha=0.8,
            label=label,
            color=color,
        )
    if manual_steps.size:
        lower = float(min(np.min(manual_steps), np.min(predicted_steps)))
        upper = float(max(np.max(manual_steps), np.max(predicted_steps)))
        axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axes[0].set(
        xlabel="Manual nearest boundary (observed-step index)",
        ylabel="Predicted boundary (observed-step index)",
        title="Predicted versus manual boundary",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].hist(errors_steps, bins=24, color="C2", alpha=0.85)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Prediction error (observed steps)",
        ylabel="Events",
        title="Signed boundary error",
    )
    axes[1].grid(alpha=0.2)

    axes[2].scatter(
        [int(row["event_id"]) for row in accepted],
        errors_steps,
        c=consensus,
        cmap="coolwarm",
        s=24,
    )
    axes[2].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[2].set(
        xlabel="Event ID",
        ylabel="Prediction error (observed steps)",
        title="Error by event",
    )
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "js445_boundary_comparison.png", dpi=300)
    plt.close(fig)

    row_by_event = {int(row["event_id"]): row for row in rows}
    n_columns = 4
    n_rows = int(np.ceil(len(events) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 1.2 * n_rows),
        squeeze=False,
    )
    for axis, event in zip(axes.flat, events):
        truth = event.truth
        row = row_by_event[truth.event_id]
        event_pa = pa[truth.start_sample : truth.end_sample]
        stride = max(1, len(event_pa) // 2500)
        time = np.arange(0, len(event_pa), stride) / sample_rate
        axis.plot(time, event_pa[::stride], color="0.55", linewidth=0.35)
        manual_time = (truth.non_dna_start_sample - truth.start_sample) / sample_rate
        axis.axvline(manual_time, color="k", linestyle="--", linewidth=0.9)
        if row["status"] == "pass":
            predicted_time = (
                int(row["predicted_boundary_sample"]) - truth.start_sample
            ) / sample_rate
            axis.axvline(predicted_time, color="C3", linewidth=0.9)
        axis.set_title(
            f"event {truth.event_id}: {row['status']}", fontsize=7
        )
        axis.tick_params(labelsize=6)
        axis.margins(x=0)
    for axis in axes.flat[len(events) :]:
        axis.axis("off")
    fig.suptitle("JS445 boundaries: manual (black dashed) vs predicted (red)", fontsize=13)
    fig.supxlabel("Time within event (s)")
    fig.supylabel("Calibrated current")
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.985))
    fig.savefig(out_dir / "js445_boundary_overview.png", dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = Path(args.raw).resolve()
    annotation_path = Path(args.annotations).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pa, sample_rate = load_raw_pa(raw_path)
    truths = load_event_truth(annotation_path)
    if max(truth.end_sample for truth in truths) > len(pa):
        raise ValueError("annotation event bounds extend beyond the raw signal")

    events = [
        segment_event_steps(
            pa,
            truth,
            sensitivity=float(args.sensitivity),
            min_level_length=int(args.min_level_length),
        )
        for truth in truths
    ]
    params = HybridSegmentParams(
        orientation="dna_to_non_dna",
        boundary_edge="profile_end",
        boundary_trim_steps=int(args.boundary_trim_steps),
        min_anchor_coverage=float(args.min_anchor_coverage),
        min_anchor_observed_fraction=float(args.min_anchor_observed_fraction),
        min_non_dna_steps=int(args.min_non_dna_steps),
        min_score_margin=float(args.min_score_margin),
        min_anchor_emission_margin=float(args.min_anchor_emission_margin),
    )
    rows, fits = predict_cross_validated(
        events,
        n_folds=int(args.folds),
        params=params,
    )
    summary = summarize(rows, sample_rate)
    summary.update(
        {
            "template_dna": TEMPLATE_DNA,
            "raw_path": str(raw_path),
            "annotation_path": str(annotation_path),
            "n_folds": int(args.folds),
            "step_finder_sensitivity": float(args.sensitivity),
            "step_finder_min_level_length": int(args.min_level_length),
            "hybrid_segment_params": asdict(params),
        }
    )

    _write_csv(out_dir / "js445_predicted_boundaries.csv", rows, PREDICTION_FIELDS)
    _write_csv(out_dir / "js445_boundary_comparison.csv", rows, COMPARISON_FIELDS)
    _write_csv(
        out_dir / "js445_calibration_models.csv",
        [asdict(fit) for fit in fits],
        ["fold", "region", "mean", "std", "n_steps"],
    )
    if bool(args.sweep_boundary_trim):
        if int(args.trim_max) < int(args.trim_min):
            raise ValueError("trim_max must be >= trim_min")
        sweep_summaries, sweep_predictions = sweep_boundary_trim_steps(
            events,
            n_folds=int(args.folds),
            base_params=params,
            sample_rate=sample_rate,
            trim_steps=range(int(args.trim_min), int(args.trim_max) + 1),
            min_acceptance_rate=float(args.sweep_min_acceptance_rate),
        )
        _write_csv(
            out_dir / "js445_boundary_trim_sweep.csv",
            sweep_summaries,
            TRIM_SWEEP_FIELDS,
        )
        _write_csv(
            out_dir / "js445_boundary_trim_predictions.csv",
            sweep_predictions,
            ["boundary_trim_steps", *COMPARISON_FIELDS],
        )
        selected = next(row for row in sweep_summaries if bool(row["selected"]))
        summary["boundary_trim_sweep"] = {
            "tested_steps": [
                int(row["boundary_trim_steps"]) for row in sweep_summaries
            ],
            "selected_steps": int(selected["boundary_trim_steps"]),
            "selection_min_acceptance_rate": float(
                args.sweep_min_acceptance_rate
            ),
            "median_absolute_error_steps": float(
                selected["median_absolute_error_steps"]
            ),
            "acceptance_rate": float(selected["acceptance_rate"]),
        }
    with (out_dir / "js445_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not bool(args.no_plots):
        make_plots(pa, events, rows, sample_rate, out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        default="../2026.07.15-dataset/JS445_synthetic.fast5",
    )
    parser.add_argument(
        "--annotations",
        default="../2026.07.15-dataset/JS445_synthetic.annot.fast5",
    )
    parser.add_argument("--out-dir", default="tmp/js445_segmentation")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--min-level-length", type=int, default=2)
    parser.add_argument(
        "--boundary-trim-steps",
        type=int,
        default=DEFAULT_BOUNDARY_TRIM_STEPS,
        help="expected profile states removed at the DNA/non-DNA boundary (JS445: 4)",
    )
    public_defaults = HybridSegmentParams()
    parser.add_argument(
        "--min-anchor-coverage",
        type=float,
        default=public_defaults.min_anchor_coverage,
    )
    parser.add_argument(
        "--min-anchor-observed-fraction",
        type=float,
        default=public_defaults.min_anchor_observed_fraction,
    )
    parser.add_argument(
        "--min-non-dna-steps",
        type=int,
        default=public_defaults.min_non_dna_steps,
    )
    parser.add_argument("--min-score-margin", type=float, default=0.0)
    parser.add_argument("--min-anchor-emission-margin", type=float, default=0.0)
    parser.add_argument(
        "--sweep-boundary-trim",
        action="store_true",
        help="evaluate full-profile anchors with trim_min..trim_max states removed",
    )
    parser.add_argument("--trim-min", type=int, default=0)
    parser.add_argument("--trim-max", type=int, default=9)
    parser.add_argument("--sweep-min-acceptance-rate", type=float, default=0.90)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
