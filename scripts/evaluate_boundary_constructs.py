"""Evaluate DNA-to-peptide boundary localisation across JS445--JS453.

The field-level ``peptide`` terminology used here denotes all signal downstream
of the annotated DNA boundary. The decoder itself retains the more conservative
``non_dna`` state label and does not infer downstream molecular identity.

Two prespecified analyses use the same segmentation, known-DNA profile, default
four-half-step boundary trim, QC thresholds, and observed-step error definition:

``within_construct_oof``
    Fit three region-emission models in eight folds within each construct and
    predict each event using only the other folds from that construct. This is
    held out by event but remains within-run validation.

``js445_frozen_transfer``
    Fit the three region-emission models once using all JS445 events, then apply
    them unchanged to JS446--JS453. No target-construct labels, boundary offset,
    or current-scale refit enter prediction. Manual ``p_start_idx`` values are
    joined only afterward to calculate errors. This tests construct/run transfer
    and can therefore expose current-scale as well as boundary-model failures.

Generated outputs are analysis artifacts and are written under ``tmp/`` by
default. Accuracy fractions use accepted events as their denominator; rejected
and errored events remain visible in counts and the event-level table.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cowler.align.hybrid_segment import (
    DEFAULT_BOUNDARY_TRIM_STEPS,
    HybridSegmentParams,
)
from scripts.evaluate_js445_segmentation import (
    COMPARISON_FIELDS,
    EventSteps,
    FittedLevel,
    fit_fold_models,
    load_event_truth,
    load_raw_pa,
    predict_cross_validated,
    predict_with_models,
    segment_event_steps,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTS = tuple(f"JS{number}" for number in range(445, 454))
TRANSFER_TARGETS = CONSTRUCTS[1:]
WITHIN_CONSTRUCT_OOF = "within_construct_oof"
JS445_FROZEN_TRANSFER = "js445_frozen_transfer"

METRIC_FIELDS = [
    "construct",
    "analysis",
    "n_events",
    "n_accepted",
    "n_rejected",
    "n_error",
    "acceptance_rate",
    "median_boundary_error_steps",
    "mean_boundary_error_steps",
    "median_absolute_error_steps",
    "mean_absolute_error_steps",
    "p90_absolute_error_steps",
    "max_absolute_error_steps",
    "exact_percent",
    "within_1_step_percent",
    "within_2_steps_percent",
]


def json_default(value: Any) -> Any:
    """Serialize numpy values and paths in the audit JSON."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def dataset_paths(
    construct: str, *, data_dir: Path, js445_dir: Path
) -> tuple[Path, Path]:
    """Resolve one raw/annotation pair without changing external state."""

    directory = js445_dir if construct == "JS445" else data_dir
    raw_path = directory / f"{construct}_synthetic.fast5"
    annotation_path = directory / f"{construct}_synthetic.annot.fast5"
    if not raw_path.is_file() or not annotation_path.is_file():
        raise FileNotFoundError(
            f"missing raw/annotation FAST5 pair for {construct} in {directory}"
        )
    return raw_path, annotation_path


def load_construct_events(
    construct: str,
    *,
    data_dir: Path,
    js445_dir: Path,
    sensitivity: float,
    min_level_length: int,
) -> tuple[list[EventSteps], float, Path, Path]:
    """Load, validate, and segment every event for one construct once."""

    raw_path, annotation_path = dataset_paths(
        construct, data_dir=data_dir, js445_dir=js445_dir
    )
    pa, sample_rate = load_raw_pa(raw_path)
    truths = load_event_truth(annotation_path)
    if max(truth.end_sample for truth in truths) > len(pa):
        raise ValueError(
            f"{construct} annotation event bounds extend beyond the raw signal"
        )
    events = [
        segment_event_steps(
            pa,
            truth,
            sensitivity=sensitivity,
            min_level_length=min_level_length,
        )
        for truth in truths
    ]
    return events, sample_rate, raw_path, annotation_path


def metric_row(
    construct: str, analysis: str, summary: dict[str, Any]
) -> dict[str, Any]:
    """Convert the shared summary into a compact manuscript-facing row."""

    def percent(key: str) -> float | str:
        value = summary.get(key)
        return "" if value is None else 100.0 * float(value)

    return {
        "construct": construct,
        "analysis": analysis,
        "n_events": int(summary["n_events"]),
        "n_accepted": int(summary["n_pass"]),
        "n_rejected": int(summary["n_rejected"]),
        "n_error": int(summary["n_error"]),
        "acceptance_rate": float(summary["acceptance_rate"]),
        "median_boundary_error_steps": summary.get("median_error_steps", ""),
        "mean_boundary_error_steps": summary.get("mean_error_steps", ""),
        "median_absolute_error_steps": summary.get(
            "median_absolute_error_steps", ""
        ),
        "mean_absolute_error_steps": summary.get("mean_absolute_error_steps", ""),
        "p90_absolute_error_steps": summary.get("p90_absolute_error_steps", ""),
        "max_absolute_error_steps": summary.get("max_absolute_error_steps", ""),
        "exact_percent": percent("exact_step_fraction"),
        "within_1_step_percent": percent("within_1_step_fraction"),
        "within_2_steps_percent": percent("within_2_steps_fraction"),
    }


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _model_rows(
    fits: Sequence[FittedLevel], *, training_construct: str, analysis: str
) -> list[dict[str, Any]]:
    return [
        {
            "training_construct": training_construct,
            "analysis": analysis,
            **asdict(fit),
        }
        for fit in fits
    ]


def _annotate_predictions(
    rows: Sequence[dict[str, Any]], *, construct: str, analysis: str
) -> list[dict[str, Any]]:
    return [
        {"construct": construct, "analysis": analysis, **row} for row in rows
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    js445_dir = Path(args.js445_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    constructs = tuple(str(value).upper() for value in args.constructs)
    invalid = sorted(set(constructs).difference(CONSTRUCTS))
    if invalid:
        raise ValueError(f"unsupported constructs: {invalid}")
    if "JS445" not in constructs:
        raise ValueError("JS445 is required to fit the frozen transfer model")

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

    loaded: dict[str, tuple[list[EventSteps], float, Path, Path]] = {}
    for construct in constructs:
        print(f"Segmenting {construct}...", flush=True)
        loaded[construct] = load_construct_events(
            construct,
            data_dir=data_dir,
            js445_dir=js445_dir,
            sensitivity=float(args.sensitivity),
            min_level_length=int(args.min_level_length),
        )

    js445_events = loaded["JS445"][0]
    frozen_models, frozen_fits = fit_fold_models(js445_events, fold=-1)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = _model_rows(
        frozen_fits,
        training_construct="JS445",
        analysis=JS445_FROZEN_TRANSFER,
    )
    detailed_summaries: list[dict[str, Any]] = []

    for construct in constructs:
        events, sample_rate, raw_path, annotation_path = loaded[construct]
        print(f"Evaluating {construct}: within-construct OOF...", flush=True)
        oof_rows, oof_fits = predict_cross_validated(
            events,
            n_folds=int(args.folds),
            params=params,
        )
        oof_summary = summarize(oof_rows, sample_rate)
        oof_summary.update(
            {
                "construct": construct,
                "analysis": WITHIN_CONSTRUCT_OOF,
                "scope": "event-held-out, within-construct and within-run",
                "model_training": (
                    f"{args.folds}-fold region models fitted from the same construct"
                ),
                "boundary_calibration": "none",
                "raw_path": str(raw_path),
                "annotation_path": str(annotation_path),
            }
        )
        metric_rows.append(metric_row(construct, WITHIN_CONSTRUCT_OOF, oof_summary))
        prediction_rows.extend(
            _annotate_predictions(
                oof_rows, construct=construct, analysis=WITHIN_CONSTRUCT_OOF
            )
        )
        model_rows.extend(
            _model_rows(
                oof_fits,
                training_construct=construct,
                analysis=WITHIN_CONSTRUCT_OOF,
            )
        )
        detailed_summaries.append(oof_summary)

        if construct != "JS445":
            print(f"Evaluating {construct}: frozen JS445 transfer...", flush=True)
            transfer_rows = predict_with_models(
                events,
                models=frozen_models,
                params=params,
                fold=-1,
            )
            transfer_summary = summarize(transfer_rows, sample_rate)
            transfer_summary.update(
                {
                    "construct": construct,
                    "analysis": JS445_FROZEN_TRANSFER,
                    "scope": "construct/run transfer from JS445",
                    "model_training": (
                        "one region model fitted from all JS445 events and applied "
                        "unchanged"
                    ),
                    "boundary_calibration": "none",
                    "target_construct_refit": False,
                    "raw_path": str(raw_path),
                    "annotation_path": str(annotation_path),
                }
            )
            metric_rows.append(
                metric_row(construct, JS445_FROZEN_TRANSFER, transfer_summary)
            )
            prediction_rows.extend(
                _annotate_predictions(
                    transfer_rows,
                    construct=construct,
                    analysis=JS445_FROZEN_TRANSFER,
                )
            )
            detailed_summaries.append(transfer_summary)

    metric_rows.sort(key=lambda row: (str(row["construct"]), str(row["analysis"])))
    prediction_rows.sort(
        key=lambda row: (
            str(row["construct"]),
            str(row["analysis"]),
            int(row["event_id"]),
        )
    )
    model_rows.sort(
        key=lambda row: (
            str(row["analysis"]),
            str(row["training_construct"]),
            int(row["fold"]),
            str(row["region"]),
        )
    )

    _write_csv(out_dir / "construct_metrics.csv", metric_rows, METRIC_FIELDS)
    _write_csv(
        out_dir / "event_predictions.csv",
        prediction_rows,
        ["construct", "analysis", *COMPARISON_FIELDS],
    )
    _write_csv(
        out_dir / "region_models.csv",
        model_rows,
        [
            "training_construct",
            "analysis",
            "fold",
            "region",
            "mean",
            "std",
            "n_steps",
        ],
    )

    result = {
        "constructs": list(constructs),
        "analyses": [WITHIN_CONSTRUCT_OOF, JS445_FROZEN_TRANSFER],
        "manual_boundary_field": "p_start_idx",
        "automatic_boundary_source": "semantic_boundary",
        "accuracy_denominator": "accepted events only",
        "step_error_axis": "observed segmented steps",
        "template_dna_shared_across_constructs": True,
        "n_folds": int(args.folds),
        "step_finder_sensitivity": float(args.sensitivity),
        "step_finder_min_level_length": int(args.min_level_length),
        "hybrid_segment_params": asdict(params),
        "summaries": detailed_summaries,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=ROOT / "tmp" / "otherfast5")
    parser.add_argument("--js445-dir", default=ROOT / "tmp")
    parser.add_argument(
        "--out-dir", default=ROOT / "tmp" / "dna_peptide_boundary_constructs"
    )
    parser.add_argument("--constructs", nargs="+", default=CONSTRUCTS)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--min-level-length", type=int, default=2)
    parser.add_argument(
        "--boundary-trim-steps",
        type=int,
        default=DEFAULT_BOUNDARY_TRIM_STEPS,
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
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    compact = [metric_row(s["construct"], s["analysis"], s) for s in result["summaries"]]
    print(json.dumps(compact, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
