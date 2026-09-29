"""Run gated Stage 3 of the PSK medoid-selection robustness experiment.

This runner consumes only the accepted Stage 1/2 artifacts.  It independently
recomputes the frozen baseline score grid from held-out analysis traces and cached
rank-1 consensuses, then validates that grid against the serialized baseline
classification outputs.  Alternative consensuses are not scored unless the complete
baseline gate passes.  After the gate, each of the 36 runs replaces exactly one
rank-1 consensus with rank 2--5 and retains all per-read and confusion-matrix output.

Stage 4 summaries, heatmaps, and optional analyses are deliberately out of scope.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from cowler.eval.peptide_consensus import (
    ClassificationMetrics,
    classification_metrics,
    fixed_profile_scores,
)
from scripts.evaluate_medoid_robustness import (
    DATASETS,
    ROOT,
    _read_csv,  # pyright: ignore[reportPrivateUsage]
    _repository_version,  # pyright: ignore[reportPrivateUsage]
    _sha256,  # pyright: ignore[reportPrivateUsage]
)
from scripts.evaluate_psk_dba_consensus import write_csv

DEFAULT_OUT_DIR = ROOT / "tmp" / "medoid_robustness"
N_PERTURBATIONS = 36
BASELINE_SCORE_ATOL = 1e-12
KNOWN_POSINF = ("JS448", 108, "JS453")
FloatArr = npt.NDArray[np.float64]
Int64Arr = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CachedProfile:
    """The immutable cached profile channels required by fixed-profile scoring."""

    mean: FloatArr
    std: FloatArr
    supported: npt.NDArray[np.bool_]
    candidate_event_id: int


@dataclass(frozen=True)
class HeldoutCohort:
    """Frozen held-out analysis traces in their accepted event order."""

    identities: tuple[tuple[str, int], ...]
    reads: tuple[tuple[FloatArr, FloatArr], ...]
    truth: Int64Arr


@dataclass(frozen=True)
class BaselineReference:
    """Serialized baseline outputs used only as validation references."""

    identities: tuple[tuple[str, int], ...]
    scores: FloatArr
    predictions: tuple[str, ...]
    confusion: Int64Arr
    recall: FloatArr
    accuracy: float
    balanced_accuracy: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _require_historical_provenance(manifest: Mapping[str, Any], label: str) -> None:
    """Require auditable repository provenance for historical or current runs."""

    repository = manifest.get("repository")
    if not isinstance(repository, dict):
        raise RuntimeError(f"{label} manifest has no repository provenance")
    commit = repository.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError(f"{label} repository commit provenance is invalid")
    if not isinstance(repository.get("worktree_dirty"), bool):
        raise RuntimeError(f"{label} repository dirty-worktree provenance is invalid")


def _verify_record(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise RuntimeError(f"accepted artifact hash differs for {label}: {path}")
    return path


def _accepted_manifests(
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    paths = {
        "stage1": out_dir / "stage1_run_manifest.json",
        "stage2": out_dir / "stage2_run_manifest.json",
        "stage2_followup": out_dir / "stage2_followup_run_manifest.json",
    }
    stage1 = _load_json(paths["stage1"])
    stage2 = _load_json(paths["stage2"])
    followup = _load_json(paths["stage2_followup"])
    for label, manifest in (
        ("Stage 1", stage1),
        ("Stage 2", stage2),
        ("Stage 2 follow-up", followup),
    ):
        _require_historical_provenance(manifest, label)
    expected_repository = stage1.get("repository")
    if stage2.get("repository") != expected_repository or followup.get("repository") != expected_repository:
        raise RuntimeError("Stage 1/2 repository provenance differs within the accepted run")

    if not bool(stage1.get("validation", {}).get("all_stage1_checks_pass")):
        raise RuntimeError("Stage 1 validation is not accepted")
    if stage1.get("validation", {}).get("rank_1_matches_production") != "9/9":
        raise RuntimeError("Stage 1 did not reproduce all production medoids")
    if not bool(stage2.get("rank1_reproduction_gate", {}).get("passed")):
        raise RuntimeError("Stage 2 rank-1 reproduction gate is not accepted")
    if not bool(stage2.get("convergence", {}).get("all_45_converged")):
        raise RuntimeError("Stage 3 requires all 45 accepted DBA fits to converge")
    if followup.get("metric", {}).get("stored_stage2_consensus_values_reproduced") != "36/36":
        raise RuntimeError("Stage 2 descriptive follow-up is not accepted")

    stage1_hash = _sha256(paths["stage1"])
    stage2_hash = _sha256(paths["stage2"])
    stage2_stage1_hash = stage2.get("provenance", {}).get("stage1_manifest_sha256")
    followup_provenance = followup.get("provenance", {})
    if stage2_stage1_hash != stage1_hash:
        raise RuntimeError("Stage 2 is not linked to the exact accepted Stage 1 manifest")
    if followup_provenance.get("stage1_manifest", {}).get("sha256") != stage1_hash:
        raise RuntimeError("Stage 2 follow-up Stage 1 manifest hash differs")
    if followup_provenance.get("stage2_manifest", {}).get("sha256") != stage2_hash:
        raise RuntimeError("Stage 2 follow-up Stage 2 manifest hash differs")

    inputs = stage1.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise RuntimeError("Stage 1 manifest has no accepted input hashes")
    for label, record in inputs.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid accepted Stage 1 input record: {label}")
        _verify_record(record, f"Stage 1/{label}")

    output_records = followup.get("outputs", {})
    for label in ("detail", "per_construct_summary", "figure"):
        record = output_records.get(label)
        if not isinstance(record, dict):
            raise RuntimeError(f"missing accepted Stage 2 follow-up output: {label}")
        relative = Path(str(record["path"]))
        expected = str(record["sha256"])
        path = out_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"accepted Stage 2 follow-up output differs: {path}")
    return stage1, stage2, followup, paths


def _load_heldout_cohort(stage1: Mapping[str, Any]) -> HeldoutCohort:
    inputs = stage1["inputs"]
    trace_record = inputs.get("analysis_trace_cache", inputs.get("native_trace_cache"))
    if not isinstance(trace_record, dict):
        raise RuntimeError("Stage 1 manifest lacks an accepted analysis trace cache")
    trace_path = Path(str(trace_record["path"]))
    trace_rows_path = Path(str(inputs["trace_cache_companion_rows"]["path"]))
    split_path = Path(str(inputs["baseline_split"]["path"]))
    trace_rows = _read_csv(trace_rows_path)
    split_rows = [
        row
        for row in _read_csv(split_path)
        if row.get("dataset") in DATASETS
        and row.get("analysis") == "primary_split"
        and row.get("active", "1") == "1"
    ]
    split_by_identity = {
        (row["dataset"], int(row["event_id"])): row for row in split_rows
    }
    if len(split_by_identity) != len(split_rows):
        raise RuntimeError("accepted split contains duplicate event identities")

    identities: list[tuple[str, int]] = []
    reads: list[tuple[FloatArr, FloatArr]] = []
    with np.load(trace_path, allow_pickle=False) as cached:
        cached_datasets = np.asarray(cached["dataset"]).astype(str)
        cached_event_ids = np.asarray(cached["event_id"], dtype=np.int64)
        cached_event_orders = np.asarray(cached["event_order"], dtype=np.int64)
        if cached_datasets.size != len(trace_rows):
            raise RuntimeError("analysis trace cache and companion row counts differ")
        for index, row in enumerate(trace_rows):
            identity = (row["dataset"], int(row["event_id"]))
            if identity not in split_by_identity:
                raise RuntimeError(f"analysis trace lacks accepted split identity {identity}")
            if (
                cached_datasets[index] != identity[0]
                or int(cached_event_ids[index]) != identity[1]
                or int(cached_event_orders[index]) != int(row["event_order"])
            ):
                raise RuntimeError(f"analysis trace cache axis differs at row {index}")
            if split_by_identity[identity]["group"] != "held_out":
                continue
            mean = np.asarray(cached[f"mean_{index}"], dtype=float)
            std = np.asarray(cached[f"std_{index}"], dtype=float)
            if (
                mean.ndim != 1
                or std.shape != mean.shape
                or mean.size != int(row["n_steps"])
                or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(std))
            ):
                raise RuntimeError(f"invalid frozen analysis trace for {identity}")
            identities.append(identity)
            reads.append((mean, std))

    expected_counts = stage1.get("frozen_heldout_counts", {})
    observed_counts = {
        dataset: sum(identity[0] == dataset for identity in identities)
        for dataset in DATASETS
    }
    if len(identities) != sum(int(value) for value in expected_counts.values()) or observed_counts != expected_counts:
        raise RuntimeError(
            f"frozen held-out cohort differs: total={len(identities)}, counts={observed_counts}"
        )
    truth = np.asarray([DATASETS.index(dataset) for dataset, _ in identities], dtype=np.int64)
    return HeldoutCohort(tuple(identities), tuple(reads), truth)


def _load_profiles(
    out_dir: Path,
    followup: Mapping[str, Any],
) -> dict[str, dict[int, CachedProfile]]:
    expected_hashes = followup.get("provenance", {}).get("consensus_caches")
    if not isinstance(expected_hashes, dict) or len(expected_hashes) != 45:
        raise RuntimeError("accepted Stage 2 follow-up does not hash all 45 consensuses")
    profiles: dict[str, dict[int, CachedProfile]] = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        for rank in range(1, 6):
            key = f"{dataset}/rank{rank}.npz"
            path = out_dir / "consensuses" / key
            if expected_hashes.get(key) != _sha256(path):
                raise RuntimeError(f"accepted consensus hash differs: {path}")
            with np.load(path, allow_pickle=False) as cached:
                if cached["construct"].item() != dataset:
                    raise RuntimeError(f"cached consensus construct differs: {path}")
                if int(cached["candidate_rank"].item()) != rank:
                    raise RuntimeError(f"cached consensus rank differs: {path}")
                if not bool(cached["converged"].item()):
                    raise RuntimeError(f"cached consensus did not converge: {path}")
                mean = np.asarray(cached["mean"], dtype=float)
                std = np.asarray(cached["std"], dtype=float)
                supported = np.asarray(cached["supported"], dtype=bool)
                if (
                    mean.ndim != 1
                    or std.shape != mean.shape
                    or supported.shape != mean.shape
                    or int(cached["consensus_length"].item()) != mean.size
                ):
                    raise RuntimeError(f"cached consensus channels differ: {path}")
                profiles[dataset][rank] = CachedProfile(
                    mean=mean,
                    std=std,
                    supported=supported,
                    candidate_event_id=int(cached["candidate_event_id"].item()),
                )
    return profiles


def _method_rows(path: Path, method: str) -> list[dict[str, str]]:
    rows = [row for row in _read_csv(path) if row.get("method") == method]
    if not rows:
        raise RuntimeError(f"{method} validation reference is empty: {path}")
    return rows


def _load_baseline_reference(
    baseline_dir: Path,
    *,
    method: str,
    files: Mapping[str, Any],
    expected_events: int,
) -> BaselineReference:
    event_rows = _method_rows(
        baseline_dir / str(files.get("heldout_event_scores", "heldout_event_scores.csv")),
        method,
    )
    if len(event_rows) != expected_events:
        raise RuntimeError(
            f"{method} baseline reference does not contain {expected_events} events"
        )
    identities = tuple(
        (row["dataset"], int(row["event_id"])) for row in event_rows
    )
    if len(set(identities)) != len(identities):
        raise RuntimeError("baseline reference contains duplicate events")
    scores = np.asarray(
        [[float(row[f"score_{label}"]) for label in DATASETS] for row in event_rows],
        dtype=float,
    )
    predictions = tuple(row["predicted_class"] for row in event_rows)

    confusion_rows = _method_rows(
        baseline_dir
        / str(files.get("classification_confusion_matrix", "classification_confusion_matrix.csv")),
        method,
    )
    confusion = np.zeros((len(DATASETS), len(DATASETS)), dtype=np.int64)
    seen_confusion: set[tuple[str, str]] = set()
    for row in confusion_rows:
        identity = (row["true_class"], row["predicted_class"])
        seen_confusion.add(identity)
        confusion[DATASETS.index(identity[0]), DATASETS.index(identity[1])] = int(
            row["count"]
        )
    if len(seen_confusion) != len(DATASETS) ** 2:
        raise RuntimeError(f"{method} confusion-matrix validation reference is incomplete")

    per_class_rows = _method_rows(
        baseline_dir / str(files.get("classification_per_class", "classification_per_class.csv")),
        method,
    )
    per_class = {row["class"]: float(row["recall"]) for row in per_class_rows}
    if set(per_class) != set(DATASETS):
        raise RuntimeError(f"{method} per-class validation reference is incomplete")
    summary_rows = _method_rows(
        baseline_dir / str(files.get("classification_summary", "classification_summary.csv")),
        method,
    )
    if len(summary_rows) != 1:
        raise RuntimeError(f"{method} baseline summary validation reference is ambiguous")
    summary = summary_rows[0]
    return BaselineReference(
        identities=identities,
        scores=scores,
        predictions=predictions,
        confusion=confusion,
        recall=np.asarray([per_class[label] for label in DATASETS], dtype=float),
        accuracy=float(summary["accuracy"]),
        balanced_accuracy=float(summary["balanced_accuracy"]),
    )


def _validate_score_values(
    scores: npt.ArrayLike,
    label: str,
    *,
    require_admissible_per_event: bool = True,
) -> FloatArr:
    grid = np.asarray(scores, dtype=float)
    if np.any(np.isnan(grid)):
        raise RuntimeError(f"{label} contains NaN")
    if np.any(np.isneginf(grid)):
        raise RuntimeError(f"{label} contains negative infinity")
    if require_admissible_per_event and np.any(~np.any(np.isfinite(grid), axis=1)):
        raise RuntimeError(f"{label} contains an event with no admissible score")
    return grid


def _baseline_gate(
    cohort: HeldoutCohort,
    computed_scores: npt.ArrayLike,
    reference: BaselineReference,
    *,
    known_positive_infinity: tuple[str, int, str] | None = KNOWN_POSINF,
) -> tuple[ClassificationMetrics, list[dict[str, Any]]]:
    scores = _validate_score_values(computed_scores, "computed baseline score grid")
    expected_shape = (len(cohort.identities), len(DATASETS))
    if scores.shape != expected_shape or reference.scores.shape != expected_shape:
        raise RuntimeError(
            f"baseline score-grid shape differs: {scores.shape}, {reference.scores.shape}"
        )
    reference_scores = _validate_score_values(
        reference.scores, "saved baseline validation reference"
    )
    computed_posinf = np.isposinf(scores)
    reference_posinf = np.isposinf(reference_scores)
    known_mask: npt.NDArray[np.bool_] | None = None
    if known_positive_infinity is not None:
        known_mask = np.zeros(expected_shape, dtype=bool)
        try:
            known_row = cohort.identities.index(known_positive_infinity[:2])
        except ValueError as error:
            raise RuntimeError(
                "declared +inf event is absent from held-out cohort"
            ) from error
        known_column = DATASETS.index(known_positive_infinity[2])
        known_mask[known_row, known_column] = True
    finite = np.isfinite(scores) & np.isfinite(reference_scores)
    finite_match = bool(
        np.allclose(
            scores[finite],
            reference_scores[finite],
            rtol=0.0,
            atol=BASELINE_SCORE_ATOL,
        )
    )
    max_difference = (
        float(np.max(np.abs(scores[finite] - reference_scores[finite])))
        if np.any(finite)
        else float("nan")
    )
    metrics = classification_metrics(cohort.truth, scores, DATASETS)
    predictions = tuple(DATASETS[int(index)] for index in metrics.predicted_index)
    checks = [
        {
            "check": "score_grid_shape_matches_frozen_cohort",
            "passed": int(scores.shape == expected_shape),
            "detail": str(scores.shape),
        },
        {
            "check": "event_identity_and_order_exact",
            "passed": int(cohort.identities == reference.identities),
            "detail": f"{len(cohort.identities)} events",
        },
        {
            "check": "computed_rejects_nan_and_negative_infinity",
            "passed": int(not np.any(np.isnan(scores)) and not np.any(np.isneginf(scores))),
            "detail": f"nan={int(np.sum(np.isnan(scores)))}, -inf={int(np.sum(np.isneginf(scores)))}",
        },
        {
            "check": "positive_infinity_mask_exact",
            "passed": int(np.array_equal(computed_posinf, reference_posinf)),
            "detail": f"computed={int(np.sum(computed_posinf))}, reference={int(np.sum(reference_posinf))}",
        },
        {
            "check": "finite_scores_within_absolute_tolerance",
            "passed": int(finite_match),
            "detail": f"rtol=0, atol={BASELINE_SCORE_ATOL}, max_abs_difference={max_difference}",
        },
        {
            "check": "predictions_exact",
            "passed": int(predictions == reference.predictions),
            "detail": f"{len(predictions)} predictions",
        },
        {
            "check": "confusion_matrix_exact",
            "passed": int(np.array_equal(metrics.confusion, reference.confusion)),
            "detail": f"total={int(np.sum(metrics.confusion))}",
        },
        {
            "check": "accuracy_exact",
            "passed": int(metrics.accuracy == reference.accuracy),
            "detail": f"computed={metrics.accuracy!r}, reference={reference.accuracy!r}",
        },
        {
            "check": "balanced_accuracy_exact",
            "passed": int(metrics.balanced_accuracy == reference.balanced_accuracy),
            "detail": (
                f"computed={metrics.balanced_accuracy!r}, "
                f"reference={reference.balanced_accuracy!r}"
            ),
        },
        {
            "check": "per_class_recall_exact",
            "passed": int(np.array_equal(metrics.recall, reference.recall)),
            "detail": "JS445--JS453 order",
        },
    ]
    if known_positive_infinity is not None and known_mask is not None:
        checks.insert(
            5,
            {
                "check": "known_single_positive_infinity_exact",
                "passed": int(np.array_equal(computed_posinf, known_mask)),
                "detail": (
                    f"{known_positive_infinity[0]}/event{known_positive_infinity[1]} "
                    f"-> {known_positive_infinity[2]}"
                ),
            },
        )
    return metrics, checks


def _one_at_a_time_grid(
    baseline_scores: npt.ArrayLike,
    column: int,
    alternative_scores: npt.ArrayLike,
) -> FloatArr:
    """Copy a baseline grid and replace exactly one construct-score column."""

    baseline = np.asarray(baseline_scores, dtype=float)
    alternative = np.asarray(alternative_scores, dtype=float)
    if baseline.ndim != 2 or not 0 <= column < baseline.shape[1]:
        raise ValueError("invalid baseline score grid or replacement column")
    if alternative.shape != (baseline.shape[0],):
        raise ValueError("alternative scores must have one value per event")
    result = baseline.copy()
    result[:, column] = alternative
    other_columns = [index for index in range(baseline.shape[1]) if index != column]
    if not np.array_equal(result[:, other_columns], baseline[:, other_columns]):
        raise RuntimeError("one-at-a-time substitution changed another score column")
    return result


def _run_metadata(run_id: str, construct: str, rank: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "perturbed_construct": construct,
        "candidate_rank": rank,
    }


def _result_rows(
    cohort: HeldoutCohort,
    scores: FloatArr,
    metrics: ClassificationMetrics,
    baseline_metrics: ClassificationMetrics,
    *,
    run_id: str,
    construct: str,
    rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_predictions = baseline_metrics.predicted_index
    changed = metrics.predicted_index != baseline_predictions
    metadata = _run_metadata(run_id, construct, rank)
    summary = {
        **metadata,
        "n_events": len(cohort.identities),
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "delta_accuracy": metrics.accuracy - baseline_metrics.accuracy,
        "delta_balanced_accuracy": (
            metrics.balanced_accuracy - baseline_metrics.balanced_accuracy
        ),
        "n_prediction_flips": int(np.sum(changed)),
        "prediction_flip_rate": float(np.mean(changed)),
        "n_inadmissible_scores": int(np.sum(np.isposinf(scores))),
        "n_events_with_inadmissible_score": int(
            np.sum(np.any(np.isposinf(scores), axis=1))
        ),
    }
    per_class = [
        {
            **metadata,
            "class": label,
            "count": int(metrics.counts[index]),
            "recall": float(metrics.recall[index]),
        }
        for index, label in enumerate(DATASETS)
    ]
    confusion = [
        {
            **metadata,
            "true_class": true_label,
            "predicted_class": predicted_label,
            "count": int(metrics.confusion[i, j]),
            "row_normalized": float(metrics.confusion[i, j] / metrics.counts[i]),
        }
        for i, true_label in enumerate(DATASETS)
        for j, predicted_label in enumerate(DATASETS)
    ]
    predictions: list[dict[str, Any]] = []
    for row, ((true_label, event_id), is_changed) in enumerate(
        zip(cohort.identities, changed)
    ):
        result = {
            **metadata,
            "heldout_event_id": event_id,
            "event_dataset": true_label,
            "true_class": DATASETS[int(cohort.truth[row])],
            "baseline_prediction": DATASETS[int(baseline_predictions[row])],
            "perturbed_prediction": DATASETS[int(metrics.predicted_index[row])],
            "changed": int(is_changed),
        }
        for column, label in enumerate(DATASETS):
            result[f"score_{label}"] = scores[row, column]
        predictions.append(result)
    return summary, per_class, confusion, predictions


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    classification_dir = out_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    stage1, _, followup, manifest_paths = _accepted_manifests(out_dir)
    cohort = _load_heldout_cohort(stage1)
    profiles = _load_profiles(out_dir, followup)
    baseline_dir = Path(str(stage1["baseline"]))
    baseline_method = str(stage1.get("baseline_method", "native"))
    baseline_files = stage1.get("baseline_files", {})
    if not isinstance(baseline_files, dict):
        raise RuntimeError("Stage 1 baseline file mapping is invalid")
    reference_paths = {
        "heldout_event_scores": baseline_dir
        / str(baseline_files.get("heldout_event_scores", "heldout_event_scores.csv")),
        "classification_summary": baseline_dir
        / str(baseline_files.get("classification_summary", "classification_summary.csv")),
        "classification_per_class": baseline_dir
        / str(baseline_files.get("classification_per_class", "classification_per_class.csv")),
        "classification_confusion_matrix": (
            baseline_dir
            / str(
                baseline_files.get(
                    "classification_confusion_matrix",
                    "classification_confusion_matrix.csv",
                )
            )
        ),
    }
    for path in reference_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = _load_baseline_reference(
        baseline_dir,
        method=baseline_method,
        files=baseline_files,
        expected_events=len(cohort.identities),
    )

    print(
        f"1. Independently recomputing the frozen {len(cohort.identities)} x 9 "
        "rank-1 baseline...",
        flush=True,
    )
    labels, baseline_scores = fixed_profile_scores(
        cohort.reads,
        {dataset: profiles[dataset][1] for dataset in DATASETS},
        max_run=3,
    )
    if labels != DATASETS:
        raise RuntimeError("baseline profile score label order changed")
    known_positive_infinity = (
        KNOWN_POSINF
        if baseline_method == "native"
        and stage1.get("split", {}).get("strategy") == "post_filter_stratified_random"
        else None
    )
    baseline_metrics, gate_rows = _baseline_gate(
        cohort,
        baseline_scores,
        reference,
        known_positive_infinity=known_positive_infinity,
    )
    gate_path = classification_dir / "stage3_validation_checks.csv"
    write_csv(gate_path, gate_rows)
    failed = [row["check"] for row in gate_rows if not bool(row["passed"])]
    if failed:
        failure = {
            "stage": 3,
            "baseline_gate_passed": False,
            "failed_checks": failed,
            "perturbations_scored": False,
            "finite_score_tolerance": {"rtol": 0.0, "atol": BASELINE_SCORE_ATOL},
        }
        (classification_dir / "stage3_gate_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            "Stage 3 baseline gate failed before perturbations: " + ", ".join(failed)
        )

    print("2. Baseline gate passed; scoring 36 one-at-a-time substitutions...", flush=True)
    summaries: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    baseline_result = _result_rows(
        cohort,
        baseline_scores,
        baseline_metrics,
        baseline_metrics,
        run_id="baseline",
        construct="",
        rank=1,
    )
    summaries.append(baseline_result[0])
    per_class_rows.extend(baseline_result[1])
    confusion_rows.extend(baseline_result[2])
    prediction_rows.extend(baseline_result[3])

    for dataset_index, dataset in enumerate(DATASETS):
        for rank in range(2, 6):
            alt_labels, alternative = fixed_profile_scores(
                cohort.reads,
                {dataset: profiles[dataset][rank]},
                max_run=3,
            )
            if alt_labels != (dataset,):
                raise RuntimeError(f"{dataset} rank {rank}: score label changed")
            alternative_column = _validate_score_values(
                alternative,
                f"{dataset} rank {rank} alternative scores",
                require_admissible_per_event=False,
            )[:, 0]
            scores = _one_at_a_time_grid(
                baseline_scores, dataset_index, alternative_column
            )
            scores = _validate_score_values(scores, f"{dataset} rank {rank} score grid")
            metrics = classification_metrics(cohort.truth, scores, DATASETS)
            run_id = f"{dataset}_rank{rank}"
            result = _result_rows(
                cohort,
                scores,
                metrics,
                baseline_metrics,
                run_id=run_id,
                construct=dataset,
                rank=rank,
            )
            summaries.append(result[0])
            per_class_rows.extend(result[1])
            confusion_rows.extend(result[2])
            prediction_rows.extend(result[3])
            print(
                f"   {run_id}: delta_balanced_accuracy="
                f"{result[0]['delta_balanced_accuracy']:+.6f}, "
                f"flips={result[0]['n_prediction_flips']}",
                flush=True,
            )

    if (
        len(summaries) != N_PERTURBATIONS + 1
        or len(prediction_rows)
        != (N_PERTURBATIONS + 1) * len(cohort.identities)
        or len(confusion_rows) != (N_PERTURBATIONS + 1) * len(DATASETS) ** 2
    ):
        raise RuntimeError("Stage 3 result cardinality differs from the 37-run design")
    write_csv(classification_dir / "run_metrics.csv", summaries)
    write_csv(classification_dir / "per_class_recall.csv", per_class_rows)
    write_csv(classification_dir / "confusion_matrices.csv", confusion_rows)
    write_csv(classification_dir / "predictions.csv", prediction_rows)

    consensus_hashes = followup["provenance"]["consensus_caches"]
    output_records = {
        label: {
            "path": str(path.relative_to(out_dir)),
            "sha256": _sha256(path),
        }
        for label, path in {
            "run_metrics": classification_dir / "run_metrics.csv",
            "per_class_recall": classification_dir / "per_class_recall.csv",
            "confusion_matrices": classification_dir / "confusion_matrices.csv",
            "predictions": classification_dir / "predictions.csv",
            "validation_checks": gate_path,
        }.items()
    }
    stage3_manifest = {
        "stage": 3,
        "authorized_scope": "medoid robustness plan sections 3.1--3.5 only",
        "stage_4_implemented": False,
        "optional_analyses_implemented": False,
        "heatmaps_generated": False,
        "implementation_repository": _repository_version(),
        "stage1_stage2_repository_provenance_preserved": stage1["repository"],
        "datasets": list(DATASETS),
        "baseline_method": baseline_method,
        "classification": {
            "n_heldout_events": len(cohort.identities),
            "n_classes": len(DATASETS),
            "n_perturbations": N_PERTURBATIONS,
            "n_runs_including_baseline": len(summaries),
            "score": "path-normalized uncertainty-aware Gaussian-DTW",
            "score_direction": "lower",
            "dtw": "closed symmetric1 max_run=3",
            "one_construct_substituted_per_run": True,
        },
        "baseline_gate": {
            "passed": True,
            "baseline_independently_recomputed": True,
            "saved_baseline_outputs_used_only_as_validation_references": True,
            "score_grid_shape": [len(cohort.identities), len(DATASETS)],
            "finite_comparison": {"rtol": 0.0, "atol": BASELINE_SCORE_ATOL},
            "nan_rejected": True,
            "negative_infinity_rejected": True,
            "positive_infinity_mask_exact": True,
            "known_single_positive_infinity": (
                {
                    "event_dataset": known_positive_infinity[0],
                    "event_id": known_positive_infinity[1],
                    "profile": known_positive_infinity[2],
                }
                if known_positive_infinity is not None
                else None
            ),
            "event_identity_and_order_exact": True,
            "predictions_exact": True,
            "confusion_matrix_exact": True,
            "accuracy_exact": True,
            "balanced_accuracy_exact": True,
            "per_class_recall_exact": True,
            "checks_file": "classification/stage3_validation_checks.csv",
        },
        "scope_guards": {
            "filtering_rerun": False,
            "splitting_rerun": False,
            "preprocessing_rerun": False,
            "candidate_selection_rerun": False,
            "dba_rerun": False,
            "balanced_accuracy_heatmap_generated": False,
            "prediction_flip_heatmap_generated": False,
            "stage4_summary_join_generated": False,
        },
        "provenance": {
            "accepted_manifests": {
                label: {"path": str(path), "sha256": _sha256(path)}
                for label, path in manifest_paths.items()
            },
            "accepted_stage1_input_artifacts": stage1["inputs"],
            "accepted_consensus_caches": consensus_hashes,
            "baseline_validation_references": {
                label: {"path": str(path), "sha256": _sha256(path)}
                for label, path in reference_paths.items()
            },
        },
        "baseline_metrics": summaries[0],
        "outputs": output_records,
    }
    stage3_path = out_dir / "stage3_run_manifest.json"
    stage3_path.write_text(
        json.dumps(stage3_manifest, indent=2) + "\n", encoding="utf-8"
    )

    combined = _load_json(out_dir / "run_manifest.json")
    _require_historical_provenance(combined, "combined Stage 1/2")
    combined["stage"] = 3
    combined["stages_2_or_3_implemented"] = {"stage_2": True, "stage_3": True}
    combined["stage3"] = {
        "manifest": "stage3_run_manifest.json",
        "manifest_sha256": _sha256(stage3_path),
        "baseline_gate_passed": True,
        "n_perturbations": N_PERTURBATIONS,
        "stage_4_implemented": False,
    }
    outputs = combined.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("combined Stage 1/2 output manifest is invalid")
    combined["outputs"] = {**outputs, **output_records}
    (out_dir / "run_manifest.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Stage 3 complete: baseline gate passed; {N_PERTURBATIONS} perturbations "
        f"and all 37 confusion matrices saved under {classification_dir}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
