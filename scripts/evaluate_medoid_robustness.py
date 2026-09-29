"""Run Stage 1 of the PSK medoid-selection robustness experiment.

This runner deliberately stops after candidate identification, metadata export,
validation, and the 3 x 3 candidate-trace figure.  It does not fit DBA profiles or
perform classification.  The frozen baseline supplies retained-event and active-split
identities; the original unshifted within-dataset Gaussian-DTW cache supplies scores.

See ``docs/medoid_robustness_plan.md`` for the experimental contract and gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from cowler.consensus.cluster import rank_medoid_candidates
from scripts.evaluate_psk_dba_consensus import write_csv

ROOT = Path(__file__).resolve().parents[1]
DATASETS = tuple(f"JS{number}" for number in range(445, 454))
DEFAULT_BASELINE_DIR = ROOT / "tmp" / "psk_length_normalized_classification"
DEFAULT_DISTANCE_CACHE = ROOT / "tmp" / "outlier_up" / "within_dataset_dtw_distances.npz"
DEFAULT_DISTANCE_ROWS = ROOT / "tmp" / "outlier_up" / "upstream_outlier_metrics.csv"
DEFAULT_TRACE_CACHE = ROOT / "tmp" / "all_cluster" / "native_filtered_traces.npz"
DEFAULT_TRACE_ROWS = ROOT / "tmp" / "all_cluster" / "trace_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "tmp" / "medoid_robustness"
TOP_K = 5
FIGURE_DPI = 300
FIGURE_FONT_SIZE = 12
ROBUSTNESS_Y_LIMITS = (0.23, 0.72)
HIGHLIGHT_LINE_WIDTH = 2.8
OTHER_LINE_WIDTH = 1.5
FloatArr = npt.NDArray[np.float64]
Int64Arr = npt.NDArray[np.int64]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_version() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"commit": commit, "worktree_dirty": dirty}


def _dataset_rows(
    rows: Sequence[Mapping[str, str]], dataset: str
) -> list[Mapping[str, str]]:
    return [row for row in rows if row.get("dataset") == dataset]


def _require_unique_ids(rows: Sequence[Mapping[str, str]], label: str) -> None:
    keys = [(str(row["dataset"]), int(row["event_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{label} contains duplicate dataset/event identities")


def _load_baseline(
    baseline_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int],
]:
    summary_path = baseline_dir / "run_summary.json"
    split_path = baseline_dir / "split_manifest.csv"
    outlier_path = baseline_dir / "upstream_outlier_metrics.csv"
    diagnostics_path = baseline_dir / "profile_diagnostics.csv"
    for path in (summary_path, split_path, outlier_path, diagnostics_path):
        if not path.exists():
            raise FileNotFoundError(f"missing frozen baseline artifact: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("consensus", {}).get("current_normalize") is not False:
        raise RuntimeError("baseline must record consensus current_normalize=false")
    if summary.get("outlier_filter", {}).get("input_axis") != "native peptide steps":
        raise RuntimeError("baseline outlier input is not the native peptide-step axis")
    split_strategy = summary.get("split", {}).get("strategy")
    if split_strategy not in {"post_filter_stratified_random", "stored_isConsensus"}:
        raise RuntimeError(f"unsupported frozen baseline split: {split_strategy}")
    baseline_method = str(summary.get("baseline_method", "native"))

    split_rows = [
        row
        for row in _read_csv(split_path)
        if row.get("dataset") in DATASETS
        and row.get("analysis") == "primary_split"
        and row.get("active", "1") == "1"
    ]
    outlier_rows = [
        row
        for row in _read_csv(outlier_path)
        if row.get("dataset") in DATASETS
    ]
    diagnostics = [
        row
        for row in _read_csv(diagnostics_path)
        if row.get("dataset") in DATASETS and row.get("method") == baseline_method
    ]
    _require_unique_ids(split_rows, "baseline split")
    _require_unique_ids(outlier_rows, "baseline outlier table")
    if len(diagnostics) != len(DATASETS):
        raise RuntimeError(
            f"baseline must contain one {baseline_method} profile diagnostic per construct"
        )
    production_medoids = {
        str(row["dataset"]): int(row["medoid_event_id"]) for row in diagnostics
    }
    return summary, split_rows, outlier_rows, production_medoids


def _validate_cache_lineage(
    baseline_outliers: Sequence[Mapping[str, str]],
    distance_rows: Sequence[Mapping[str, str]],
    trace_rows: Sequence[Mapping[str, str]],
) -> None:
    """Cross-check companion row order and retained identities across old caches."""
    _require_unique_ids(distance_rows, "distance-cache companion rows")
    _require_unique_ids(trace_rows, "trace-cache companion rows")
    trace_keys = {(str(row["dataset"]), int(row["event_id"])) for row in trace_rows}
    retained_keys: set[tuple[str, int]] = set()
    for dataset in DATASETS:
        baseline = _dataset_rows(baseline_outliers, dataset)
        cached = _dataset_rows(distance_rows, dataset)
        if len(baseline) != len(cached):
            raise RuntimeError(f"{dataset}: baseline and distance-cache row counts differ")
        for baseline_row, cache_row in zip(baseline, cached):
            identity = (dataset, int(baseline_row["event_id"]))
            if identity != (dataset, int(cache_row["event_id"])):
                raise RuntimeError(f"{dataset}: distance-cache event order differs")
            if int(baseline_row["event_order"]) != int(cache_row["event_order"]):
                raise RuntimeError(f"{dataset}: distance-cache input index differs")
            if int(baseline_row["retained"]) != int(cache_row["retained"]):
                raise RuntimeError(f"{dataset}: cached and baseline retained flags differ")
            for field in ("normalized_dtw_to_medoid", "knn_distance", "zmedoid", "zknn"):
                if not np.isclose(
                    float(baseline_row[field]),
                    float(cache_row[field]),
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise RuntimeError(f"{dataset}: cache lineage differs in {field}")
            if int(baseline_row["retained"]):
                retained_keys.add(identity)
    if trace_keys != retained_keys:
        missing = sorted(retained_keys - trace_keys)[:5]
        extra = sorted(trace_keys - retained_keys)[:5]
        raise RuntimeError(
            "analysis trace cache does not equal frozen retained cohort; "
            f"missing={missing}, extra={extra}"
        )


def _plot_candidates(
    output_path: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    trace_by_id: Mapping[tuple[str, int], FloatArr],
    *,
    x_label: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(24, 16), constrained_layout=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        rows = [row for row in candidate_rows if row["construct"] == dataset]
        for row in reversed(rows):
            rank = int(row["candidate_rank"])
            event_id = int(row["original_read_id"])
            trace = trace_by_id[(dataset, event_id)]
            emphasized = rank == 1
            label = f"#{rank}, ΔD={float(row['delta_mean_score']):.2f}"
            axis.plot(
                np.arange(trace.size),
                trace,
                color="#B04A8B" if emphasized else "0.55",
                linewidth=HIGHLIGHT_LINE_WIDTH if emphasized else OTHER_LINE_WIDTH,
                alpha=1.0 if emphasized else 0.60,
                label=label,
                zorder=10 if emphasized else rank,
            )
        axis.set_title(dataset, fontsize=FIGURE_FONT_SIZE)
        axis.set_ylim(*ROBUSTNESS_Y_LIMITS)
        axis.tick_params(labelsize=FIGURE_FONT_SIZE)
        handles, labels = axis.get_legend_handles_labels()
        order = np.argsort([int(label.split(",", 1)[0][1:]) for label in labels])
        axis.legend(
            [handles[int(index)] for index in order],
            [labels[int(index)] for index in order],
            fontsize=FIGURE_FONT_SIZE,
            frameon=False,
        )
    figure.suptitle("Top five medoid candidates", fontsize=FIGURE_FONT_SIZE)
    figure.supxlabel(x_label, fontsize=FIGURE_FONT_SIZE)
    figure.supylabel("DNA-calibrated current", fontsize=FIGURE_FONT_SIZE)
    figure.savefig(
        output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def _plot_js447_candidate_context(
    output_path: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    full_ranking_rows: Sequence[Mapping[str, Any]],
    trace_by_id: Mapping[tuple[str, int], FloatArr],
    *,
    x_label: str,
) -> None:
    """Show the five JS447 candidates in their retained-training context."""
    import matplotlib.pyplot as plt

    dataset = "JS447"
    candidates = [row for row in candidate_rows if row["construct"] == dataset]
    training = [row for row in full_ranking_rows if row["construct"] == dataset]
    if len(candidates) != TOP_K or not training:
        raise RuntimeError("JS447 context plot requires five candidates and full ranking")

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for row in training:
        event_id = int(row["original_read_id"])
        trace = trace_by_id[(dataset, event_id)]
        axis.plot(
            np.arange(trace.size),
            trace,
            color="0.82",
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )

    alternative_colours = ("#0072B2", "#009E73", "#E69F00", "#56B4E9")
    for row in reversed(candidates):
        rank = int(row["candidate_rank"])
        event_id = int(row["original_read_id"])
        trace = trace_by_id[(dataset, event_id)]
        emphasized = rank == 1
        colour = "#B04A8B" if emphasized else alternative_colours[rank - 2]
        axis.plot(
            np.arange(trace.size),
            trace,
            color=colour,
            linewidth=3.0 if emphasized else 1.7,
            alpha=1.0 if emphasized else 0.9,
            label=(
                f"Candidate #{rank}, event {event_id}, "
                f"ΔD̄={float(row['delta_mean_score']):.4g}"
            ),
            zorder=20 if emphasized else 10 - rank,
        )

    handles, labels = axis.get_legend_handles_labels()
    order = np.argsort([int(label.split(",", 1)[0].split("#")[1]) for label in labels])
    axis.legend(
        [handles[int(index)] for index in order],
        [labels[int(index)] for index in order],
        frameon=False,
        fontsize=FIGURE_FONT_SIZE,
    )
    axis.set_title(
        "2. JS447 medoid candidates within the retained training set",
        fontsize=FIGURE_FONT_SIZE,
    )
    axis.set_xlabel(x_label, fontsize=FIGURE_FONT_SIZE)
    axis.set_ylabel("DNA-calibrated current", fontsize=FIGURE_FONT_SIZE)
    axis.tick_params(labelsize=FIGURE_FONT_SIZE)
    figure.savefig(
        output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline_dir).resolve()
    distance_cache = Path(args.distance_cache).resolve()
    distance_rows_path = Path(args.distance_rows).resolve()
    trace_cache = Path(args.trace_cache).resolve()
    trace_rows_path = Path(args.trace_rows).resolve()
    out_dir = Path(args.out_dir).resolve()
    candidate_dir = out_dir / "candidate_scores"
    figure_dir = out_dir / "figures"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary, split_rows, baseline_outliers, production_medoids = _load_baseline(
        baseline_dir
    )
    baseline_method = str(summary.get("baseline_method", "native"))
    trace_representation = summary.get(
        "trace_representation",
        {
            "axis": "native peptide steps",
            "current": "per-event common-DNA-to-LUT affine calibrated",
            "step_finder": "CPIC on robust-normalized complete-event signal",
            "sensitivity": 1.0,
            "min_level_length": 2,
            "tail_threshold_source_relative_current": 0.6,
            "tail_control_steps": 5,
            "dna_window_steps": 30,
            "min_trace_steps": 5,
            "min_std": 0.001,
        },
    )
    if not isinstance(trace_representation, dict):
        raise RuntimeError("baseline trace representation is invalid")
    x_label = (
        "100-point normalized position (1–100)"
        if int(trace_representation.get("target_length", 0)) == 100
        else "Native peptide-step index"
    )
    distance_rows = [
        row for row in _read_csv(distance_rows_path) if row.get("dataset") in DATASETS
    ]
    trace_rows = [
        row for row in _read_csv(trace_rows_path) if row.get("dataset") in DATASETS
    ]
    _validate_cache_lineage(baseline_outliers, distance_rows, trace_rows)

    split_by_id = {
        (str(row["dataset"]), int(row["event_id"])): str(row["group"])
        for row in split_rows
    }
    retained_ids = {
        (str(row["dataset"]), int(row["event_id"]))
        for row in baseline_outliers
        if int(row["retained"])
    }
    if set(split_by_id) != retained_ids:
        raise RuntimeError("active split does not contain every and only retained event")

    trace_index = {
        (str(row["dataset"]), int(row["event_id"])): index
        for index, row in enumerate(trace_rows)
    }
    trace_by_id: dict[tuple[str, int], FloatArr] = {}
    trace_lengths: dict[tuple[str, int], int] = {}
    with np.load(trace_cache, allow_pickle=False) as traces:
        for identity, index in trace_index.items():
            mean = np.asarray(traces[f"mean_{index}"], dtype=float)
            std = np.asarray(traces[f"std_{index}"], dtype=float)
            if mean.ndim != 1 or std.shape != mean.shape or not np.all(np.isfinite(mean)):
                raise RuntimeError(f"invalid frozen trace for {identity}")
            if mean.size != int(trace_rows[index]["n_steps"]):
                raise RuntimeError(f"trace-length mapping differs for {identity}")
            trace_by_id[identity] = mean
            trace_lengths[identity] = int(mean.size)

    candidates: list[dict[str, Any]] = []
    full_rankings: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    cached_training: dict[str, FloatArr] = {}
    cached_ids: dict[str, Int64Arr] = {}
    cached_event_order: dict[str, Int64Arr] = {}
    rank_one_matches = 0

    with np.load(distance_cache, allow_pickle=False) as matrices:
        for dataset in DATASETS:
            source_rows = _dataset_rows(distance_rows, dataset)
            D = np.asarray(matrices[dataset], dtype=float)
            if D.shape != (len(source_rows), len(source_rows)):
                raise RuntimeError(f"{dataset}: distance matrix and companion rows differ")
            if (
                not np.all(np.isfinite(D))
                or not np.allclose(D, D.T, rtol=1e-10, atol=1e-12)
                or not np.allclose(np.diag(D), 0.0, rtol=0.0, atol=1e-12)
            ):
                raise RuntimeError(f"{dataset}: cached original distance matrix is invalid")

            training_positions = np.asarray(
                [
                    index
                    for index, row in enumerate(source_rows)
                    if split_by_id.get((dataset, int(row["event_id"]))) == "training"
                ],
                dtype=int,
            )
            heldout_ids = {
                event_id
                for (name, event_id), group in split_by_id.items()
                if name == dataset and group == "held_out"
            }
            if training_positions.size < TOP_K:
                raise RuntimeError(f"{dataset}: fewer than {TOP_K} retained training events")
            ranking = rank_medoid_candidates(D, training_positions, top_k=None)
            training_ids = [int(source_rows[int(index)]["event_id"]) for index in training_positions]
            local_index = {int(position): index for index, position in enumerate(training_positions)}
            best_mean = ranking[0].row_sum / (training_positions.size - 1)
            best_position = ranking[0].index
            best_event_id = int(source_rows[best_position]["event_id"])
            rank_one_match = best_event_id == production_medoids[dataset]
            rank_one_matches += int(rank_one_match)

            tie_order_ok = True
            for previous, current in zip(ranking, ranking[1:]):
                if current.row_sum == previous.row_sum and current.index < previous.index:
                    tie_order_ok = False
            mapping_ok = True
            no_heldout = True
            for rank, candidate in enumerate(ranking, start=1):
                source = source_rows[candidate.index]
                event_id = int(source["event_id"])
                local = local_index[candidate.index]
                identity = (dataset, event_id)
                mapping_ok &= training_ids[local] == event_id and identity in trace_by_id
                no_heldout &= event_id not in heldout_ids
                mean_row_sum = candidate.row_sum / (training_positions.size - 1)
                row = {
                    "construct": dataset,
                    "candidate_rank": rank,
                    "local_training_index": local,
                    "original_read_id": event_id,
                    "original_input_index": int(source["event_order"]),
                    "row_sum": candidate.row_sum,
                    "mean_row_sum": mean_row_sum,
                    "delta_mean_score": mean_row_sum - best_mean,
                    "trace_length": trace_lengths[identity],
                    "distance_to_rank1": float(D[best_position, candidate.index]),
                }
                full_rankings.append(row)
                if rank <= TOP_K:
                    candidates.append(row.copy())

            off_diagonal = D[~np.eye(D.shape[0], dtype=bool)]
            validation_rows.extend(
                [
                    {
                        "construct": dataset,
                        "check": "A_rank_1_reproduces_production_medoid",
                        "passed": int(rank_one_match),
                        "detail": f"rank1={best_event_id}; production={production_medoids[dataset]}",
                    },
                    {
                        "construct": dataset,
                        "check": "B_original_unshifted_matrix",
                        "passed": 1,
                        "detail": (
                            "ranking used cached original matrix directly; transform=none; "
                            f"min_off_diagonal={float(np.min(off_diagonal)):.17g}; "
                            f"linkage_only_shift_would_be={max(0.0, -float(np.min(off_diagonal))):.17g}"
                        ),
                    },
                    {
                        "construct": dataset,
                        "check": "C_candidate_identity_mapping",
                        "passed": int(mapping_ok),
                        "detail": "local index, event ID, event order, and trace-cache row agree",
                    },
                    {
                        "construct": dataset,
                        "check": "D_no_heldout_candidates",
                        "passed": int(no_heldout),
                        "detail": f"ranked_training={len(ranking)}; heldout_overlap=0",
                    },
                    {
                        "construct": dataset,
                        "check": "E_input_index_tie_handling",
                        "passed": int(tie_order_ok),
                        "detail": "equal row sums ordered by original distance-matrix index",
                    },
                ]
            )

            cached_training[f"{dataset}_distance"] = D[np.ix_(training_positions, training_positions)]
            cached_ids[f"{dataset}_event_id"] = np.asarray(training_ids, dtype=np.int64)
            cached_event_order[f"{dataset}_event_order"] = np.asarray(
                [int(source_rows[int(index)]["event_order"]) for index in training_positions],
                dtype=np.int64,
            )

    write_csv(candidate_dir / "medoid_candidates.csv", candidates)
    write_csv(candidate_dir / "full_medoid_rankings.csv", full_rankings)
    write_csv(candidate_dir / "stage1_validation_checks.csv", validation_rows)
    cached_values: dict[str, Any] = {
        **cached_training,
        **cached_ids,
        **cached_event_order,
    }
    np.savez_compressed(
        candidate_dir / "training_distance_matrices.npz", **cached_values
    )
    _plot_candidates(
        figure_dir / "1_candidate_traces_3x3.png",
        candidates,
        trace_by_id,
        x_label=x_label,
    )
    _plot_js447_candidate_context(
        figure_dir / "2_js447_candidate_context.png",
        candidates,
        full_rankings,
        trace_by_id,
        x_label=x_label,
    )

    all_checks_pass = all(bool(int(row["passed"])) for row in validation_rows)
    input_paths = {
        "baseline_run_summary": baseline_dir / "run_summary.json",
        "baseline_split": baseline_dir / "split_manifest.csv",
        "baseline_outlier_metrics": baseline_dir / "upstream_outlier_metrics.csv",
        "baseline_profile_diagnostics": baseline_dir / "profile_diagnostics.csv",
        "original_distance_cache": distance_cache,
        "distance_cache_companion_rows": distance_rows_path,
        "analysis_trace_cache": trace_cache,
        "trace_cache_companion_rows": trace_rows_path,
    }
    manifest = {
        "stage": 1,
        "stages_2_or_3_implemented": False,
        "datasets": list(DATASETS),
        "baseline": str(baseline_dir),
        "baseline_method": baseline_method,
        "baseline_files": summary.get(
            "baseline_files",
            {
                "profiles": "profiles_native_control.csv",
                "diagnostics": "profile_diagnostics.csv",
                "heldout_event_scores": "heldout_event_scores.csv",
                "classification_summary": "classification_summary.csv",
                "classification_per_class": "classification_per_class.csv",
                "classification_confusion_matrix": "classification_confusion_matrix.csv",
            },
        ),
        "repository": _repository_version(),
        "trace_representation": trace_representation,
        "frozen_training_counts": {
            dataset: sum(
                row["construct"] == dataset for row in full_rankings
            )
            for dataset in DATASETS
        },
        "frozen_heldout_counts": {
            dataset: sum(
                name == dataset and group == "held_out"
                for (name, _), group in split_by_id.items()
            )
            for dataset in DATASETS
        },
        "gaussian_dtw": {
            "matrix": "original unshifted path-normalized Gaussian-DTW",
            "normalize": False,
            "max_run": 3,
            "ranking_transform": "none",
        },
        "split": summary["split"],
        "outlier_filter": summary["outlier_filter"],
        "stage2_dba_settings_frozen_but_not_executed": {
            "normalize": False,
            "max_run": 3,
            "max_iter": 50,
            "tol": 0.0001,
            "min_std": 0.001,
            "min_depth": "max(2, ceil(0.5 * n_training))",
            "endpoint_handling": "closed",
        },
        "validation": {
            "all_stage1_checks_pass": all_checks_pass,
            "rank_1_matches_production": f"{rank_one_matches}/{len(DATASETS)}",
            "checks_file": "candidate_scores/stage1_validation_checks.csv",
        },
        "cache_provenance": (
            "distance and trace NPZ axes are validated against companion CSV event IDs, "
            "event order, retained flags, and exact upstream outlier lineage fields"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "outputs": {
            "top_five": "candidate_scores/medoid_candidates.csv",
            "complete_ranking": "candidate_scores/full_medoid_rankings.csv",
            "training_matrices": "candidate_scores/training_distance_matrices.npz",
            "candidate_figure": "figures/1_candidate_traces_3x3.png",
            "js447_candidate_context": "figures/2_js447_candidate_context.png",
        },
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (out_dir / "stage1_run_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    (out_dir / "run_manifest.json").write_text(manifest_text, encoding="utf-8")
    if not all_checks_pass or rank_one_matches != len(DATASETS):
        raise RuntimeError(
            "Stage 1 validation gate failed; inspect candidate_scores/stage1_validation_checks.csv"
        )
    print(f"Stage 1 passed: production medoid reproduced {rank_one_matches}/9")
    print(f"Wrote Stage 1 outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--distance-cache", default=DEFAULT_DISTANCE_CACHE)
    parser.add_argument("--distance-rows", default=DEFAULT_DISTANCE_ROWS)
    parser.add_argument("--trace-cache", default=DEFAULT_TRACE_CACHE)
    parser.add_argument("--trace-rows", default=DEFAULT_TRACE_ROWS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
