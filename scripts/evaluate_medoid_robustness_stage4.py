"""Build the read-only Stage 4 medoid-robustness summary and heatmaps.

The runner consumes only accepted Stage 1--3 artifacts.  It validates their hashes and
joins their existing candidate-, consensus-, and classification-level measurements by
``(construct, candidate_rank)``.  It does not rerun filtering, splitting,
preprocessing, candidate selection, DBA, or classification, and it does not add
thresholds, inferential analyses, global substitutions, or optional plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from scripts.evaluate_medoid_robustness import (
    DATASETS,
    FIGURE_DPI,
    FIGURE_FONT_SIZE,
    ROOT,
    _read_csv,  # pyright: ignore[reportPrivateUsage]
    _repository_version,  # pyright: ignore[reportPrivateUsage]
    _sha256,  # pyright: ignore[reportPrivateUsage]
)
from scripts.evaluate_medoid_robustness_stage3 import (
    _load_json,  # pyright: ignore[reportPrivateUsage]
    _require_historical_provenance,  # pyright: ignore[reportPrivateUsage]
)
from scripts.evaluate_psk_dba_consensus import write_csv

DEFAULT_OUT_DIR = ROOT / "tmp" / "medoid_robustness"
DEFAULT_CONSTRUCT_ICON_DIR = ROOT.parent / "Inkscape" / "psk_no_border"
RANKS = (2, 3, 4, 5)
N_PERTURBATIONS = len(DATASETS) * len(RANKS)
NEGATIVE_BALANCED_ACCURACY_COLOUR = "#D9822B"
NEUTRAL_BALANCED_ACCURACY_COLOUR = "#FAFAF7"
POSITIVE_BALANCED_ACCURACY_COLOUR = "#1261A0"
FloatArr = npt.NDArray[np.float64]
JoinKey = tuple[str, int]
CONSTRUCT_ICON_ZOOM = 0.043


def _verify_absolute_record(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise RuntimeError(f"accepted artifact hash differs for {label}: {path}")
    return path


def _verify_relative_record(
    out_dir: Path, record: Mapping[str, Any], label: str
) -> Path:
    path = out_dir / str(record.get("path", ""))
    expected = str(record.get("sha256", ""))
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise RuntimeError(f"accepted artifact hash differs for {label}: {path}")
    return path


def _accepted_inputs(
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    manifest_paths = {
        "stage1": out_dir / "stage1_run_manifest.json",
        "stage2": out_dir / "stage2_run_manifest.json",
        "stage2_followup": out_dir / "stage2_followup_run_manifest.json",
        "stage3": out_dir / "stage3_run_manifest.json",
    }
    manifests = {label: _load_json(path) for label, path in manifest_paths.items()}
    stage1 = manifests["stage1"]
    stage2 = manifests["stage2"]
    followup = manifests["stage2_followup"]
    stage3 = manifests["stage3"]
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
    if not bool(stage2.get("rank1_reproduction_gate", {}).get("passed")):
        raise RuntimeError("Stage 2 rank-1 gate is not accepted")
    if not bool(stage2.get("convergence", {}).get("all_45_converged")):
        raise RuntimeError("Stage 2 did not accept all 45 DBA fits")
    if followup.get("metric", {}).get("stored_stage2_consensus_values_reproduced") != "36/36":
        raise RuntimeError("Stage 2 descriptive follow-up is not accepted")
    if not bool(stage3.get("baseline_gate", {}).get("passed")):
        raise RuntimeError("Stage 3 baseline gate is not accepted")
    if int(stage3.get("classification", {}).get("n_perturbations", -1)) != N_PERTURBATIONS:
        raise RuntimeError("Stage 3 does not contain the accepted 36 perturbations")
    if stage3.get("stage_4_implemented") is not False:
        raise RuntimeError("accepted Stage 3 manifest does not preserve its stage boundary")

    followup_provenance = followup.get("provenance", {})
    stage3_provenance = stage3.get("provenance", {})
    accepted_manifests = stage3_provenance.get("accepted_manifests", {})
    for label in ("stage1", "stage2", "stage2_followup"):
        record = accepted_manifests.get(label)
        if not isinstance(record, dict):
            raise RuntimeError(f"Stage 3 lacks accepted {label} manifest provenance")
        if record.get("sha256") != _sha256(manifest_paths[label]):
            raise RuntimeError(f"Stage 3 {label} manifest hash differs")
    if followup_provenance.get("stage1_manifest", {}).get("sha256") != _sha256(
        manifest_paths["stage1"]
    ):
        raise RuntimeError("Stage 2 follow-up Stage 1 manifest hash differs")
    if followup_provenance.get("stage2_manifest", {}).get("sha256") != _sha256(
        manifest_paths["stage2"]
    ):
        raise RuntimeError("Stage 2 follow-up Stage 2 manifest hash differs")

    combined = _load_json(out_dir / "run_manifest.json")
    _require_historical_provenance(combined, "combined Stage 1--3")
    if combined.get("stage3", {}).get("manifest_sha256") != _sha256(
        manifest_paths["stage3"]
    ):
        raise RuntimeError("combined manifest is not linked to accepted Stage 3")

    candidate_record = followup_provenance.get("candidate_table")
    comparison_record = followup_provenance.get("consensus_comparisons")
    followup_detail_record = followup.get("outputs", {}).get("detail")
    stage3_metrics_record = stage3.get("outputs", {}).get("run_metrics")
    for label, record in (
        ("Stage 1 candidate table", candidate_record),
        ("Stage 2 consensus comparisons", comparison_record),
    ):
        if not isinstance(record, dict):
            raise RuntimeError(f"missing accepted input record: {label}")
        _verify_absolute_record(record, label)
    for label, record in (
        ("Stage 2 connected detail", followup_detail_record),
        ("Stage 3 run metrics", stage3_metrics_record),
    ):
        if not isinstance(record, dict):
            raise RuntimeError(f"missing accepted input record: {label}")
        _verify_relative_record(out_dir, record, label)

    input_paths = {
        "candidate_table": Path(str(candidate_record["path"])),
        "consensus_comparisons": Path(str(comparison_record["path"])),
        "candidate_vs_consensus": out_dir / str(followup_detail_record["path"]),
        "classification_run_metrics": out_dir / str(stage3_metrics_record["path"]),
    }
    return followup, stage3, combined, {**manifest_paths, **input_paths}


def _indexed_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    construct_field: str,
    rank_field: str,
    label: str,
) -> dict[JoinKey, Mapping[str, str]]:
    result: dict[JoinKey, Mapping[str, str]] = {}
    for row in rows:
        construct = row.get(construct_field, "")
        rank = int(row.get(rank_field, "-1"))
        if construct not in DATASETS or rank not in RANKS:
            continue
        key = (construct, rank)
        if key in result:
            raise RuntimeError(f"{label} contains duplicate join key {key}")
        result[key] = row
    expected = {(dataset, rank) for dataset in DATASETS for rank in RANKS}
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise RuntimeError(f"{label} join keys differ; missing={missing}, extra={extra}")
    return result


def _require_equal_float(a: str, b: str, label: str) -> None:
    if float(a) != float(b):
        raise RuntimeError(f"accepted repeated value differs for {label}: {a} != {b}")


def _joined_summary_rows(
    candidate_rows: Sequence[Mapping[str, str]],
    comparison_rows: Sequence[Mapping[str, str]],
    connected_rows: Sequence[Mapping[str, str]],
    metric_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Join the accepted evidence chain without deriving new robustness metrics."""

    baseline_rows = [row for row in metric_rows if row.get("run_id") == "baseline"]
    if len(baseline_rows) != 1:
        raise RuntimeError("Stage 3 metrics must contain exactly one baseline row")
    baseline_balanced_accuracy = float(baseline_rows[0]["balanced_accuracy"])
    candidates = _indexed_rows(
        candidate_rows,
        construct_field="construct",
        rank_field="candidate_rank",
        label="Stage 1 candidates",
    )
    comparisons = _indexed_rows(
        comparison_rows,
        construct_field="construct",
        rank_field="candidate_rank",
        label="Stage 2 consensus comparisons",
    )
    connected = _indexed_rows(
        connected_rows,
        construct_field="construct",
        rank_field="candidate_rank",
        label="Stage 2 connected detail",
    )
    metrics = _indexed_rows(
        metric_rows,
        construct_field="perturbed_construct",
        rank_field="candidate_rank",
        label="Stage 3 run metrics",
    )

    joined: list[dict[str, Any]] = []
    for construct in DATASETS:
        for rank in RANKS:
            key = (construct, rank)
            candidate = candidates[key]
            comparison = comparisons[key]
            detail = connected[key]
            metric = metrics[key]
            expected_run_id = f"{construct}_rank{rank}"
            if metric["run_id"] != expected_run_id:
                raise RuntimeError(f"Stage 3 run identity differs for {key}")
            candidate_event_id = int(candidate["original_read_id"])
            for label, actual in (
                ("Stage 2 comparison", comparison["candidate_event_id"]),
                ("Stage 2 connected detail", detail["alternative_medoid_event_id"]),
            ):
                if int(actual) != candidate_event_id:
                    raise RuntimeError(f"candidate event identity differs for {key}: {label}")
            _require_equal_float(
                candidate["delta_mean_score"],
                comparison["delta_candidate_mean_score"],
                f"{key} candidate delta/comparison",
            )
            _require_equal_float(
                candidate["delta_mean_score"],
                detail["delta_mean_score"],
                f"{key} candidate delta/connected detail",
            )
            _require_equal_float(
                comparison["aligned_rmse"],
                detail["consensus_aligned_rmse"],
                f"{key} consensus aligned RMSE",
            )
            joined.append(
                {
                    "construct": construct,
                    "candidate_rank": rank,
                    "candidate_event_id": candidate_event_id,
                    "delta_candidate_mean_score": float(candidate["delta_mean_score"]),
                    "candidate_distance_to_rank1": float(candidate["distance_to_rank1"]),
                    "candidate_aligned_rmse": float(detail["candidate_aligned_rmse"]),
                    "consensus_gaussian_dtw_distance": float(
                        comparison["gaussian_dtw_distance"]
                    ),
                    "consensus_aligned_rmse": float(detail["consensus_aligned_rmse"]),
                    "baseline_balanced_accuracy": baseline_balanced_accuracy,
                    "perturbed_balanced_accuracy": float(metric["balanced_accuracy"]),
                    "delta_balanced_accuracy": float(metric["delta_balanced_accuracy"]),
                    "n_prediction_flips": int(metric["n_prediction_flips"]),
                    "prediction_flip_rate": float(metric["prediction_flip_rate"]),
                }
            )
    if len(joined) != N_PERTURBATIONS:
        raise RuntimeError("Stage 4 join did not produce exactly 36 rows")
    return joined


def _heatmap_matrix(rows: Sequence[Mapping[str, Any]], field: str) -> FloatArr:
    values = {
        (str(row["construct"]), int(row["candidate_rank"])): float(row[field])
        for row in rows
    }
    expected = {(dataset, rank) for dataset in DATASETS for rank in RANKS}
    if set(values) != expected:
        raise RuntimeError(f"{field} heatmap keys do not form a complete 9 x 4 grid")
    matrix = np.asarray(
        [[values[(dataset, rank)] for rank in RANKS] for dataset in DATASETS],
        dtype=float,
    )
    if matrix.shape != (9, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"{field} heatmap contains invalid values")
    return matrix


def _construct_icon_paths(icon_dir: Path) -> dict[str, Path]:
    paths = {dataset: icon_dir / f"{dataset}.png" for dataset in DATASETS}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing construct icon(s): " + ", ".join(str(path) for path in missing)
        )
    return paths


def _plot_heatmap(
    path: Path,
    matrix: FloatArr,
    *,
    title: str,
    colourbar_label: str,
    divergent: bool,
    construct_icon_paths: Mapping[str, Path] | None = None,
) -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.ticker import PercentFormatter

    if divergent:
        limit = float(np.max(np.abs(matrix)))
        norm: mcolors.Normalize = mcolors.TwoSlopeNorm(
            vmin=-limit, vcenter=0.0, vmax=limit
        )
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "medoid_robustness_diverging",
            [
                NEGATIVE_BALANCED_ACCURACY_COLOUR,
                NEUTRAL_BALANCED_ACCURACY_COLOUR,
                POSITIVE_BALANCED_ACCURACY_COLOUR,
            ],
        )
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=float(np.max(matrix)))
        cmap = "viridis"
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(range(len(RANKS)), labels=[f"Rank {rank}" for rank in RANKS])
    axis.set_yticks(range(len(DATASETS)), labels=DATASETS)
    axis.set_xlabel("Alternative medoid candidate", fontsize=FIGURE_FONT_SIZE)
    axis.set_ylabel("Perturbed variant", fontsize=FIGURE_FONT_SIZE)
    axis.set_title(title, fontsize=FIGURE_FONT_SIZE)
    axis.tick_params(labelsize=FIGURE_FONT_SIZE)
    if construct_icon_paths is not None:
        axis.yaxis.set_label_coords(-0.255, 0.5)
        for row, dataset in enumerate(DATASETS):
            icon = OffsetImage(
                plt.imread(construct_icon_paths[dataset]), zoom=CONSTRUCT_ICON_ZOOM
            )
            axis.add_artist(
                AnnotationBbox(
                    icon,
                    (-0.155, row),
                    xycoords=axis.get_yaxis_transform(),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(0.5, 0.5),
                    annotation_clip=False,
                )
            )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                column,
                row,
                f"{100.0 * value:+.2f}%" if divergent else f"{100.0 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=FIGURE_FONT_SIZE,
                color="white" if luminance < 0.5 else "#222222",
            )
    colourbar = figure.colorbar(image, ax=axis)
    colourbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    colourbar.set_label(colourbar_label, fontsize=FIGURE_FONT_SIZE)
    colourbar.ax.tick_params(labelsize=FIGURE_FONT_SIZE)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _interpretation_markdown() -> str:
    """Return the plan-defined interpretation framework without new thresholds."""

    return """# Connected medoid-robustness interpretation

## Scope

Stage 4 connects the 36 accepted one-at-a-time perturbations through the evidence chain:

> candidate difference → consensus difference → classification difference

The exact row-wise evidence is in `perturbation_summary.csv`. Candidate evidence uses
the accepted Stage 1 score delta and candidate aligned RMSE; consensus evidence uses
the accepted Stage 2 Gaussian-DTW distance and aligned RMSE; classification evidence
uses the accepted Stage 3 delta balanced accuracy and prediction-flip rate.

## Outcome framework

- **Outcome A:** Candidate changes little, consensus changes little, and classification
  changes little. Exact medoid choice is practically unimportant among these candidates,
  although the candidate traces were already very similar.
- **Outcome B:** Candidate changes appreciably, consensus remains similar, and
  classification remains stable. This supports DBA robustness to near-optimal
  initialization.
- **Outcome C:** Candidate and consensus change, while classification remains stable.
  Initialization influences DBA consensus generation, but downstream classification is
  robust.
- **Outcome D:** Candidate, consensus, and classification all change. Medoid sensitivity
  propagates through DBA into classification.

The plan defines no numerical boundary for “little”, “appreciably”, or “stable”. Stage 4
therefore does not assign perturbations to Outcomes A–D and does not introduce a new
threshold. The joined table and Figures 5–6 provide the complete evidence needed for a
subsequent domain-level interpretation under this framework.

## Explicit exclusions

No top-10/20 ranking plot, simultaneous global substitution, bootstrap confidence
interval, new robustness metric, new threshold, or inferential analysis was generated.
"""


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    summary_dir = out_dir / "summary"
    figure_dir = out_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    followup, stage3, combined, input_paths = _accepted_inputs(out_dir)
    candidate_rows = _read_csv(input_paths["candidate_table"])
    comparison_rows = _read_csv(input_paths["consensus_comparisons"])
    connected_rows = _read_csv(input_paths["candidate_vs_consensus"])
    metric_rows = _read_csv(input_paths["classification_run_metrics"])

    print("1. Joining accepted Stage 1--3 evidence by construct and rank...", flush=True)
    joined = _joined_summary_rows(
        candidate_rows, comparison_rows, connected_rows, metric_rows
    )
    summary_path = summary_dir / "perturbation_summary.csv"
    write_csv(summary_path, joined)

    validation_rows = [
        {
            "check": "accepted_stage1_stage2_stage3_hashes_match",
            "passed": 1,
            "detail": "all selected input and manifest SHA-256 records verified",
        },
        {
            "check": "join_keys_exact_and_complete",
            "passed": int(len(joined) == N_PERTURBATIONS),
            "detail": "36/36 construct-rank keys",
        },
        {
            "check": "candidate_identity_and_repeated_values_exact",
            "passed": 1,
            "detail": "event IDs, candidate deltas, and consensus RMSE cross-checked",
        },
        {
            "check": "stage4_reads_accepted_outputs_only",
            "passed": 1,
            "detail": "no filtering, splitting, preprocessing, candidate, DBA, or classification rerun",
        },
        {
            "check": "optional_and_inferential_analyses_excluded",
            "passed": 1,
            "detail": "no optional plot, global substitution, bootstrap, threshold, or new metric",
        },
    ]
    validation_path = summary_dir / "stage4_validation_checks.csv"
    write_csv(validation_path, validation_rows)

    print("2. Rendering the two mandatory 9 x 4 classification heatmaps...", flush=True)
    balanced_matrix = _heatmap_matrix(joined, "delta_balanced_accuracy")
    flip_matrix = _heatmap_matrix(joined, "prediction_flip_rate")
    balanced_path = figure_dir / "5_delta_balanced_accuracy_heatmap.png"
    flip_path = figure_dir / "6_prediction_flip_heatmap.png"
    balanced_icon_path = (
        figure_dir / "5_delta_balanced_accuracy_heatmap_with_construct_icons.png"
    )
    flip_icon_path = figure_dir / "6_prediction_flip_heatmap_with_construct_icons.png"
    _plot_heatmap(
        balanced_path,
        balanced_matrix,
        title="5. Change in balanced accuracy after one medoid substitution",
        colourbar_label="Δ balanced accuracy",
        divergent=True,
    )
    _plot_heatmap(
        flip_path,
        flip_matrix,
        title="6. Prediction-flip rate after one medoid substitution",
        colourbar_label="Prediction-flip rate",
        divergent=False,
    )
    construct_icon_paths = _construct_icon_paths(DEFAULT_CONSTRUCT_ICON_DIR)
    _plot_heatmap(
        balanced_icon_path,
        balanced_matrix,
        title="5. Change in balanced accuracy after one medoid substitution",
        colourbar_label="Δ balanced accuracy",
        divergent=True,
        construct_icon_paths=construct_icon_paths,
    )
    _plot_heatmap(
        flip_icon_path,
        flip_matrix,
        title="6. Prediction-flip rate after one medoid substitution",
        colourbar_label="Prediction-flip rate",
        divergent=False,
        construct_icon_paths=construct_icon_paths,
    )

    interpretation_path = summary_dir / "connected_interpretation.md"
    interpretation_path.write_text(_interpretation_markdown(), encoding="utf-8")
    output_paths = {
        "perturbation_summary": summary_path,
        "validation_checks": validation_path,
        "connected_interpretation": interpretation_path,
        "delta_balanced_accuracy_heatmap": balanced_path,
        "prediction_flip_heatmap": flip_path,
        "delta_balanced_accuracy_heatmap_with_construct_icons": balanced_icon_path,
        "prediction_flip_heatmap_with_construct_icons": flip_icon_path,
    }
    output_records = {
        label: {
            "path": str(path.relative_to(out_dir)),
            "sha256": _sha256(path),
        }
        for label, path in output_paths.items()
    }
    stage4_manifest = {
        "stage": 4,
        "authorized_scope": "medoid robustness plan sections 4.1--4.3 only",
        "mandatory_stage4_complete": True,
        "optional_analyses_implemented": False,
        "implementation_repository": _repository_version(),
        "stage1_stage2_repository_provenance_preserved": combined["repository"],
        "accepted_stage3_implementation_repository": stage3[
            "implementation_repository"
        ],
        "datasets": list(DATASETS),
        "candidate_ranks": list(RANKS),
        "n_joined_perturbations": len(joined),
        "summary_contract": {
            "join_key": ["construct", "candidate_rank"],
            "chain": (
                "candidate difference -> consensus difference -> classification difference"
            ),
            "new_metrics_introduced": False,
            "new_thresholds_introduced": False,
            "categorical_outcome_assignments_made": False,
            "categorical_assignment_reason": (
                "the plan defines no numerical boundaries for little, appreciable, or stable"
            ),
        },
        "heatmaps": {
            "shape": [9, 4],
            "rows": list(DATASETS),
            "columns": [f"rank{rank}" for rank in RANKS],
            "delta_balanced_accuracy": True,
            "prediction_flip_rate": True,
            "construct_icon_variants": True,
            "dpi": FIGURE_DPI,
            "canvas_inches": [10, 6],
            "font_size": FIGURE_FONT_SIZE,
        },
        "scope_guards": {
            "filtering_rerun": False,
            "splitting_rerun": False,
            "preprocessing_rerun": False,
            "candidate_selection_rerun": False,
            "dba_rerun": False,
            "classification_rerun": False,
            "top_10_or_20_plot_generated": False,
            "simultaneous_global_substitution_run": False,
            "bootstrap_confidence_intervals_generated": False,
            "inferential_analysis_added": False,
        },
        "provenance": {
            "accepted_manifests": {
                label: {
                    "path": str(input_paths[label]),
                    "sha256": _sha256(input_paths[label]),
                }
                for label in ("stage1", "stage2", "stage2_followup", "stage3")
            },
            "accepted_selected_inputs": {
                label: {"path": str(path), "sha256": _sha256(path)}
                for label, path in input_paths.items()
                if label not in {"stage1", "stage2", "stage2_followup", "stage3"}
            },
            "stage2_followup_metric": followup["metric"],
            "construct_icon_assets": {
                dataset: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for dataset, path in construct_icon_paths.items()
            },
        },
        "validation": {
            "all_checks_pass": True,
            "checks_file": "summary/stage4_validation_checks.csv",
        },
        "outputs": output_records,
    }
    stage4_path = out_dir / "stage4_run_manifest.json"
    stage4_path.write_text(
        json.dumps(stage4_manifest, indent=2) + "\n", encoding="utf-8"
    )

    combined["stage"] = 4
    combined["stage4"] = {
        "manifest": "stage4_run_manifest.json",
        "manifest_sha256": _sha256(stage4_path),
        "mandatory_sections": ["4.1", "4.2", "4.3"],
        "n_joined_perturbations": len(joined),
        "optional_analyses_implemented": False,
    }
    existing_outputs = combined.get("outputs")
    if not isinstance(existing_outputs, dict):
        raise RuntimeError("combined Stage 1--3 output manifest is invalid")
    combined["outputs"] = {
        **existing_outputs,
        **{f"stage4_{label}": record for label, record in output_records.items()},
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Stage 4 complete: joined {len(joined)} perturbations and wrote Figures 5--6",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
