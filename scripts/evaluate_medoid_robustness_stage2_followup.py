"""Compare medoid-trace and converged-consensus differences after Stage 2.

This descriptive follow-up reads the accepted Stage 1 traces/candidates and Stage 2
consensus caches.  It does not select candidates, filter events, cluster, split,
preprocess, fit DBA, or classify reads.  Candidate and consensus differences both use
the exact closed-DTW aligned-RMSE helper defined by the Stage 2 runner.
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
from scripts.evaluate_medoid_robustness_stage2 import (
    _aligned_rmse,  # pyright: ignore[reportPrivateUsage]
    _load_candidates_and_matrices,  # pyright: ignore[reportPrivateUsage]
    _load_training_traces,  # pyright: ignore[reportPrivateUsage]
    _stage1_manifest,  # pyright: ignore[reportPrivateUsage]
    _verify_stage1_hashes,  # pyright: ignore[reportPrivateUsage]
)
from scripts.evaluate_psk_dba_consensus import write_csv

DEFAULT_OUT_DIR = ROOT / "tmp" / "medoid_robustness"
FloatArr = npt.NDArray[np.float64]
CONSTRUCT_COLOURS = {
    "JS445": "#8D8D8D",
    "JS446": "#B9BD18",
    "JS447": "#4CAF59",
    "JS448": "#FF7F0E",
    "JS449": "#0878B9",
    "JS450": "#F0AD16",
    "JS451": "#DC69C5",
    "JS452": "#8E5BBB",
    "JS453": "#D62728",
}
RMSE_AXIS_LOWER_BOUND = 0.005


def _load_stage2_manifest(out_dir: Path) -> tuple[dict[str, Any], Path]:
    path = out_dir / "stage2_run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing accepted Stage 2 manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not bool(manifest.get("rank1_reproduction_gate", {}).get("passed")):
        raise RuntimeError("Stage 2 rank-1 reproduction gate is not accepted")
    if not bool(manifest.get("convergence", {}).get("all_45_converged")):
        raise RuntimeError("Stage 2 follow-up requires all 45 cached fits to converge")
    if manifest.get("stage_3_implemented") is not False:
        raise RuntimeError("Stage 2 manifest does not preserve the Stage 3 boundary")
    return manifest, path


def _comparison_rows_by_identity(
    path: Path,
) -> dict[tuple[str, int], dict[str, str]]:
    rows = _read_csv(path)
    lookup = {
        (row["construct"], int(row["candidate_rank"])): row for row in rows
    }
    expected = {(dataset, rank) for dataset in DATASETS for rank in range(2, 6)}
    if set(lookup) != expected:
        raise RuntimeError("Stage 2 consensus comparison identities are incomplete")
    return lookup


def _cached_consensus_mean(
    path: Path,
    *,
    dataset: str,
    rank: int,
    event_id: int,
) -> FloatArr:
    with np.load(path, allow_pickle=False) as cached:
        if cached["construct"].item() != dataset:
            raise RuntimeError(f"cached construct identity differs: {path}")
        if int(cached["candidate_rank"].item()) != rank:
            raise RuntimeError(f"cached candidate rank differs: {path}")
        if int(cached["candidate_event_id"].item()) != event_id:
            raise RuntimeError(f"cached candidate event differs: {path}")
        if not bool(cached["converged"].item()):
            raise RuntimeError(f"cached consensus did not converge: {path}")
        return np.asarray(cached["mean"], dtype=float)


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        selected = [row for row in rows if row["construct"] == dataset]
        if len(selected) != 4:
            raise RuntimeError(f"{dataset}: expected four Stage 2 perturbations")
        candidate_rmse = np.asarray(
            [float(row["candidate_aligned_rmse"]) for row in selected]
        )
        consensus_rmse = np.asarray(
            [float(row["consensus_aligned_rmse"]) for row in selected]
        )
        alternative_lengths = np.asarray(
            [int(row["candidate_consensus_length"]) for row in selected]
        )
        baseline_lengths = {int(row["baseline_consensus_length"]) for row in selected}
        if len(baseline_lengths) != 1:
            raise RuntimeError(f"{dataset}: baseline consensus length changed")
        summaries.append(
            {
                "construct": dataset,
                "mean_candidate_aligned_rmse": float(np.mean(candidate_rmse)),
                "max_candidate_aligned_rmse": float(np.max(candidate_rmse)),
                "mean_consensus_aligned_rmse": float(np.mean(consensus_rmse)),
                "max_consensus_aligned_rmse": float(np.max(consensus_rmse)),
                "mean_consensus_minus_candidate_rmse": float(
                    np.mean(consensus_rmse - candidate_rmse)
                ),
                "baseline_consensus_length": baseline_lengths.pop(),
                "min_alternative_consensus_length": int(np.min(alternative_lengths)),
                "max_alternative_consensus_length": int(np.max(alternative_lengths)),
            }
        )
    return summaries


def _plot_candidate_vs_consensus_rmse(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    all_values: list[float] = []
    for dataset in DATASETS:
        colour = CONSTRUCT_COLOURS[dataset]
        selected = sorted(
            [row for row in rows if row["construct"] == dataset],
            key=lambda row: int(row["candidate_rank"]),
        )
        x = np.asarray([float(row["candidate_aligned_rmse"]) for row in selected])
        y = np.asarray([float(row["consensus_aligned_rmse"]) for row in selected])
        all_values.extend(x.tolist())
        all_values.extend(y.tolist())
        axis.scatter(
            x,
            y,
            s=60,
            color=colour,
            label=dataset,
            zorder=3,
        )
        for row, x_value, y_value in zip(selected, x, y):
            axis.annotate(
                str(int(row["candidate_rank"])),
                (x_value, y_value),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=FIGURE_FONT_SIZE,
                color=colour,
            )
    upper = 1.05 * max(all_values)
    axis.plot(
        [RMSE_AXIS_LOWER_BOUND, upper],
        [RMSE_AXIS_LOWER_BOUND, upper],
        color="0.45",
        linestyle="--",
        linewidth=1.2,
        label="Equal RMSE",
        zorder=1,
    )
    axis.set_xlim(RMSE_AXIS_LOWER_BOUND, upper)
    axis.set_ylim(RMSE_AXIS_LOWER_BOUND, upper)
    axis.set_title(
        "4. Candidate-trace versus DBA-consensus aligned RMSE",
        fontsize=FIGURE_FONT_SIZE,
    )
    axis.set_xlabel("Candidate aligned RMSE", fontsize=FIGURE_FONT_SIZE)
    axis.set_ylabel("Consensus aligned RMSE", fontsize=FIGURE_FONT_SIZE)
    axis.tick_params(labelsize=FIGURE_FONT_SIZE)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=FIGURE_FONT_SIZE,
    )
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    stage1, stage1_manifest_path = _stage1_manifest(out_dir)
    _verify_stage1_hashes(stage1)
    _, stage2_manifest_path = _load_stage2_manifest(out_dir)
    candidates, _, event_ids, _ = _load_candidates_and_matrices(out_dir)
    training = _load_training_traces(stage1, event_ids)
    comparison_path = out_dir / "consensus_comparisons.csv"
    stored_comparisons = _comparison_rows_by_identity(comparison_path)

    rows: list[dict[str, Any]] = []
    verified_consensus_rmse = 0
    for dataset in DATASETS:
        baseline_candidate = candidates[dataset][0]
        baseline_local = int(baseline_candidate["local_training_index"])
        baseline_trace = training[dataset][baseline_local][0]
        baseline_consensus = _cached_consensus_mean(
            out_dir / "consensuses" / dataset / "rank1.npz",
            dataset=dataset,
            rank=1,
            event_id=int(baseline_candidate["original_read_id"]),
        )
        for candidate in candidates[dataset][1:]:
            rank = int(candidate["candidate_rank"])
            alternative_local = int(candidate["local_training_index"])
            alternative_trace = training[dataset][alternative_local][0]
            candidate_rmse, candidate_path_length = _aligned_rmse(
                baseline_trace, alternative_trace
            )
            alternative_consensus = _cached_consensus_mean(
                out_dir / "consensuses" / dataset / f"rank{rank}.npz",
                dataset=dataset,
                rank=rank,
                event_id=int(candidate["original_read_id"]),
            )
            consensus_rmse, consensus_path_length = _aligned_rmse(
                baseline_consensus, alternative_consensus
            )
            stored = stored_comparisons[(dataset, rank)]
            if (
                consensus_rmse != float(stored["aligned_rmse"])
                or consensus_path_length != int(stored["aligned_path_length"])
            ):
                raise RuntimeError(
                    f"{dataset} rank {rank}: Stage 2 aligned-RMSE definition changed"
                )
            if (
                int(stored["rank1_event_id"])
                != int(baseline_candidate["original_read_id"])
                or int(stored["candidate_event_id"])
                != int(candidate["original_read_id"])
            ):
                raise RuntimeError(
                    f"{dataset} rank {rank}: Stage 2 candidate identity changed"
                )
            verified_consensus_rmse += 1
            rows.append(
                {
                    "construct": dataset,
                    "candidate_rank": rank,
                    "baseline_medoid_event_id": int(
                        baseline_candidate["original_read_id"]
                    ),
                    "alternative_medoid_event_id": int(candidate["original_read_id"]),
                    "baseline_local_training_index": baseline_local,
                    "alternative_local_training_index": alternative_local,
                    "delta_mean_score": float(candidate["delta_mean_score"]),
                    "candidate_aligned_rmse": candidate_rmse,
                    "consensus_aligned_rmse": consensus_rmse,
                    "consensus_minus_candidate_rmse": consensus_rmse - candidate_rmse,
                    "consensus_to_candidate_rmse_ratio": consensus_rmse / candidate_rmse,
                    "candidate_alignment_path_length": candidate_path_length,
                    "consensus_alignment_path_length": consensus_path_length,
                    "baseline_candidate_trace_length": int(baseline_trace.size),
                    "alternative_candidate_trace_length": int(alternative_trace.size),
                    "baseline_consensus_length": int(stored["rank1_consensus_length"]),
                    "candidate_consensus_length": int(
                        stored["candidate_consensus_length"]
                    ),
                }
            )

    if verified_consensus_rmse != 36:
        raise RuntimeError("did not verify all 36 existing consensus RMSE values")
    summary_rows = _summary_rows(rows)
    detail_path = out_dir / "candidate_vs_consensus_aligned_rmse.csv"
    summary_path = out_dir / "candidate_vs_consensus_rmse_summary.csv"
    figure_path = figure_dir / "4_candidate_vs_consensus_aligned_rmse.png"
    write_csv(detail_path, rows)
    write_csv(summary_path, summary_rows)
    _plot_candidate_vs_consensus_rmse(figure_path, rows)

    manifest = {
        "stage": "2_descriptive_followup",
        "new_experimental_stage": False,
        "stage_3_implemented": False,
        "repository": _repository_version(),
        "scope": (
            "descriptive candidate-trace versus converged-consensus aligned RMSE; "
            "no decision threshold or robustness classification"
        ),
        "metric": {
            "name": "aligned_rmse",
            "implementation": (
                "closed DTW on cost_l2(baseline_mean, alternative_mean), max_run=3; "
                "sqrt(mean squared paired mean difference over the returned path)"
            ),
            "candidate_and_consensus_use_same_helper": True,
            "stored_stage2_consensus_values_reproduced": "36/36",
        },
        "provenance": {
            "stage1_manifest": {
                "path": str(stage1_manifest_path),
                "sha256": _sha256(stage1_manifest_path),
            },
            "stage2_manifest": {
                "path": str(stage2_manifest_path),
                "sha256": _sha256(stage2_manifest_path),
            },
            "candidate_table": {
                "path": str(out_dir / "candidate_scores" / "medoid_candidates.csv"),
                "sha256": _sha256(
                    out_dir / "candidate_scores" / "medoid_candidates.csv"
                ),
            },
            "training_matrices": {
                "path": str(
                    out_dir / "candidate_scores" / "training_distance_matrices.npz"
                ),
                "sha256": _sha256(
                    out_dir / "candidate_scores" / "training_distance_matrices.npz"
                ),
            },
            "consensus_comparisons": {
                "path": str(comparison_path),
                "sha256": _sha256(comparison_path),
            },
            "consensus_caches": {
                f"{dataset}/rank{rank}.npz": _sha256(
                    out_dir / "consensuses" / dataset / f"rank{rank}.npz"
                )
                for dataset in DATASETS
                for rank in range(1, 6)
            },
            "stage1_input_hashes_reverified": True,
            "candidate_ranking_revalidated": True,
            "dba_rerun": False,
            "candidate_selection_rerun": False,
            "filtering_clustering_splitting_preprocessing_rerun": False,
            "js447_visual_sanity_check_used_for_selection": False,
        },
        "outputs": {
            "detail": {
                "path": detail_path.name,
                "sha256": _sha256(detail_path),
            },
            "per_construct_summary": {
                "path": summary_path.name,
                "sha256": _sha256(summary_path),
            },
            "figure": {
                "path": f"figures/{figure_path.name}",
                "sha256": _sha256(figure_path),
                "dpi": FIGURE_DPI,
                "canvas_inches": [10, 6],
                "font_size": FIGURE_FONT_SIZE,
            },
        },
    }
    (out_dir / "stage2_followup_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Stage 2 descriptive follow-up complete: 36 perturbations", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
