"""Run gated Stage 2 of the PSK medoid-selection robustness experiment.

The accepted Stage 1 artifacts own candidate selection and frozen input identity.
This runner verifies those artifacts, fits rank 1 for all nine constructs, and stops
before alternative fits unless every rank-1 profile reproduces the serialized frozen
baseline.  After that gate passes, it fits ranks 2--5, caches all 45 profiles, compares
each alternative with rank 1, and writes numbered Figure 3.  It never classifies reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from cowler.align.cost import cost_gaussian, cost_l2
from cowler.align.dtw import dtw, dtw_pairwise
from cowler.consensus.cluster import rank_medoid_candidates
from cowler.consensus.dba import Barycenter, dba
from scripts.evaluate_medoid_robustness import (
    DATASETS,
    FIGURE_DPI,
    FIGURE_FONT_SIZE,
    HIGHLIGHT_LINE_WIDTH,
    OTHER_LINE_WIDTH,
    ROOT,
    ROBUSTNESS_Y_LIMITS,
    _read_csv,  # pyright: ignore[reportPrivateUsage]
    _repository_version,  # pyright: ignore[reportPrivateUsage]
    _sha256,  # pyright: ignore[reportPrivateUsage]
)
from scripts.evaluate_psk_dba_consensus import write_csv

DEFAULT_OUT_DIR = ROOT / "tmp" / "medoid_robustness"
TOP_K = 5
BASELINE_ATOL = 0.0
FloatArr = npt.NDArray[np.float64]
Int64Arr = npt.NDArray[np.int64]
Array = npt.NDArray[Any]


def _stage1_manifest(out_dir: Path) -> tuple[dict[str, Any], Path]:
    immutable_path = out_dir / "stage1_run_manifest.json"
    current_path = out_dir / "run_manifest.json"
    source_path = immutable_path if immutable_path.exists() else current_path
    if not source_path.exists():
        raise FileNotFoundError(f"missing accepted Stage 1 manifest: {source_path}")
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    validation = manifest.get("validation", manifest.get("stage1_validation", {}))
    if not bool(validation.get("all_stage1_checks_pass")):
        raise RuntimeError("Stage 1 validation is not accepted in the input manifest")
    if validation.get("rank_1_matches_production") != "9/9":
        raise RuntimeError("Stage 1 did not reproduce all nine production medoids")
    if immutable_path.exists():
        return manifest, immutable_path
    immutable_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest, immutable_path


def _verify_stage1_hashes(manifest: Mapping[str, Any]) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise RuntimeError("Stage 1 manifest has no input hashes")
    for name, value in inputs.items():
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid Stage 1 input record for {name}")
        path = Path(str(value["path"]))
        expected = str(value["sha256"])
        if not path.exists() or _sha256(path) != expected:
            raise RuntimeError(f"Stage 1 provenance hash differs for {name}: {path}")


def _combined_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _trace_input_paths(manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    """Return representation-aware trace inputs, with historical-manifest fallback."""

    inputs = manifest["inputs"]
    cache_record = inputs.get("analysis_trace_cache", inputs.get("native_trace_cache"))
    rows_record = inputs.get("trace_cache_companion_rows")
    if not isinstance(cache_record, dict) or not isinstance(rows_record, dict):
        raise RuntimeError("Stage 1 manifest lacks accepted trace-cache inputs")
    return Path(str(cache_record["path"])), Path(str(rows_record["path"]))


def _load_candidates_and_matrices(
    out_dir: Path,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, FloatArr],
    dict[str, Int64Arr],
    dict[str, Int64Arr],
]:
    candidate_path = out_dir / "candidate_scores" / "medoid_candidates.csv"
    matrix_path = out_dir / "candidate_scores" / "training_distance_matrices.npz"
    candidates = _read_csv(candidate_path)
    by_dataset: dict[str, list[dict[str, str]]] = {}
    matrices: dict[str, FloatArr] = {}
    event_ids: dict[str, Int64Arr] = {}
    event_orders: dict[str, Int64Arr] = {}
    with np.load(matrix_path, allow_pickle=False) as cached:
        for dataset in DATASETS:
            rows = sorted(
                [row for row in candidates if row["construct"] == dataset],
                key=lambda row: int(row["candidate_rank"]),
            )
            if [int(row["candidate_rank"]) for row in rows] != list(range(1, TOP_K + 1)):
                raise RuntimeError(f"{dataset}: Stage 1 candidate ranks are incomplete")
            D = np.asarray(cached[f"{dataset}_distance"], dtype=float)
            ids = np.asarray(cached[f"{dataset}_event_id"], dtype=np.int64)
            orders = np.asarray(cached[f"{dataset}_event_order"], dtype=np.int64)
            if D.shape != (ids.size, ids.size) or orders.shape != ids.shape:
                raise RuntimeError(f"{dataset}: Stage 1 training cache axes differ")
            ranking = rank_medoid_candidates(D, top_k=TOP_K)
            for rank, (row, expected) in enumerate(zip(rows, ranking), start=1):
                local_index = int(row["local_training_index"])
                if local_index != expected.index:
                    raise RuntimeError(f"{dataset} rank {rank}: local medoid index changed")
                if int(row["original_read_id"]) != int(ids[local_index]):
                    raise RuntimeError(f"{dataset} rank {rank}: candidate event ID changed")
                if int(row["original_input_index"]) != int(orders[local_index]):
                    raise RuntimeError(f"{dataset} rank {rank}: candidate input index changed")
                if float(row["row_sum"]) != expected.row_sum:
                    raise RuntimeError(f"{dataset} rank {rank}: candidate score changed")
            by_dataset[dataset] = rows
            matrices[dataset] = D
            event_ids[dataset] = ids
            event_orders[dataset] = orders
    return by_dataset, matrices, event_ids, event_orders


def _load_training_traces(
    stage1_manifest: Mapping[str, Any],
    event_ids: Mapping[str, Int64Arr],
) -> dict[str, list[tuple[FloatArr, FloatArr, FloatArr]]]:
    trace_cache, trace_rows_path = _trace_input_paths(stage1_manifest)
    trace_rows = _read_csv(trace_rows_path)
    trace_index = {
        (row["dataset"], int(row["event_id"])): index
        for index, row in enumerate(trace_rows)
        if row["dataset"] in DATASETS
    }
    training: dict[str, list[tuple[FloatArr, FloatArr, FloatArr]]] = {}
    with np.load(trace_cache, allow_pickle=False) as traces:
        for dataset in DATASETS:
            selected: list[tuple[FloatArr, FloatArr, FloatArr]] = []
            for event_id in event_ids[dataset]:
                identity = (dataset, int(event_id))
                if identity not in trace_index:
                    raise RuntimeError(f"missing frozen trace for {identity}")
                index = trace_index[identity]
                mean = np.asarray(traces[f"mean_{index}"], dtype=float)
                std = np.asarray(traces[f"std_{index}"], dtype=float)
                dwell = np.asarray(traces[f"dwell_{index}"], dtype=float)
                if mean.ndim != 1 or std.shape != mean.shape or dwell.shape != mean.shape:
                    raise RuntimeError(f"invalid frozen trace channels for {identity}")
                if int(trace_rows[index]["n_steps"]) != mean.size:
                    raise RuntimeError(f"frozen trace length differs for {identity}")
                selected.append((mean, std, dwell))
            training[dataset] = selected
    return training


def _fit_candidate(
    traces: Sequence[tuple[FloatArr, FloatArr, FloatArr]],
    *,
    local_medoid_index: int,
) -> tuple[Barycenter, float]:
    started = time.perf_counter()
    profile = dba(
        traces,
        medoid_index=local_medoid_index,
        normalize=False,
        max_iter=50,
        tol=1e-4,
        min_depth=max(2, int(np.ceil(0.5 * len(traces)))),
        min_std=1e-3,
        max_run=3,
    )
    return profile, time.perf_counter() - started


def _cache_profile(
    path: Path,
    profile: Barycenter,
    candidate: Mapping[str, str],
    *,
    input_fingerprint: str,
) -> None:
    np.savez_compressed(
        path,
        mean=profile.mean,
        std=profile.std,
        dwell=profile.dwell,
        depth=profile.depth,
        supported=profile.supported,
        objective=profile.objective,
        delta=profile.delta,
        construct=np.asarray(candidate["construct"]),
        candidate_rank=np.asarray(int(candidate["candidate_rank"]), dtype=np.int64),
        candidate_event_id=np.asarray(int(candidate["original_read_id"]), dtype=np.int64),
        candidate_input_index=np.asarray(
            int(candidate["original_input_index"]), dtype=np.int64
        ),
        medoid_index=np.asarray(profile.medoid_index, dtype=np.int64),
        n_iter=np.asarray(profile.n_iter, dtype=np.int64),
        converged=np.asarray(profile.converged),
        consensus_length=np.asarray(profile.mean.size, dtype=np.int64),
        final_delta=np.asarray(profile.delta[-1]),
        normalize=np.asarray(False),
        max_run=np.asarray(3, dtype=np.int64),
        max_iter=np.asarray(50, dtype=np.int64),
        tol=np.asarray(1e-4),
        min_std=np.asarray(1e-3),
        endpoint_handling=np.asarray("closed"),
        input_fingerprint=np.asarray(input_fingerprint),
    )


def _diagnostic_row(
    dataset: str,
    candidate: Mapping[str, str],
    profile: Barycenter,
    *,
    n_training: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "construct": dataset,
        "candidate_rank": int(candidate["candidate_rank"]),
        "candidate_event_id": int(candidate["original_read_id"]),
        "candidate_input_index": int(candidate["original_input_index"]),
        "local_training_index": int(candidate["local_training_index"]),
        "n_training": n_training,
        "consensus_length": int(profile.mean.size),
        "converged": int(profile.converged),
        "n_iter": profile.n_iter,
        "final_delta": float(profile.delta[-1]),
        "tol": 1e-4,
        "objective_start": float(profile.objective[0]),
        "objective_end": float(profile.objective[-1]),
        "supported_fraction": float(np.mean(profile.supported)),
        "min_depth": int(np.min(profile.depth)),
        "median_depth": float(np.median(profile.depth)),
        "runtime_seconds": runtime_seconds,
        "medoid_supplied_explicitly": 1,
    }


def _baseline_profiles(
    baseline_dir: Path,
    *,
    method: str,
    profile_filename: str,
) -> tuple[dict[str, dict[str, Array]], dict[str, dict[str, str]]]:
    profile_path = baseline_dir / profile_filename
    diagnostic_path = baseline_dir / "profile_diagnostics.csv"
    profile_rows = _read_csv(profile_path)
    diagnostic_rows = _read_csv(diagnostic_path)
    profiles: dict[str, dict[str, Array]] = {}
    diagnostics: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        rows = sorted(
            [
                row
                for row in profile_rows
                if row["dataset"] == dataset and row.get("method", method) == method
            ],
            key=lambda row: int(row["position"]),
        )
        method_diagnostics = [
            row
            for row in diagnostic_rows
            if row["dataset"] == dataset and row["method"] == method
        ]
        if not rows or len(method_diagnostics) != 1:
            raise RuntimeError(f"{dataset}: serialized {method} baseline is incomplete")
        profiles[dataset] = {
            "mean": np.asarray([float(row["mean"]) for row in rows]),
            "std": np.asarray([float(row["std"]) for row in rows]),
            "depth": np.asarray([int(row["depth"]) for row in rows], dtype=np.int64),
            "supported": np.asarray(
                [bool(int(row["supported"])) for row in rows], dtype=bool
            ),
        }
        diagnostics[dataset] = method_diagnostics[0]
    return profiles, diagnostics


def _rank1_gate_row(
    dataset: str,
    candidate: Mapping[str, str],
    profile: Barycenter,
    baseline: Mapping[str, Array],
    diagnostic: Mapping[str, str],
) -> dict[str, Any]:
    shapes_match = all(profile.mean.shape == baseline[field].shape for field in baseline)
    mean_delta = (
        float(np.max(np.abs(profile.mean - baseline["mean"])))
        if shapes_match
        else float("inf")
    )
    std_delta = (
        float(np.max(np.abs(profile.std - baseline["std"])))
        if shapes_match
        else float("inf")
    )
    arrays_match = bool(
        shapes_match
        and np.allclose(profile.mean, baseline["mean"], rtol=0.0, atol=BASELINE_ATOL)
        and np.allclose(profile.std, baseline["std"], rtol=0.0, atol=BASELINE_ATOL)
        and np.array_equal(profile.depth, baseline["depth"])
        and np.array_equal(profile.supported, baseline["supported"])
    )
    metadata_match = bool(
        int(candidate["original_read_id"]) == int(diagnostic["medoid_event_id"])
        and profile.mean.size == int(diagnostic["profile_length"])
        and profile.n_iter == int(diagnostic["n_iter"])
        and int(profile.converged) == int(diagnostic["converged"])
        and float(profile.objective[0]) == float(diagnostic["objective_start"])
        and float(profile.objective[-1]) == float(diagnostic["objective_end"])
    )
    passed = arrays_match and metadata_match
    return {
        "construct": dataset,
        "check": "rank1_reproduces_serialized_frozen_baseline",
        "passed": int(passed),
        "candidate_event_id": int(candidate["original_read_id"]),
        "baseline_medoid_event_id": int(diagnostic["medoid_event_id"]),
        "candidate_consensus_length": int(profile.mean.size),
        "baseline_consensus_length": int(diagnostic["profile_length"]),
        "candidate_n_iter": profile.n_iter,
        "baseline_n_iter": int(diagnostic["n_iter"]),
        "candidate_converged": int(profile.converged),
        "baseline_converged": int(diagnostic["converged"]),
        "max_abs_mean_difference": mean_delta,
        "max_abs_std_difference": std_delta,
        "depth_exact": int(shapes_match and np.array_equal(profile.depth, baseline["depth"])),
        "supported_exact": int(
            shapes_match and np.array_equal(profile.supported, baseline["supported"])
        ),
        "serialized_dwell_available": 0,
    }


def _comparison_row(
    dataset: str,
    baseline_candidate: Mapping[str, str],
    candidate: Mapping[str, str],
    baseline: Barycenter,
    alternative: Barycenter,
) -> dict[str, Any]:
    gaussian_distance = float(
        dtw_pairwise(
            [baseline, alternative],
            cost=cost_gaussian,
            normalize=False,
            max_run=3,
        )[0, 1]
    )
    aligned_rmse, aligned_path_length = _aligned_rmse(
        baseline.mean, alternative.mean
    )
    return {
        "construct": dataset,
        "candidate_rank": int(candidate["candidate_rank"]),
        "rank1_event_id": int(baseline_candidate["original_read_id"]),
        "candidate_event_id": int(candidate["original_read_id"]),
        "delta_candidate_mean_score": float(candidate["delta_mean_score"]),
        "candidate_distance_to_rank1": float(candidate["distance_to_rank1"]),
        "rank1_consensus_length": int(baseline.mean.size),
        "candidate_consensus_length": int(alternative.mean.size),
        "gaussian_dtw_distance": gaussian_distance,
        "aligned_rmse": aligned_rmse,
        "aligned_path_length": aligned_path_length,
        "rank1_converged": int(baseline.converged),
        "candidate_converged": int(alternative.converged),
        "interpretable": int(baseline.converged and alternative.converged),
    }


def _aligned_rmse(
    baseline_mean: FloatArr, alternative_mean: FloatArr
) -> tuple[float, int]:
    """Return Stage 2's closed-DTW aligned waveform RMSE and path length."""
    alignment = dtw(cost_l2(baseline_mean, alternative_mean), max_run=3)
    aligned_difference = (
        baseline_mean[alignment.path[:, 0]]
        - alternative_mean[alignment.path[:, 1]]
    )
    return (
        float(np.sqrt(np.mean(aligned_difference**2))),
        int(alignment.path.shape[0]),
    )


def _plot_consensuses(
    output_path: Path,
    profiles: Mapping[str, Mapping[int, Barycenter]],
    candidates: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    x_label: str,
) -> None:
    import matplotlib.pyplot as plt

    colours = {1: "#B04A8B", 2: "#0072B2", 3: "#009E73", 4: "#E69F00", 5: "#56B4E9"}
    figure, axes = plt.subplots(3, 3, figsize=(24, 16), constrained_layout=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        for candidate in reversed(candidates[dataset]):
            rank = int(candidate["candidate_rank"])
            profile = profiles[dataset][rank]
            axis.plot(
                np.arange(profile.mean.size),
                profile.mean,
                color=colours[rank],
                linewidth=HIGHLIGHT_LINE_WIDTH if rank == 1 else OTHER_LINE_WIDTH,
                alpha=1.0 if rank == 1 else 0.9,
                label=f"#{rank} baseline" if rank == 1 else f"#{rank}",
                zorder=20 if rank == 1 else 10 - rank,
            )
        axis.set_title(dataset, fontsize=FIGURE_FONT_SIZE)
        axis.set_ylim(*ROBUSTNESS_Y_LIMITS)
        axis.tick_params(labelsize=FIGURE_FONT_SIZE)
        handles, labels = axis.get_legend_handles_labels()
        order = np.argsort([int(label.split()[0][1:]) for label in labels])
        axis.legend(
            [handles[int(index)] for index in order],
            [labels[int(index)] for index in order],
            frameon=False,
            fontsize=FIGURE_FONT_SIZE,
        )
    figure.suptitle(
        "DBA consensus by medoid initialization", fontsize=FIGURE_FONT_SIZE
    )
    figure.supxlabel(x_label, fontsize=FIGURE_FONT_SIZE)
    figure.supylabel("DNA-calibrated current", fontsize=FIGURE_FONT_SIZE)
    figure.savefig(
        output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    consensus_dir = out_dir / "consensuses"
    figure_dir = out_dir / "figures"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    stage1, immutable_stage1_path = _stage1_manifest(out_dir)
    _verify_stage1_hashes(stage1)
    candidates, _, event_ids, _ = _load_candidates_and_matrices(out_dir)
    training = _load_training_traces(stage1, event_ids)
    baseline_dir = Path(str(stage1["baseline"]))
    baseline_method = str(stage1.get("baseline_method", "native"))
    baseline_files = stage1.get("baseline_files", {})
    if not isinstance(baseline_files, dict):
        raise RuntimeError("Stage 1 baseline file mapping is invalid")
    profile_filename = str(baseline_files.get("profiles", "profiles_native_control.csv"))
    diagnostic_filename = str(baseline_files.get("diagnostics", "profile_diagnostics.csv"))
    baseline_profile_path = baseline_dir / profile_filename
    baseline_diagnostic_path = baseline_dir / diagnostic_filename
    baseline_profiles, baseline_diagnostics = _baseline_profiles(
        baseline_dir,
        method=baseline_method,
        profile_filename=profile_filename,
    )
    candidate_path = out_dir / "candidate_scores" / "medoid_candidates.csv"
    matrix_path = out_dir / "candidate_scores" / "training_distance_matrices.npz"
    input_fingerprint = _combined_fingerprint(
        [
            immutable_stage1_path,
            candidate_path,
            matrix_path,
            *_trace_input_paths(stage1),
            baseline_profile_path,
            baseline_diagnostic_path,
        ]
    )

    profiles: dict[str, dict[int, Barycenter]] = {dataset: {} for dataset in DATASETS}
    diagnostic_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    print("1. Fitting and validating rank-1 DBA baselines...", flush=True)
    for dataset in DATASETS:
        candidate = candidates[dataset][0]
        profile, runtime = _fit_candidate(
            training[dataset],
            local_medoid_index=int(candidate["local_training_index"]),
        )
        profiles[dataset][1] = profile
        dataset_dir = consensus_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _cache_profile(
            dataset_dir / "rank1.npz",
            profile,
            candidate,
            input_fingerprint=input_fingerprint,
        )
        diagnostic_rows.append(
            _diagnostic_row(
                dataset,
                candidate,
                profile,
                n_training=len(training[dataset]),
                runtime_seconds=runtime,
            )
        )
        gate = _rank1_gate_row(
            dataset,
            candidate,
            profile,
            baseline_profiles[dataset],
            baseline_diagnostics[dataset],
        )
        gate_rows.append(gate)
        print(
            f"   {dataset}: event={candidate['original_read_id']}, "
            f"L={profile.mean.size}, iterations={profile.n_iter}, "
            f"converged={profile.converged}, reproduced={bool(gate['passed'])}",
            flush=True,
        )

    write_csv(out_dir / "stage2_validation_checks.csv", gate_rows)
    write_csv(out_dir / "dba_fit_diagnostics.csv", diagnostic_rows)
    failed_gate = [row["construct"] for row in gate_rows if not bool(row["passed"])]
    if failed_gate:
        failure = {
            "stage": 2,
            "gate_passed": False,
            "failed_constructs": failed_gate,
            "alternatives_generated": False,
            "reason": "rank-1 DBA did not reproduce the serialized frozen baseline",
        }
        (out_dir / "stage2_gate_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            "Stage 2 rank-1 reproduction gate failed for " + ", ".join(failed_gate)
        )

    print("2. Rank-1 gate passed 9/9; fitting medoid ranks 2–5...", flush=True)
    for dataset in DATASETS:
        for candidate in candidates[dataset][1:]:
            rank = int(candidate["candidate_rank"])
            profile, runtime = _fit_candidate(
                training[dataset],
                local_medoid_index=int(candidate["local_training_index"]),
            )
            profiles[dataset][rank] = profile
            _cache_profile(
                consensus_dir / dataset / f"rank{rank}.npz",
                profile,
                candidate,
                input_fingerprint=input_fingerprint,
            )
            diagnostic_rows.append(
                _diagnostic_row(
                    dataset,
                    candidate,
                    profile,
                    n_training=len(training[dataset]),
                    runtime_seconds=runtime,
                )
            )
            print(
                f"   {dataset} rank {rank}: event={candidate['original_read_id']}, "
                f"L={profile.mean.size}, iterations={profile.n_iter}, "
                f"converged={profile.converged}",
                flush=True,
            )
    write_csv(out_dir / "dba_fit_diagnostics.csv", diagnostic_rows)

    print("3. Comparing converged consensuses with rank 1...", flush=True)
    comparison_rows = [
        _comparison_row(
            dataset,
            candidates[dataset][0],
            candidates[dataset][rank - 1],
            profiles[dataset][1],
            profiles[dataset][rank],
        )
        for dataset in DATASETS
        for rank in range(2, TOP_K + 1)
    ]
    write_csv(out_dir / "consensus_comparisons.csv", comparison_rows)
    _plot_consensuses(
        figure_dir / "3_dba_consensuses_3x3.png",
        profiles,
        candidates,
        x_label=(
            "100-point normalized position (1–100)"
            if int(stage1.get("trace_representation", {}).get("target_length", 0)) == 100
            else "DBA consensus position"
        ),
    )

    nonconverged = [
        f"{row['construct']}/rank{row['candidate_rank']}"
        for row in diagnostic_rows
        if not bool(row["converged"])
    ]
    stage2_manifest = {
        "stage": 2,
        "stage_1_accepted": True,
        "stage_2_implemented": True,
        "stage_3_implemented": False,
        "repository": _repository_version(),
        "datasets": list(DATASETS),
        "baseline_method": baseline_method,
        "n_dba_fits": len(diagnostic_rows),
        "rank1_reproduction_gate": {
            "passed": True,
            "constructs_reproduced": "9/9",
            "float_atol": BASELINE_ATOL,
            "serialized_fields_checked": [
                "mean",
                "std",
                "depth",
                "supported",
                "medoid_event_id",
                "profile_length",
                "n_iter",
                "converged",
                "objective_start",
                "objective_end",
            ],
            "serialized_baseline_dwell_available": False,
            "checks_file": "stage2_validation_checks.csv",
        },
        "dba_settings": {
            "medoid_supplied_explicitly": True,
            "normalize": False,
            "max_run": 3,
            "max_iter": 50,
            "tol": 1e-4,
            "min_std": 1e-3,
            "min_depth": "max(2, ceil(0.5 * n_training))",
            "endpoint_handling": "closed",
        },
        "convergence": {
            "all_45_converged": not nonconverged,
            "nonconverged": nonconverged,
            "statistic": "max(abs(new_mean - old_mean))",
            "history_cached_as": "delta",
        },
        "provenance": {
            "stage1_manifest": str(immutable_stage1_path),
            "stage1_manifest_sha256": _sha256(immutable_stage1_path),
            "input_fingerprint": input_fingerprint,
            "baseline_profiles": {
                "path": str(baseline_profile_path),
                "sha256": _sha256(baseline_profile_path),
            },
            "baseline_diagnostics": {
                "path": str(baseline_diagnostic_path),
                "sha256": _sha256(baseline_diagnostic_path),
            },
            "stage1_input_hashes_reverified": True,
            "candidate_ranking_revalidated_from_cached_training_matrices": True,
            "js447_context_plot_used_for_selection": False,
        },
        "codebase_conflicts": [
            (
                "serialized baseline profiles do not include dwell; rank-1 gate "
                "checks every serialized profile channel and deterministic fit metadata"
            ),
            (
                "Barycenter previously did not expose the stopping delta; Stage 2 adds "
                "an observation-only delta history without changing DBA behavior"
            ),
        ],
        "outputs": {
            "fit_diagnostics": "dba_fit_diagnostics.csv",
            "validation_checks": "stage2_validation_checks.csv",
            "comparisons": "consensus_comparisons.csv",
            "consensus_cache": "consensuses/<construct>/rank<1-5>.npz",
            "consensus_figure": "figures/3_dba_consensuses_3x3.png",
        },
    }
    stage2_manifest_path = out_dir / "stage2_run_manifest.json"
    stage2_manifest_path.write_text(
        json.dumps(stage2_manifest, indent=2) + "\n", encoding="utf-8"
    )

    combined = dict(stage1)
    combined["stage"] = 2
    combined["stages_2_or_3_implemented"] = {"stage_2": True, "stage_3": False}
    combined["stage1_validation"] = combined.pop("validation")
    combined["stage2_validation"] = stage2_manifest["rank1_reproduction_gate"]
    combined["stage2"] = {
        "manifest": "stage2_run_manifest.json",
        "n_dba_fits": len(diagnostic_rows),
        "all_45_converged": not nonconverged,
    }
    stage1_outputs = combined.get("outputs")
    if not isinstance(stage1_outputs, dict):
        raise RuntimeError("Stage 1 manifest outputs are invalid")
    stage2_outputs = stage2_manifest.get("outputs")
    if not isinstance(stage2_outputs, dict):
        raise RuntimeError("Stage 2 manifest outputs are invalid")
    combined["outputs"] = {
        **stage1_outputs,
        **stage2_outputs,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Stage 2 complete: rank-1 gate 9/9; "
        f"converged={len(diagnostic_rows) - len(nonconverged)}/{len(diagnostic_rows)}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
