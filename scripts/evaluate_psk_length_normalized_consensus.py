"""Fit and classify native and length-normalized PSK representations.

The runner preserves the established PSK preprocessing contract: native peptide
step traces are DNA-calibrated, screened by the upstream cluster-aware DBA rule,
and assigned to the seeded post-filter split before any length calculation.  Only
retained active-training traces select the data-derived target and enter DBA.

Two derived representations are fitted alongside a same-cohort native control:

* ``p95_max``: ceil(max(per-dataset 95th-percentile training length));
* ``fixed_100``: exactly 100 endpoint-inclusive relative-position points.

Both linearly interpolate mean and variance with ``numpy.interp`` before running
the ordinary closed-end DBA/DTW fit.  Held-out traces are transformed with the
same frozen target as their method's training traces before immutable profile
scoring.  Dwell is omitted.  Interpolated positions are correlated derived
points, not physical enzyme steps.

Run from the repository root:

    uv run python -m scripts.evaluate_psk_length_normalized_consensus
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from cowler.align.cost import cost_gaussian
from cowler.align.dtw import dtw_pairwise
from cowler.consensus.cluster import medoid
from cowler.consensus.dba import Barycenter, dba
from cowler.consensus.length_normalize import (
    ResampledSignal,
    TargetLengthSelection,
    resample_signal,
    select_fixed_target,
    select_percentile_max_target,
)
from cowler.eval.peptide_consensus import (
    DEFAULT_DBA_UPSTREAM_OUTLIER_Z,
    align_profiles,
    bootstrap_classification,
    fixed_profile_scores,
    percentile_interval,
)
from cowler.io.lut import predict_DNA_6mer_5_3
from cowler.io.normalize import robust_params
from scripts.evaluate_psk_dba_consensus import (
    DEFAULT_DATA_DIR,
    DEFAULT_JS445_DIR,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_FRACTION,
    TEMPLATE_DNA,
    PreparedEvent,
    SignalTrace,
    _apply_outlier_filter,  # pyright: ignore[reportPrivateUsage]
    _classification_rows,  # pyright: ignore[reportPrivateUsage]
    json_default,
    preprocess_dataset,
    stratified_random_split,
    write_csv,
)

FloatArr = npt.NDArray[np.float64]
BCN_DATASETS = tuple(f"JS{number}" for number in range(445, 454))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "tmp" / "psk_length_normalized"
DEFAULT_ICON_DIR = ROOT.parent / "Inkscape" / "psk_legends"
COMPARISON_GRID_LENGTH = 401
REPORTABLE_TRAINING_EVENTS = 523
REPORTABLE_HELDOUT_EVENTS = 518
REPORTABLE_P95_TARGET = 93
METHODS = ("native", "p95_max", "fixed_100")
PAIRED_METHODS = (
    ("p95_max", "native"),
    ("fixed_100", "native"),
    ("fixed_100", "p95_max"),
)
CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
FIGURE_DPI = 300
AXIS_TEXT_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12


@dataclass(frozen=True)
class FittedProfile:
    """One consensus plus the event identity of its initializing medoid."""

    profile: Barycenter
    medoid_event_id: int
    n_training: int


def prepare_heldout_trace(
    trace: SignalTrace,
    method: str,
    *,
    p95_target: int,
    fixed_target: int,
    min_std: float,
) -> SignalTrace | ResampledSignal:
    """Return the frozen representation used to score one held-out trace."""
    if method == "native":
        return trace
    if method == "p95_max":
        target = p95_target
    elif method == "fixed_100":
        target = fixed_target
    else:
        raise ValueError(f"unknown representation method {method!r}")
    return resample_signal(trace.mean, trace.std, target, min_std=min_std)


def validate_transformed_score_grid(
    method: str,
    scores: npt.ArrayLike,
    *,
    expected_events: int = REPORTABLE_HELDOUT_EVENTS,
) -> FloatArr:
    """Require the complete finite held-out score grid for transformed methods."""
    grid = np.asarray(scores, dtype=float)
    expected_shape = (expected_events, len(BCN_DATASETS))
    if grid.shape != expected_shape:
        raise RuntimeError(
            f"{method} score grid has shape {grid.shape}, expected {expected_shape}"
        )
    if method in {"p95_max", "fixed_100"} and not np.all(np.isfinite(grid)):
        n_bad = int(np.sum(~np.isfinite(grid)))
        raise RuntimeError(
            f"{method} score grid contains {n_bad} non-finite values; "
            "the reportable transformed run requires every score to be finite"
        )
    return grid


def shared_bootstrap_values(
    truth: npt.NDArray[np.int64],
    scores_by_method: Mapping[str, FloatArr],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, FloatArr]]:
    """Reuse one seed/truth order so every method receives identical draws."""
    return {
        method: bootstrap_classification(
            truth,
            scores_by_method[method],
            BCN_DATASETS,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for method in METHODS
    }


def add_representation_audit_fields(
    rows: Sequence[dict[str, Any]],
    events: Sequence[PreparedEvent],
    representations: Sequence[SignalTrace | ResampledSignal],
    *,
    method: str,
) -> None:
    """Add the frozen source/scoring representation to event score rows."""
    if len(rows) != len(events) or len(rows) != len(representations):
        raise RuntimeError("event score rows and held-out representations differ")
    for row, event, representation in zip(rows, events, representations):
        if row["dataset"] != event.dataset or row["event_id"] != event.event_id:
            raise RuntimeError("event score ordering changed before audit output")
        row.update(
            {
                "method": method,
                "true_label": row["true_class"],
                "source_length": int(event.peptide.mean.size),
                "scoring_length": int(representation.mean.size),
                "predicted_label": row["predicted_class"],
                "score_direction": "lower",
                "score_name": "path-normalized uncertainty-aware Gaussian-DTW",
            }
        )


def _length_summary(dataset: str, lengths: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{dataset} has no training trace lengths")
    return {
        "dataset": dataset,
        "n_training_traces": int(values.size),
        "min_length": int(np.min(values)),
        "q25_length": float(np.percentile(values, 25.0, method="linear")),
        "median_length": float(np.percentile(values, 50.0, method="linear")),
        "q75_length": float(np.percentile(values, 75.0, method="linear")),
        "q90_length": float(np.percentile(values, 90.0, method="linear")),
        "q95_length": float(np.percentile(values, 95.0, method="linear")),
        "max_length": int(np.max(values)),
    }


def _fit_profile(
    traces: Sequence[Any],
    event_ids: Sequence[int],
    *,
    max_iter: int,
    tol: float,
    min_std: float,
) -> FittedProfile:
    if len(traces) < 2 or len(event_ids) != len(traces):
        raise ValueError("profile fitting requires matching traces/event IDs and depth >= 2")
    distances = dtw_pairwise(
        traces, cost=cost_gaussian, normalize=False, max_run=3
    )
    medoid_index = medoid(distances)
    profile = dba(
        traces,
        medoid_index=medoid_index,
        distance_matrix=distances,
        normalize=False,
        max_iter=max_iter,
        tol=tol,
        min_depth=max(2, int(np.ceil(0.5 * len(traces)))),
        min_std=min_std,
        max_run=3,
    )
    return FittedProfile(
        profile=profile,
        medoid_event_id=int(event_ids[medoid_index]),
        n_training=len(traces),
    )


def _profile_rows(
    method: str,
    dataset: str,
    fitted: FittedProfile,
) -> list[dict[str, Any]]:
    profile = fitted.profile
    length = int(profile.mean.size)
    relative = np.linspace(0.0, 1.0, length)
    return [
        {
            "method": method,
            "dataset": dataset,
            "position": position,
            "relative_position": relative[position],
            "percent_position": 100.0 * relative[position],
            "mean": profile.mean[position],
            "std": profile.std[position],
            "depth": profile.depth[position],
            "supported": int(profile.supported[position]),
            "profile_length": length,
            "n_training_traces": fitted.n_training,
            "medoid_event_id": fitted.medoid_event_id,
        }
        for position in range(length)
    ]


def _diagnostic_row(
    method: str,
    dataset: str,
    fitted: FittedProfile,
    *,
    source_lengths: Sequence[int],
) -> dict[str, Any]:
    profile = fitted.profile
    target = int(profile.mean.size)
    lengths = np.asarray(source_lengths, dtype=int)
    objective = np.asarray(profile.objective, dtype=float)
    return {
        "method": method,
        "dataset": dataset,
        "n_training_traces": fitted.n_training,
        "medoid_event_id": fitted.medoid_event_id,
        "profile_length": target,
        "n_iter": profile.n_iter,
        "converged": int(profile.converged),
        "supported_fraction": float(np.mean(profile.supported)),
        "min_depth": int(np.min(profile.depth)),
        "median_depth": float(np.median(profile.depth)),
        "objective_start": float(objective[0]),
        "objective_end": float(objective[-1]),
        "quality_pass": int(profile.converged and np.mean(profile.supported) >= 0.8),
        "n_expanded": int(np.sum(lengths < target)) if method != "native" else 0,
        "n_compressed": int(np.sum(lengths > target)) if method != "native" else 0,
        "n_unchanged": int(np.sum(lengths == target)) if method != "native" else 0,
    }


def _resampling_rows(
    method: str,
    dataset: str,
    events: Sequence[PreparedEvent],
    traces: Sequence[ResampledSignal],
    *,
    outlier_threshold: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    for event, trace in zip(events, traces):
        source = trace.source_length
        target = trace.target_length
        if source < target:
            operation = "expanded"
        elif source > target:
            operation = "compressed"
        else:
            operation = "unchanged"
        manifest.append(
            {
                "method": method,
                "dataset": dataset,
                "event_id": event.event_id,
                "source_length": source,
                "target_length": target,
                "interval_scale_factor": (target - 1) / (source - 1),
                "operation": operation,
                "source_total_duration_s": float(np.sum(event.peptide.dwell)),
                "is_training": int(event.is_training),
                "upstream_retained": 1,
                "outlier_threshold": outlier_threshold,
                "split_seed": split_seed,
            }
        )
        values.extend(
            {
                "method": method,
                "dataset": dataset,
                "event_id": event.event_id,
                "position": position,
                "relative_position": trace.relative_position[position],
                "mean": trace.mean[position],
                "std": trace.std[position],
            }
            for position in range(target)
        )
    return manifest, values


def _curve_metrics(a: Barycenter, b: Barycenter) -> dict[str, float]:
    grid = np.linspace(0.0, 1.0, COMPARISON_GRID_LENGTH)
    a_values = np.interp(grid, np.linspace(0.0, 1.0, a.mean.size), a.mean)
    b_values = np.interp(grid, np.linspace(0.0, 1.0, b.mean.size), b.mean)
    difference = a_values - b_values
    correlation = (
        float(np.corrcoef(a_values, b_values)[0, 1])
        if np.std(a_values) > 0.0 and np.std(b_values) > 0.0
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "correlation": correlation,
        "max_absolute_difference": float(np.max(np.abs(difference))),
    }


def _comparison_rows(
    profiles: Mapping[str, Mapping[str, FittedProfile]],
) -> list[dict[str, Any]]:
    comparisons = (
        ("native", "p95_max"),
        ("native", "fixed_100"),
        ("p95_max", "fixed_100"),
    )
    rows: list[dict[str, Any]] = []
    for dataset in BCN_DATASETS:
        for method_a, method_b in comparisons:
            a = profiles[method_a][dataset].profile
            b = profiles[method_b][dataset].profile
            rows.append(
                {
                    "dataset": dataset,
                    "method_a": method_a,
                    "method_b": method_b,
                    "length_a": int(a.mean.size),
                    "length_b": int(b.mean.size),
                    **_curve_metrics(a, b),
                }
            )
    return rows


def _pairwise_rows(
    profiles: Mapping[str, Mapping[str, FittedProfile]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, fitted_by_dataset in profiles.items():
        for index, dataset_a in enumerate(BCN_DATASETS):
            for dataset_b in BCN_DATASETS[index + 1 :]:
                a = fitted_by_dataset[dataset_a].profile
                b = fitted_by_dataset[dataset_b].profile
                metrics = align_profiles(
                    a.mean,
                    b.mean,
                    supported_a=a.supported,
                    supported_b=b.supported,
                )
                rows.append(
                    {
                        "method": method,
                        "dataset_a": dataset_a,
                        "dataset_b": dataset_b,
                        "normalized_l2_dtw": metrics.normalized_l2_dtw,
                        "rmse": metrics.rmse,
                        "mae": metrics.mae,
                        "correlation": metrics.correlation,
                        "length_a": metrics.n_a,
                        "length_b": metrics.n_b,
                        "mean_warp_deviation": metrics.mean_warp_deviation,
                    }
                )
    return rows


def _target_rows(
    p95: TargetLengthSelection, fixed: TargetLengthSelection
) -> list[dict[str, Any]]:
    rows = [
        {
            "method": "p95_max",
            "dataset": dataset,
            "percentile": p95.percentile,
            "dataset_percentile_length": value,
            "determines_target": int(dataset in p95.determining_datasets),
            "target_length": p95.target_length,
            "rounding": "ceil(max(per-dataset percentile))",
        }
        for dataset, value in p95.per_dataset_percentile.items()
    ]
    rows.append(
        {
            "method": "fixed_100",
            "dataset": "ALL",
            "percentile": "",
            "dataset_percentile_length": "",
            "determines_target": 1,
            "target_length": fixed.target_length,
            "rounding": "fixed",
        }
    )
    return rows


def _plot_results(
    out_dir: Path,
    lengths_by_dataset: Mapping[str, Sequence[int]],
    p95_selection: TargetLengthSelection,
    profiles: Mapping[str, Mapping[str, FittedProfile]],
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    values = [lengths_by_dataset[dataset] for dataset in BCN_DATASETS]
    ax.boxplot(values, tick_labels=BCN_DATASETS, showfliers=True)
    q95 = [p95_selection.per_dataset_percentile[dataset] for dataset in BCN_DATASETS]
    ax.scatter(range(1, len(BCN_DATASETS) + 1), q95, color="#d55e00", label="dataset q95")
    ax.axhline(
        p95_selection.target_length,
        color="#0072b2",
        linestyle="--",
        label=f"selected T={p95_selection.target_length}",
    )
    ax.set_ylabel(
        "Retained training peptide steps", fontsize=AXIS_TEXT_FONT_SIZE
    )
    ax.set_title(
        "Post-upstream-filter training-trace length selection",
        fontsize=AXIS_TEXT_FONT_SIZE,
    )
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_training_trace_lengths.png", dpi=FIGURE_DPI)
    plt.close(fig)

    colors = {"native": "0.55", "p95_max": "#0072b2", "fixed_100": "#d55e00"}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    draw_consensus_comparison_grid(
        axes,
        profiles,
        methods=("native", "p95_max", "fixed_100"),
        colors=colors,
    )
    fig.supxlabel("Relative peptide position", fontsize=AXIS_TEXT_FONT_SIZE)
    fig.supylabel("DNA-calibrated current", fontsize=AXIS_TEXT_FONT_SIZE)
    fig.suptitle(
        "Native and pre-DBA length-normalized consensuses",
        fontsize=AXIS_TEXT_FONT_SIZE,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "02_consensus_comparison_grid.png", dpi=FIGURE_DPI)
    plt.close(fig)

    matrices = {
        method: np.vstack([profiles[method][dataset].profile.mean for dataset in BCN_DATASETS])
        for method in ("p95_max", "fixed_100")
    }
    all_values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), constrained_layout=True)
    image = None
    for ax, (method, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            extent=(0.0, 100.0, len(BCN_DATASETS) - 0.5, -0.5),
            vmin=float(np.min(all_values)),
            vmax=float(np.max(all_values)),
        )
        ax.set_yticks(range(len(BCN_DATASETS)), labels=BCN_DATASETS)
        ax.set_title(method, fontsize=AXIS_TEXT_FONT_SIZE)
        ax.set_ylabel("Dataset", fontsize=AXIS_TEXT_FONT_SIZE)
    axes[-1].set_xlabel(
        "Normalized peptide position (%)", fontsize=AXIS_TEXT_FONT_SIZE
    )
    if image is not None:
        colourbar = fig.colorbar(image, ax=axes)
        colourbar.set_label(
            "DNA-calibrated current", fontsize=AXIS_TEXT_FONT_SIZE
        )
    fig.savefig(figure_dir / "03_length_normalized_heatmaps.png", dpi=FIGURE_DPI)
    plt.close(fig)


def draw_consensus_comparison_grid(
    axes: Any,
    profiles: Mapping[str, Mapping[str, FittedProfile]],
    *,
    methods: Sequence[str],
    colors: Mapping[str, str],
    labels: Mapping[str, str] | None = None,
    linestyles: Mapping[str, str] | None = None,
    linewidths: Mapping[str, float] | None = None,
    zorders: Mapping[str, float] | None = None,
) -> None:
    """Draw method-matched consensus curves on an existing 3×3 axes grid.

    Keeping the drawing operation separate lets downstream manuscript figures reuse
    the established relative-position comparison without rerunning the analysis.
    """
    flat_axes = list(np.asarray(axes, dtype=object).flat)
    if len(flat_axes) != len(BCN_DATASETS):
        raise ValueError("consensus comparison requires one axis per BCN dataset")
    for method in methods:
        if method not in profiles:
            raise ValueError(f"profiles do not contain method {method!r}")
        if method not in colors:
            raise ValueError(f"colors do not contain method {method!r}")
    label_by_method = labels or {}
    linestyle_by_method = linestyles or {}
    linewidth_by_method = linewidths or {}
    zorder_by_method = zorders or {}
    for axis, dataset in zip(flat_axes, BCN_DATASETS):
        for method in methods:
            profile = profiles[method][dataset].profile
            x = np.linspace(0.0, 1.0, profile.mean.size)
            axis.plot(
                x,
                profile.mean,
                color=colors[method],
                linestyle=linestyle_by_method.get(method, "-"),
                linewidth=linewidth_by_method.get(method, 1.4),
                label=label_by_method.get(method, method),
                zorder=zorder_by_method.get(method, 2.0),
            )
        axis.set_title(dataset, fontsize=AXIS_TEXT_FONT_SIZE)
        axis.grid(alpha=0.15)
    flat_axes[0].legend(frameon=False, fontsize=LEGEND_FONT_SIZE)


def _plot_confusion_matrices(
    out_dir: Path,
    confusion_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    icon_dir: Path,
) -> None:
    """Render each method with the established Figure 10c icon styling."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    ink = "#222222"
    muted = "#777777"
    dba_color = "#B04A8B"
    light_grey = "#D0D0D0"
    grid = "#E8E8E8"
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.size": 10,
            "axes.titlesize": AXIS_TEXT_FONT_SIZE,
            "axes.labelsize": AXIS_TEXT_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "figure.titlesize": AXIS_TEXT_FONT_SIZE,
            "axes.edgecolor": light_grey,
            "axes.labelcolor": ink,
            "axes.titlecolor": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "grid.color": grid,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    icon_paths = {label: icon_dir / f"{label}.png" for label in BCN_DATASETS}
    missing = [str(path) for path in icon_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing construct icons: {missing}")

    summary_by_method = {str(row["method"]): row for row in summary_rows}
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    method_titles = {
        "native": "Native representation pipeline",
        "p95_max": "p95-normalized representation pipeline (T=93)",
        "fixed_100": "Fixed-100 representation pipeline",
    }
    for figure_index, method in enumerate(METHODS, start=4):
        counts = np.zeros((len(BCN_DATASETS), len(BCN_DATASETS)), dtype=int)
        for row in confusion_rows:
            if row["method"] != method:
                continue
            true_index = BCN_DATASETS.index(str(row["true_class"]))
            predicted_index = BCN_DATASETS.index(str(row["predicted_class"]))
            counts[true_index, predicted_index] = int(row["count"])
        row_totals = counts.sum(axis=1)
        if np.any(row_totals == 0):
            raise RuntimeError(f"{method} confusion matrix has an empty true class")
        fractions = counts / row_totals[:, None]

        fig, ax = plt.subplots(figsize=(11.5, 8), dpi=FIGURE_DPI)
        colour_map = mcolors.LinearSegmentedColormap.from_list(
            "outlier_magenta", ["#F7F1F5", dba_color]
        )
        image = ax.imshow(
            fractions, cmap=colour_map, vmin=0.0, vmax=1.0, aspect="equal"
        )
        for row in range(len(BCN_DATASETS)):
            for column in range(len(BCN_DATASETS)):
                count = int(counts[row, column])
                if count == 0:
                    continue
                fraction = float(fractions[row, column])
                red, green, blue, _ = image.cmap(image.norm(fraction))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                annotation_colour = "white" if luminance < 0.52 else ink
                ax.text(
                    column,
                    row - 0.08,
                    f"{fraction:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color=annotation_colour,
                )
                ax.text(
                    column,
                    row + 0.15,
                    f"n={count}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=annotation_colour,
                    alpha=0.78,
                )
        ax.set_xticks(
            np.arange(len(BCN_DATASETS)), BCN_DATASETS, rotation=45, ha="right"
        )
        ax.set_yticks(np.arange(len(BCN_DATASETS)), BCN_DATASETS)
        ax.set_xlabel("Predicted construct")
        ax.set_ylabel("True construct", labelpad=0)
        ax.yaxis.set_label_coords(-0.255, 0.5)
        for row, construct in enumerate(BCN_DATASETS):
            icon = OffsetImage(plt.imread(icon_paths[construct]), zoom=0.043)
            ax.add_artist(
                AnnotationBbox(
                    icon,
                    (-0.155, row),
                    xycoords=ax.get_yaxis_transform(),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(0.5, 0.5),
                    annotation_clip=False,
                )
            )
        summary = summary_by_method[method]
        ax.set_title(
            method_titles[method], fontsize=AXIS_TEXT_FONT_SIZE, pad=30
        )
        ax.text(
            0.5,
            1.015,
            f"Accuracy {float(summary['accuracy']):.1%} · "
            f"Balanced accuracy {float(summary['balanced_accuracy']):.1%}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color=muted,
        )
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(BCN_DATASETS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(BCN_DATASETS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colourbar_ticks = np.linspace(0.0, 1.0, 6)
        colourbar.set_ticks(colourbar_ticks.tolist())
        colourbar.set_ticklabels(
            [f"{value:.0%}" for value in colourbar_ticks]
        )
        colourbar.set_label("Fraction of true class")
        fig.tight_layout(rect=(0.08, 0, 1, 1))
        fig.savefig(
            figure_dir
            / f"{figure_index:02d}_{method}_confusion_matrix_with_construct_icons.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    js445_dir = Path(args.js445_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dna_profile = predict_DNA_6mer_5_3(TEMPLATE_DNA)
    valid = dna_profile["mean"].notna() & dna_profile["std"].notna()
    expected_dna = dna_profile.loc[valid, "mean"].to_numpy(float)
    target_shift, target_scale = robust_params(expected_dna)

    events_by_dataset: dict[str, list[PreparedEvent]] = {}
    preprocessing_rows: list[dict[str, Any]] = []
    for dataset in BCN_DATASETS:
        print(f"1. Preprocessing {dataset}...", flush=True)
        events, rows = preprocess_dataset(
            dataset,
            data_dir=data_dir,
            js445_dir=js445_dir,
            sensitivity=float(args.sensitivity),
            min_level_length=int(args.min_level_length),
            tail_threshold=float(args.tail_threshold),
            tail_control_steps=int(args.tail_control_steps),
            dna_window_steps=int(args.dna_window_steps),
            min_trace_steps=int(args.min_trace_steps),
            min_std=float(args.min_std),
            target_shift=target_shift,
            target_scale=target_scale,
        )
        events_by_dataset[dataset] = events
        preprocessing_rows.extend(rows)
    write_csv(out_dir / "preprocessing_manifest.csv", preprocessing_rows)

    print("2. Applying native-trace upstream outlier removal...", flush=True)
    events_by_dataset, outlier_rows, outlier_summary = _apply_outlier_filter(
        events_by_dataset,
        scope="upstream",
        threshold=float(args.outlier_z),
    )
    events_by_dataset, split_rows = stratified_random_split(
        events_by_dataset,
        train_fraction=float(args.train_fraction),
        seed=int(args.split_seed),
    )
    write_csv(out_dir / "upstream_outlier_metrics.csv", outlier_rows)
    write_csv(out_dir / "upstream_outlier_summary.csv", outlier_summary)
    write_csv(out_dir / "split_manifest.csv", split_rows)

    training_by_dataset = {
        dataset: [event for event in events_by_dataset[dataset] if event.is_training]
        for dataset in BCN_DATASETS
    }
    heldout_events = [
        event
        for dataset in BCN_DATASETS
        for event in events_by_dataset[dataset]
        if not event.is_training
    ]
    n_training_total = sum(len(events) for events in training_by_dataset.values())
    if n_training_total != REPORTABLE_TRAINING_EVENTS:
        raise RuntimeError(
            "reportable comparison requires exactly "
            f"{REPORTABLE_TRAINING_EVENTS} training events, found {n_training_total}"
        )
    if len(heldout_events) != REPORTABLE_HELDOUT_EVENTS:
        raise RuntimeError(
            "reportable comparison requires exactly "
            f"{REPORTABLE_HELDOUT_EVENTS} held-out events, found {len(heldout_events)}"
        )
    lengths_by_dataset = {
        dataset: [int(event.peptide.mean.size) for event in events]
        for dataset, events in training_by_dataset.items()
    }
    summary_rows = [
        _length_summary(dataset, lengths_by_dataset[dataset])
        for dataset in BCN_DATASETS
    ]
    pooled = [length for dataset in BCN_DATASETS for length in lengths_by_dataset[dataset]]
    summary_rows.append(_length_summary("POOLED_INFORMATIONAL", pooled))
    write_csv(out_dir / "training_trace_length_summary.csv", summary_rows)

    p95_selection = select_percentile_max_target(
        lengths_by_dataset, percentile=95.0
    )
    fixed_selection = select_fixed_target(100)
    if p95_selection.target_length != REPORTABLE_P95_TARGET:
        raise RuntimeError(
            "reportable comparison requires the frozen training-derived p95 "
            f"target {REPORTABLE_P95_TARGET}, found {p95_selection.target_length}"
        )
    write_csv(
        out_dir / "target_length_selection.csv",
        _target_rows(p95_selection, fixed_selection),
    )
    print(
        f"3. Selected p95 target {p95_selection.target_length}; fixed target 100...",
        flush=True,
    )

    profiles: dict[str, dict[str, FittedProfile]] = {
        "native": {},
        "p95_max": {},
        "fixed_100": {},
    }
    profile_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    resampling_manifest: list[dict[str, Any]] = []
    resampled_values: list[dict[str, Any]] = []

    for dataset in BCN_DATASETS:
        events = training_by_dataset[dataset]
        event_ids = [event.event_id for event in events]
        source_lengths = lengths_by_dataset[dataset]
        native_traces: list[SignalTrace] = [event.peptide for event in events]
        p95_traces = [
            resample_signal(
                event.peptide.mean,
                event.peptide.std,
                p95_selection.target_length,
                min_std=float(args.min_std),
            )
            for event in events
        ]
        fixed_traces = [
            resample_signal(
                event.peptide.mean,
                event.peptide.std,
                fixed_selection.target_length,
                min_std=float(args.min_std),
            )
            for event in events
        ]
        for method, traces in (
            ("p95_max", p95_traces),
            ("fixed_100", fixed_traces),
        ):
            manifest_rows, value_rows = _resampling_rows(
                method,
                dataset,
                events,
                traces,
                outlier_threshold=float(args.outlier_z),
                split_seed=int(args.split_seed),
            )
            resampling_manifest.extend(manifest_rows)
            resampled_values.extend(value_rows)

        print(f"   fitting {dataset}: native, p95, fixed-100", flush=True)
        for method, traces in (
            ("native", native_traces),
            ("p95_max", p95_traces),
            ("fixed_100", fixed_traces),
        ):
            fitted = _fit_profile(
                traces,
                event_ids,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                min_std=float(args.min_std),
            )
            profiles[method][dataset] = fitted
            profile_rows.extend(_profile_rows(method, dataset, fitted))
            diagnostics.append(
                _diagnostic_row(
                    method,
                    dataset,
                    fitted,
                    source_lengths=source_lengths,
                )
            )

    write_csv(out_dir / "resampling_manifest.csv", resampling_manifest)
    write_csv(out_dir / "resampled_training_traces.csv", resampled_values)
    write_csv(out_dir / "profiles.csv", profile_rows)
    write_csv(
        out_dir / "profiles_method_1_p95.csv",
        [row for row in profile_rows if row["method"] == "p95_max"],
    )
    write_csv(
        out_dir / "profiles_method_2_100_point.csv",
        [row for row in profile_rows if row["method"] == "fixed_100"],
    )
    write_csv(
        out_dir / "profiles_native_control.csv",
        [row for row in profile_rows if row["method"] == "native"],
    )
    write_csv(out_dir / "profile_diagnostics.csv", diagnostics)
    write_csv(out_dir / "profile_comparisons.csv", _comparison_rows(profiles))
    write_csv(out_dir / "pairwise_profile_distances.csv", _pairwise_rows(profiles))

    failed_profiles = [
        f"{method}/{dataset}"
        for method in METHODS
        for dataset in BCN_DATASETS
        if (
            not profiles[method][dataset].profile.converged
            or np.mean(profiles[method][dataset].profile.supported) < 0.8
        )
    ]
    if failed_profiles:
        raise RuntimeError(
            "held-out classification requires every profile to converge and pass "
            "the support gate; failed: " + ", ".join(failed_profiles)
        )

    print("4. Preparing and scoring frozen held-out representations...", flush=True)
    representations = {
        method: [
            prepare_heldout_trace(
                event.peptide,
                method,
                p95_target=p95_selection.target_length,
                fixed_target=fixed_selection.target_length,
                min_std=float(args.min_std),
            )
            for event in heldout_events
        ]
        for method in METHODS
    }
    truth = np.asarray(
        [BCN_DATASETS.index(event.dataset) for event in heldout_events],
        dtype=np.int64,
    )
    scores_by_method: dict[str, FloatArr] = {}
    for method in METHODS:
        score_labels, scores = fixed_profile_scores(
            representations[method],
            {
                dataset: profiles[method][dataset].profile
                for dataset in BCN_DATASETS
            },
            max_run=3,
        )
        if score_labels != BCN_DATASETS:
            raise RuntimeError(f"{method} profile score label order changed")
        scores_by_method[method] = validate_transformed_score_grid(method, scores)

    bootstrap_values = shared_bootstrap_values(
        truth,
        scores_by_method,
        n_bootstrap=int(args.classification_bootstrap),
        seed=int(args.bootstrap_seed),
    )
    analysis = (
        "held-out classification of the native, p95-normalized, and "
        "fixed-100 representation pipelines"
    )
    summaries: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    event_scores: list[dict[str, Any]] = []
    for method in METHODS:
        method_per_class, method_confusion, method_events, method_summary = (
            _classification_rows(
                analysis,
                "nine_class_bcn",
                heldout_events,
                BCN_DATASETS,
                truth,
                scores_by_method[method],
                n_bootstrap=int(args.classification_bootstrap),
                seed=int(args.bootstrap_seed),
            )
        )
        for row in [method_summary, *method_per_class, *method_confusion]:
            row["method"] = method
        add_representation_audit_fields(
            method_events,
            heldout_events,
            representations[method],
            method=method,
        )
        summaries.append(method_summary)
        per_class.extend(method_per_class)
        confusion.extend(method_confusion)
        event_scores.extend(method_events)

    bootstrap_rows = [
        {
            "replicate": repeat,
            "bootstrap_seed": int(args.bootstrap_seed),
            "method": method,
            **{
                metric: bootstrap_values[method][metric][repeat]
                for metric in CLASSIFICATION_METRICS
            },
        }
        for repeat in range(int(args.classification_bootstrap))
        for method in METHODS
    ]
    summary_by_method = {str(row["method"]): row for row in summaries}
    paired_rows: list[dict[str, Any]] = []
    for method_a, method_b in PAIRED_METHODS:
        for metric in CLASSIFICATION_METRICS:
            differences = (
                bootstrap_values[method_a][metric]
                - bootstrap_values[method_b][metric]
            )
            low, high = percentile_interval(differences)
            paired_rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "difference_definition": "method_a - method_b",
                    "estimate": (
                        float(summary_by_method[method_a][metric])
                        - float(summary_by_method[method_b][metric])
                    ),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": int(args.classification_bootstrap),
                    "bootstrap_seed": int(args.bootstrap_seed),
                }
            )
    write_csv(out_dir / "classification_summary.csv", summaries)
    write_csv(out_dir / "classification_per_class.csv", per_class)
    write_csv(out_dir / "classification_confusion_matrix.csv", confusion)
    write_csv(out_dir / "heldout_event_scores.csv", event_scores)
    write_csv(out_dir / "paired_bootstrap_metrics.csv", bootstrap_rows)
    write_csv(out_dir / "paired_classification_differences.csv", paired_rows)
    _plot_results(out_dir, lengths_by_dataset, p95_selection, profiles)
    _plot_confusion_matrices(
        out_dir,
        confusion,
        summaries,
        icon_dir=Path(args.icon_dir).resolve(),
    )

    summary = {
        "scope": (
            "held-out classification of the native, p95-normalized, and fixed-100 "
            "representation pipelines for JS445--JS453"
        ),
        "outlier_filter": {
            "scope": "upstream",
            "threshold": float(args.outlier_z),
            "input_axis": "native peptide steps",
        },
        "split": {
            "strategy": "post_filter_stratified_random",
            "train_fraction": float(args.train_fraction),
            "seed": int(args.split_seed),
        },
        "length_normalization": {
            "coordinate": "endpoint-inclusive relative position [0, 1]",
            "interpolation": "numpy.interp linear mean and variance",
            "dwell": "omitted",
            "points_are_physical_steps": False,
            "p95_max_target": p95_selection.target_length,
            "p95_per_dataset": p95_selection.per_dataset_percentile,
            "p95_determining_datasets": p95_selection.determining_datasets,
            "fixed_target": fixed_selection.target_length,
        },
        "consensus": {
            "engine": "DBA",
            "current_normalize": False,
            "dtw": "closed symmetric1 max_run=3",
            "dwell_output": False,
        },
        "training_counts": {
            dataset: len(training_by_dataset[dataset]) for dataset in BCN_DATASETS
        },
        "heldout_counts": {
            dataset: sum(
                not event.is_training for event in events_by_dataset[dataset]
            )
            for dataset in BCN_DATASETS
        },
        "heldout_classification": {
            "analysis": (
                "held-out classification of the native, p95-normalized, and "
                "fixed-100 representation pipelines"
            ),
            "n_events": len(heldout_events),
            "n_classes": len(BCN_DATASETS),
            "representations": {
                "native": "native held-out peptide steps",
                "p95_max": (
                    f"held-out mean/variance resampled to frozen T="
                    f"{p95_selection.target_length}"
                ),
                "fixed_100": "held-out mean/variance resampled to 100 points",
            },
            "score": "path-normalized uncertainty-aware Gaussian-DTW",
            "score_direction": "lower",
            "dtw": "closed symmetric1 max_run=3",
            "primary_metric": "balanced_accuracy",
            "bootstrap_replicates": int(args.classification_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "shared_paired_draws": True,
            "confusion_matrix_style": {
                "source": (
                    "notebooks/psk_variant_consensus_evaluation.ipynb "
                    "main_10c_dba_outlier_filter_z_ge_3_confusion_matrix_"
                    "with_construct_icons.png"
                ),
                "row_normalized": True,
                "counts_annotated": True,
                "construct_icons": True,
                "dpi": 300,
                "colormap": ["#F7F1F5", "#B04A8B"],
                "vmin": 0.0,
                "vmax": 1.0,
            },
            "score_grid_qc": {
                method: {
                    "shape": list(scores_by_method[method].shape),
                    "all_finite": bool(np.all(np.isfinite(scores_by_method[method]))),
                }
                for method in METHODS
            },
            "classification_summary": summaries,
            "paired_differences": paired_rows,
        },
        "outputs_include_heldout_representation_classification": True,
        "native_profiles_remain_authoritative_step_profiles": True,
        "within_run": True,
        "test_informed_upstream_filter": True,
        "external_validation": False,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(f"5. Wrote length-normalized analysis to {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--js445-dir", default=DEFAULT_JS445_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--icon-dir", default=DEFAULT_ICON_DIR)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--min-level-length", type=int, default=2)
    parser.add_argument("--tail-threshold", type=float, default=0.6)
    parser.add_argument("--tail-control-steps", type=int, default=5)
    parser.add_argument("--dna-window-steps", type=int, default=30)
    parser.add_argument("--min-trace-steps", type=int, default=5)
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--outlier-z", type=float, default=DEFAULT_DBA_UPSTREAM_OUTLIER_Z
    )
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--classification-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
