"""Evaluate PSK PTM-variant DBA consensuses on the deposited FAST5 family.

This is the first runnable milestone from
``docs/psk-variant-consensus-evaluation.md``.  It implements:

* stored isConsensus training/held-out labels applied after preprocessing;
* annotation-assisted peptide extraction after complete-event CPIC segmentation;
* one global terminal-tail rule (trailing source-relative current > 0.6);
* per-event robust affine calibration from the common DNA region to the DNA LUT;
* upstream cluster-aware distance-outlier removal before the active split
  (z >= 3.5 by default);
* 100-point peptide mean/variance interpolation after filtering, before split assignment
  (use ``--representation native`` for native segmented traces);
* supervised DBA profiles with ``normalize=False``;
* blocked split-profile reproducibility and whole-event bootstrap refits;
* held-out Gaussian-DTW calls, equal-depth sensitivity, targeted contrasts; and
* DNA-window, tail, nuisance-only, reversed, and shuffled controls.

The output remains a signal profile, never an amino-acid sequence.  Gaussian-DTW
scores are lower-is-better distances, not likelihoods.  All results are
within-run and run/pore/variant confounded.  The upstream-default held-out result
is also test-informed because held-out traces participate in outlier scoring.

Run from the repository root, for example:

    MPLCONFIGDIR=tmp/matplotlib .venv/bin/python \
        scripts/evaluate_psk_dba_consensus.py \
        --data-dir tmp/otherfast5 --js445-dir tmp \
        --profile-bootstrap 10
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import h5py
import numpy as np
import numpy.typing as npt

from cowler.align.cost import cost_gaussian
from cowler.align.dtw import dtw_pairwise
from cowler.align.segment import find_steps
from cowler.consensus.cluster import medoid
from cowler.consensus.dba import Barycenter, dba
from cowler.consensus.length_normalize import resample_signal
from cowler.eval.peptide_consensus import (
    DEFAULT_DBA_UPSTREAM_OUTLIER_Z,
    align_profiles,
    binary_rank_metrics,
    bootstrap_classification,
    classification_metrics,
    distance_outlier_metrics,
    fixed_profile_scores,
    percentile_interval,
    select_distance_outlier_clusters,
    separation_ratio,
)
from cowler.io.lut import predict_DNA_6mer_5_3
from cowler.io.normalize import normalize_signal, robust_params

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "tmp" / "otherfast5"
DEFAULT_JS445_DIR = ROOT / "tmp"
DEFAULT_OUT_DIR = ROOT / "tmp" / "psk_variant_consensus"
OUTLIER_NEIGHBORS = 5
DEFAULT_SPLIT_SEED = 20260810
DEFAULT_TRAIN_FRACTION = 0.5
TEMPLATE_DNA = (
    "TTACTGAAGTCTCACGTGCCTGGTATATTAGCGTCCACTCTCACTATCGGATTCTACATCGGTCGTAGCC"
)


@dataclass(frozen=True)
class MoleculeInfo:
    peptide: str
    linker: str


MOLECULES: dict[str, MoleculeInfo] = {
    "JS445": MoleculeInfo("YIYTQ", "BCN-triazole"),
    "JS446": MoleculeInfo("sYIYTQ", "BCN-triazole"),
    "JS447": MoleculeInfo("YIsYTQ", "BCN-triazole"),
    "JS448": MoleculeInfo("sYIsYTQ", "BCN-triazole"),
    "JS449": MoleculeInfo("sYIpYTQ", "BCN-triazole"),
    "JS450": MoleculeInfo("pYIsYTQ", "BCN-triazole"),
    "JS451": MoleculeInfo("pYIYTQ", "BCN-triazole"),
    "JS452": MoleculeInfo("YIpYTQ", "BCN-triazole"),
    "JS453": MoleculeInfo("pYIpYTQ", "BCN-triazole"),
    "JS527": MoleculeInfo("PEG8-YIYTQ", "BCN-triazole-PEG8"),
    "JS528": MoleculeInfo("PEG8-YIsYTQ", "BCN-triazole-PEG8"),
    "JS529": MoleculeInfo("PEG8-sYIYTQ", "BCN-triazole-PEG8"),
    "JS530": MoleculeInfo("PEG8-sYIsYTQ", "BCN-triazole-PEG8"),
}

TASKS: dict[str, tuple[str, ...]] = {
    "nine_class_bcn": tuple(f"JS{number}" for number in range(445, 454)),
    "four_sulfation": ("JS445", "JS446", "JS447", "JS448"),
    "four_phosphorylation": ("JS445", "JS451", "JS452", "JS453"),
    "four_peg8": ("JS527", "JS528", "JS529", "JS530"),
}

TARGETED_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("sulfate_site", "JS446", "JS447"),
    ("phosphate_site", "JS451", "JS452"),
    ("chemistry_site_1", "JS446", "JS451"),
    ("chemistry_site_2", "JS447", "JS452"),
    ("swapped_mixed_ptm", "JS449", "JS450"),
    ("double_sulfate_vs_phosphate", "JS448", "JS453"),
    ("linker_unmodified", "JS445", "JS527"),
    ("linker_site_1_sulfate", "JS446", "JS529"),
    ("linker_site_2_sulfate", "JS447", "JS528"),
    ("linker_double_sulfate", "JS448", "JS530"),
)


FloatArr = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SignalTrace:
    mean: FloatArr
    std: FloatArr
    dwell: FloatArr

    def reversed(self) -> "SignalTrace":
        return SignalTrace(
            self.mean[::-1].copy(), self.std[::-1].copy(), self.dwell[::-1].copy()
        )

    def shuffled(self, seed: int) -> "SignalTrace":
        order = np.random.default_rng(seed).permutation(self.mean.size)
        return SignalTrace(
            self.mean[order], self.std[order], self.dwell[order]
        )


@dataclass(frozen=True)
class PreparedEvent:
    dataset: str
    event_id: int
    event_order: int
    stored_is_consensus: bool
    is_training: bool
    peptide: SignalTrace
    dna_window: SignalTrace
    tail: SignalTrace | None
    dna_shift: float
    dna_scale: float
    nuisance: FloatArr
    tail_steps: int
    raw_peptide_steps: int
    peptide_start_sample_retained: int = -1
    peptide_end_sample_retained: int = -1


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write heterogeneous result rows with stable first-seen field ordering."""
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    """Serialize numpy scalars/arrays and paths in run-summary JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _dataset_paths(
    dataset: str, data_dir: Path, js445_dir: Path
) -> tuple[Path, Path]:
    directory = js445_dir if dataset == "JS445" else data_dir
    raw_path = directory / f"{dataset}_synthetic.fast5"
    annotation_path = directory / f"{dataset}_synthetic.annot.fast5"
    if not raw_path.exists() or not annotation_path.exists():
        raise FileNotFoundError(f"missing FAST5 pair for {dataset} in {directory}")
    return raw_path, annotation_path


def _load_current(raw_path: Path) -> tuple[FloatArr, float]:
    with h5py.File(raw_path, "r") as handle:
        signal = cast(h5py.Dataset, handle["Raw/Channel_1/Signal"])
        raw = np.asarray(signal[:], dtype=float)
        attrs = signal.parent.attrs
        offset = float(cast(Any, attrs["offset"]))
        signal_range = float(cast(Any, attrs["range"]))
        digitisation = float(cast(Any, attrs["digitisation"]))
        sample_rate = float(cast(Any, attrs["sample_rate"]))
    return (raw + offset) * signal_range / digitisation, sample_rate


def _step_trace(
    current: FloatArr,
    event_start: int,
    local_steps: Sequence[Any],
    selection: npt.NDArray[np.bool_],
    sample_rate: float,
    *,
    min_std: float,
) -> SignalTrace:
    chosen = [step for step, keep in zip(local_steps, selection) if keep]
    means = np.empty(len(chosen), dtype=float)
    stds = np.empty(len(chosen), dtype=float)
    dwell = np.empty(len(chosen), dtype=float)
    for index, step in enumerate(chosen):
        start = event_start + int(step.start_sample)
        end = event_start + int(step.end_sample)
        samples = current[start:end]
        means[index] = float(np.mean(samples))
        stds[index] = max(float(np.std(samples)), min_std)
        dwell[index] = (end - start) / sample_rate
    return SignalTrace(means, stds, dwell)


def _calibrate_trace(
    trace: SignalTrace,
    *,
    observed_shift: float,
    observed_scale: float,
    target_shift: float,
    target_scale: float,
    min_std: float,
) -> SignalTrace:
    factor = target_scale / observed_scale
    return SignalTrace(
        (trace.mean - observed_shift) * factor + target_shift,
        np.maximum(trace.std * factor, min_std),
        trace.dwell.copy(),
    )


def preprocess_dataset(
    dataset: str,
    *,
    data_dir: Path,
    js445_dir: Path,
    sensitivity: float,
    min_level_length: int,
    tail_threshold: float,
    tail_control_steps: int,
    dna_window_steps: int,
    min_trace_steps: int,
    min_std: float,
    target_shift: float,
    target_scale: float,
) -> tuple[list[PreparedEvent], list[dict[str, Any]]]:
    """Segment, region-select, tail-trim, and DNA-calibrate one construct."""

    raw_path, annotation_path = _dataset_paths(dataset, data_dir, js445_dir)
    current, sample_rate = _load_current(raw_path)
    with h5py.File(annotation_path, "r") as handle:
        events = np.asarray(cast(h5py.Dataset, handle["Events"])[:])
    events = np.sort(events, order="index")

    prepared: list[PreparedEvent] = []
    manifest: list[dict[str, Any]] = []
    for event_order, event in enumerate(events):
        event_id = int(event["index"])
        event_start = int(event["start_idx"])
        event_end = int(event["end_idx"])
        dna_start = int(event["d_start_idx"])
        dna_end = int(event["d_end_idx"])
        peptide_start = int(event["p_start_idx"])
        peptide_end = int(event["p_end_idx"])
        row: dict[str, Any] = {
            "dataset": dataset,
            "event_id": event_id,
            "event_order": event_order,
            "stored_is_consensus": int(event["isConsensus"]),
            "is_hand_pick": int(event["isHandPick"]),
            "status": "excluded",
            "exclusion_reason": "",
            "sample_rate_hz": sample_rate,
            "event_start_sample": event_start,
            "event_end_sample": event_end,
            "dna_start_sample": dna_start,
            "dna_end_sample": dna_end,
            "peptide_start_sample": peptide_start,
            "peptide_end_sample": peptide_end,
        }
        try:
            event_current = current[event_start:event_end]
            local_steps = find_steps(
                normalize_signal(event_current),
                sensitivity=sensitivity,
                min_level_length=min_level_length,
            )
            midpoints = np.asarray(
                [
                    event_start
                    + (int(step.start_sample) + int(step.end_sample)) // 2
                    for step in local_steps
                ],
                dtype=np.int64,
            )
            dna_mask = (midpoints >= dna_start) & (midpoints < dna_end)
            peptide_mask = (midpoints >= peptide_start) & (midpoints < peptide_end)
            peptide_steps = [
                step for step, selected in zip(local_steps, peptide_mask) if selected
            ]
            dna_raw = _step_trace(
                current,
                event_start,
                local_steps,
                dna_mask,
                sample_rate,
                min_std=min_std,
            )
            peptide_raw = _step_trace(
                current,
                event_start,
                local_steps,
                peptide_mask,
                sample_rate,
                min_std=min_std,
            )
            if dna_raw.mean.size < min_trace_steps:
                raise ValueError("too_few_dna_steps")
            if peptide_raw.mean.size < min_trace_steps:
                raise ValueError("too_few_peptide_steps")

            below = np.flatnonzero(peptide_raw.mean <= tail_threshold)
            keep = int(below[-1] + 1) if below.size else 0
            tail_steps = int(peptide_raw.mean.size - keep)
            if keep < min_trace_steps:
                raise ValueError("tail_rule_left_too_few_peptide_steps")
            peptide_start_retained = event_start + int(
                peptide_steps[0].start_sample
            )
            peptide_end_retained = event_start + int(
                peptide_steps[keep - 1].end_sample
            )
            peptide_kept = SignalTrace(
                peptide_raw.mean[:keep], peptide_raw.std[:keep], peptide_raw.dwell[:keep]
            )
            tail_raw = (
                SignalTrace(
                    peptide_raw.mean[keep : keep + tail_control_steps],
                    peptide_raw.std[keep : keep + tail_control_steps],
                    peptide_raw.dwell[keep : keep + tail_control_steps],
                )
                if tail_steps >= tail_control_steps
                else None
            )

            observed_shift, observed_scale = robust_params(dna_raw.mean)
            if not np.isfinite(observed_scale) or observed_scale <= 0.0:
                raise ValueError("invalid_dna_scale")
            dna_calibrated = _calibrate_trace(
                dna_raw,
                observed_shift=observed_shift,
                observed_scale=observed_scale,
                target_shift=target_shift,
                target_scale=target_scale,
                min_std=min_std,
            )
            peptide_calibrated = _calibrate_trace(
                peptide_kept,
                observed_shift=observed_shift,
                observed_scale=observed_scale,
                target_shift=target_shift,
                target_scale=target_scale,
                min_std=min_std,
            )
            tail_calibrated = (
                _calibrate_trace(
                    tail_raw,
                    observed_shift=observed_shift,
                    observed_scale=observed_scale,
                    target_shift=target_shift,
                    target_scale=target_scale,
                    min_std=min_std,
                )
                if tail_raw is not None
                else None
            )
            n_dna_window = min(dna_window_steps, dna_calibrated.mean.size)
            dna_window = SignalTrace(
                dna_calibrated.mean[-n_dna_window:],
                dna_calibrated.std[-n_dna_window:],
                dna_calibrated.dwell[-n_dna_window:],
            )
            pre_dna = current[event_start:dna_start]
            open_current = float(np.median(pre_dna)) if pre_dna.size else float("nan")
            nuisance = np.asarray(
                [
                    open_current,
                    float(event["start_time"]),
                    (peptide_end - peptide_start) / sample_rate,
                    float(peptide_raw.mean.size),
                    float(tail_steps),
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(nuisance)):
                raise ValueError("non_finite_nuisance_feature")

            prepared.append(
                PreparedEvent(
                    dataset=dataset,
                    event_id=event_id,
                    event_order=event_order,
                    stored_is_consensus=bool(event["isConsensus"]),
                    is_training=bool(event["isConsensus"]),
                    peptide=peptide_calibrated,
                    dna_window=dna_window,
                    tail=tail_calibrated,
                    dna_shift=observed_shift,
                    dna_scale=observed_scale / target_scale,
                    nuisance=nuisance,
                    tail_steps=tail_steps,
                    raw_peptide_steps=int(peptide_raw.mean.size),
                    peptide_start_sample_retained=peptide_start_retained,
                    peptide_end_sample_retained=peptide_end_retained,
                )
            )
            row.update(
                {
                    "status": "included",
                    "n_event_steps": len(local_steps),
                    "n_dna_steps": int(dna_raw.mean.size),
                    "n_dna_window_steps": int(n_dna_window),
                    "n_peptide_steps_raw": int(peptide_raw.mean.size),
                    "n_peptide_steps_kept": int(peptide_calibrated.mean.size),
                    "peptide_start_sample_retained": peptide_start_retained,
                    "peptide_end_sample_retained": peptide_end_retained,
                    "n_peptide_samples_retained": (
                        peptide_end_retained - peptide_start_retained
                    ),
                    "n_tail_steps": tail_steps,
                    "n_tail_control_steps": (
                        int(tail_control_steps) if tail_raw is not None else 0
                    ),
                    "tail_fraction": tail_steps / peptide_raw.mean.size,
                    "terminal_raw_current": float(peptide_raw.mean[-1]),
                    "terminal_above_threshold": int(
                        peptide_raw.mean[-1] > tail_threshold
                    ),
                    "dna_observed_shift": observed_shift,
                    "dna_observed_scale": observed_scale,
                    "dna_to_lut_factor": target_scale / observed_scale,
                    "open_current": nuisance[0],
                    "event_time_s": nuisance[1],
                    "peptide_duration_s": nuisance[2],
                }
            )
        except (ValueError, RuntimeError) as exc:
            row["exclusion_reason"] = str(exc)
        manifest.append(row)
    return prepared, manifest


def peptide_representation(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    representation: str,
    *,
    min_std: float,
) -> dict[str, list[PreparedEvent]]:
    """Transform retained peptide traces, preserving splits and native QC fields.

    Unit dwell is an internal placeholder on the relative-position axis. It must
    not be exported as measured duration. Native events remain unmodified.
    """
    if representation not in ("native", "fixed_100"):
        raise ValueError("representation must be native or fixed_100")
    transformed: dict[str, list[PreparedEvent]] = {}
    for dataset, events in events_by_dataset.items():
        transformed[dataset] = []
        for event in events:
            if representation == "native":
                transformed[dataset].append(event)
                continue
            signal = resample_signal(
                event.peptide.mean, event.peptide.std, 100, min_std=min_std
            )
            transformed[dataset].append(replace(
                event,
                peptide=SignalTrace(signal.mean, signal.std, np.ones(100)),
            ))
    return transformed


def _fit_profile(
    events: Sequence[PreparedEvent],
    channel: str,
    *,
    max_iter: int,
    tol: float,
    min_std: float,
) -> tuple[Barycenter, int]:
    traces = [cast(SignalTrace, getattr(event, channel)) for event in events]
    if len(traces) < 2:
        raise ValueError(f"{channel} profile requires at least two events")
    distance_matrix = dtw_pairwise(
        traces, cost=cost_gaussian, normalize=False, max_run=3
    )
    medoid_index = medoid(distance_matrix)
    profile = dba(
        traces,
        medoid_index=medoid_index,
        distance_matrix=distance_matrix,
        normalize=False,
        max_iter=max_iter,
        tol=tol,
        min_depth=max(2, int(np.ceil(0.5 * len(traces)))),
        min_std=min_std,
        max_run=3,
    )
    return profile, events[medoid_index].event_id


def _profile_rows(
    dataset: str,
    analysis: str,
    profile: Barycenter,
    *,
    replicate: int = -1,
    split_group: str = "",
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "analysis": analysis,
            "replicate": replicate,
            "split_group": split_group,
            "position": position,
            "mean": profile.mean[position],
            "std": profile.std[position],
            "dwell": profile.dwell[position],
            "depth": profile.depth[position],
            "supported": int(profile.supported[position]),
        }
        for position in range(profile.mean.size)
    ]


def _profile_summary_row(
    dataset: str,
    analysis: str,
    profile: Barycenter,
    medoid_event_id: int,
    n_events: int,
    *,
    replicate: int = -1,
    split_group: str = "",
) -> dict[str, Any]:
    objective = profile.objective
    monotone = bool(np.all(np.diff(objective) <= 1e-8)) if objective.size > 1 else True
    return {
        "dataset": dataset,
        "analysis": analysis,
        "replicate": replicate,
        "split_group": split_group,
        "n_events": n_events,
        "medoid_event_id": medoid_event_id,
        "profile_length": int(profile.mean.size),
        "n_iter": profile.n_iter,
        "converged": int(profile.converged),
        "supported_fraction": float(np.mean(profile.supported)),
        "min_depth": int(np.min(profile.depth)),
        "median_depth": float(np.median(profile.depth)),
        "objective_start": float(objective[0]),
        "objective_end": float(objective[-1]),
        "objective_nonincreasing": int(monotone),
        "quality_pass": int(profile.converged and np.mean(profile.supported) >= 0.8),
    }


def _even_depth(events: Sequence[PreparedEvent], depth: int) -> list[PreparedEvent]:
    ordered = sorted(events, key=lambda event: event.event_order)
    if depth > len(ordered):
        raise ValueError("requested depth exceeds available events")
    if depth == len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, depth)
    indices = np.unique(np.rint(positions).astype(int))
    if indices.size != depth:
        indices = np.arange(depth)
    return [ordered[int(index)] for index in indices]


def blocked_groups(
    events: Sequence[PreparedEvent], assignment: int
) -> tuple[list[PreparedEvent], list[PreparedEvent], str]:
    """Return one deterministic pair of disjoint event-order block groups."""
    ordered = sorted(events, key=lambda event: event.event_order)
    blocks = [list(block) for block in np.array_split(np.asarray(ordered, dtype=object), 4)]
    pairings = (
        ((0, 1), (2, 3), "early_vs_late"),
        ((0, 2), (1, 3), "alternating_blocks"),
        ((0, 3), (1, 2), "outer_vs_inner"),
    )
    a_blocks, b_blocks, name = pairings[assignment % len(pairings)]
    group_a = [cast(PreparedEvent, event) for block in a_blocks for event in blocks[block]]
    group_b = [cast(PreparedEvent, event) for block in b_blocks for event in blocks[block]]
    depth = min(len(group_a), len(group_b))
    return group_a[:depth], group_b[:depth], name


def stored_split(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
) -> tuple[dict[str, list[PreparedEvent]], list[dict[str, Any]]]:
    """Apply source labels to retained events without rebalancing either group."""
    assigned = {
        dataset: [
            replace(event, is_training=event.stored_is_consensus)
            for event in events
        ]
        for dataset, events in events_by_dataset.items()
    }
    rows = [
        {
            "dataset": dataset,
            "event_id": event.event_id,
            "analysis": "primary_split",
            "replicate": -1,
            "assignment": "stored_isConsensus",
            "group": "training" if event.is_training else "held_out",
            "draw_order": -1,
            "stored_group": "training" if event.stored_is_consensus else "held_out",
            "train_fraction": "",
            "split_seed": "",
            "active": 1,
        }
        for dataset, events in assigned.items()
        for event in events
    ]
    return assigned, rows


def stratified_random_split(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[dict[str, list[PreparedEvent]], list[dict[str, Any]]]:
    """Assign retained events randomly within each construct.

    The returned events carry the active split in ``is_training`` so profile fitting
    and held-out scoring consume the new assignment. ``stored_is_consensus`` remains
    unchanged and is retained in the manifest solely as provenance.
    Each construct uses an independent deterministic RNG stream, making its split
    insensitive to mapping order or the presence of other constructs.
    """

    if not np.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train fraction must be finite and strictly between zero and one"
        )
    if seed < 0:
        raise ValueError("split seed must be non-negative")

    split: dict[str, list[PreparedEvent]] = {}
    manifest: list[dict[str, Any]] = []
    for dataset, source_events in events_by_dataset.items():
        events = sorted(
            source_events, key=lambda event: (event.event_order, event.event_id)
        )
        n_events = len(events)
        n_training = int(np.floor(n_events * train_fraction + 0.5))
        if n_training < 2 or n_training >= n_events:
            raise ValueError(
                f"{dataset} cannot support {train_fraction:g} training fraction: "
                f"need at least two training and one held-out event, found {n_events}"
            )

        dataset_seed = np.random.SeedSequence([int(seed), *dataset.encode("utf-8")])
        order = np.random.default_rng(dataset_seed).permutation(n_events)
        training_indices = {int(index) for index in order[:n_training]}
        draw_order = {int(index): rank for rank, index in enumerate(order)}
        assigned: list[PreparedEvent] = []
        for index, event in enumerate(events):
            is_training = index in training_indices
            assigned.append(replace(event, is_training=is_training))
            manifest.append(
                {
                    "dataset": dataset,
                    "event_id": event.event_id,
                    "analysis": "primary_split",
                    "replicate": -1,
                    "assignment": "post_filter_stratified_random",
                    "group": "training" if is_training else "held_out",
                    "draw_order": draw_order[index],
                    "stored_group": (
                        "training" if event.stored_is_consensus else "held_out"
                    ),
                    "train_fraction": train_fraction,
                    "split_seed": seed,
                    "active": 1,
                }
            )
        split[dataset] = assigned
    return split, manifest


def _apply_outlier_filter(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    scope: str,
    threshold: float | None,
) -> tuple[
    dict[str, list[PreparedEvent]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Calibrate and optionally remove cluster-aware distance outliers.

    ``scope='upstream'`` scores every eligible event before consulting the stored
    split and can therefore remove both training and held-out events.  The
    ``scope='training'`` compatibility mode calibrates and filters stored training
    events only.  A ``None`` threshold computes the complete audit tables for the
    selected scope but retains every event for unfiltered reproduction.

    This workflow assumes that every labelled dataset is homologous and contains
    one genuine signal species.  Multiple genuine species within a dataset are out
    of scope because a valid minority species may be rejected as outlying.
    """

    if scope not in {"upstream", "training"}:
        raise ValueError("outlier scope must be 'upstream' or 'training'")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0.0):
        raise ValueError("outlier threshold must be finite and non-negative")

    filtered: dict[str, list[PreparedEvent]] = {}
    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset, events in events_by_dataset.items():
        candidates = (
            list(events)
            if scope == "upstream"
            else [event for event in events if event.stored_is_consensus]
        )
        if len(candidates) < 2:
            raise ValueError(f"{dataset} requires at least two outlier candidates")
        traces = [event.peptide for event in candidates]
        distances = dtw_pairwise(
            traces, cost=cost_gaussian, normalize=False, max_run=3
        )
        selection = select_distance_outlier_clusters(distances)
        clustering = selection.clustering
        metrics = distance_outlier_metrics(
            distances,
            clustering.labels,
            n_neighbors=OUTLIER_NEIGHBORS,
            knn_within_clusters=True,
        )
        outlier_score = np.maximum(
            np.maximum(metrics.robust_z_score, 0.0),
            np.maximum(metrics.knn_robust_z_score, 0.0),
        )
        rejected = (
            outlier_score >= threshold
            if threshold is not None
            else np.zeros(len(candidates), dtype=bool)
        )
        retained_ids = {
            event.event_id
            for index, event in enumerate(candidates)
            if not bool(rejected[index])
        }
        if len(retained_ids) < 2:
            raise RuntimeError(
                f"{scope} outlier filter leaves fewer than two events for {dataset}"
            )
        filtered[dataset] = [
            event
            for event in events
            if (
                (scope == "training" and not event.stored_is_consensus)
                or event.event_id in retained_ids
            )
        ]

        cluster_counts = np.bincount(clustering.labels)
        for index, event in enumerate(candidates):
            centre = int(metrics.medoid_index[index])
            z3 = bool(outlier_score[index] >= 3.0)
            z3_5 = bool(outlier_score[index] >= 3.5)
            metric_rows.append(
                {
                    "dataset": dataset,
                    "event_id": event.event_id,
                    "event_order": event.event_order,
                    "stored_is_consensus": int(event.stored_is_consensus),
                    "calibration_scope": scope,
                    "cluster": int(clustering.labels[index]) + 1,
                    "cluster_size": int(metrics.cluster_size[index]),
                    "medoid_flag": int(metrics.medoid_flag[index]),
                    "medoid_event_id": candidates[centre].event_id,
                    "normalized_dtw_to_medoid": metrics.normalized_dtw_to_medoid[index],
                    "robust_z_score": metrics.robust_z_score[index],
                    "zmedoid": metrics.robust_z_score[index],
                    "knn_k": min(
                        OUTLIER_NEIGHBORS,
                        int(metrics.cluster_size[index]) - 1,
                    ),
                    "knn_distance": metrics.knn_distance[index],
                    "knn_robust_z_score": metrics.knn_robust_z_score[index],
                    "zknn": metrics.knn_robust_z_score[index],
                    "distance_outlier_score": outlier_score[index],
                    "outlier_score": outlier_score[index],
                    "candidate_outlier_z_ge_3": int(z3),
                    "candidate_outlier_z_ge_3_5": int(z3_5),
                    "threshold": "" if threshold is None else threshold,
                    "outlier_z_ge_3_5": int(z3_5),
                    "active_threshold": "" if threshold is None else threshold,
                    "retained": int(not rejected[index]),
                    "excluded": int(rejected[index]),
                }
            )

        off_diagonal = distances[~np.eye(len(candidates), dtype=bool)]
        candidate_training = sum(event.stored_is_consensus for event in candidates)
        candidate_heldout = len(candidates) - candidate_training
        retained_training = sum(
            event.stored_is_consensus and event.event_id in retained_ids
            for event in candidates
        )
        retained_heldout = sum(
            not event.stored_is_consensus and event.event_id in retained_ids
            for event in candidates
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "calibration_scope": scope,
                "n_candidates": len(candidates),
                "n_retained": len(retained_ids),
                "n_excluded": int(np.sum(rejected)),
                "n_training_candidates": candidate_training,
                "n_training_retained": retained_training,
                "n_training_excluded": candidate_training - retained_training,
                "n_heldout_candidates": candidate_heldout,
                "n_heldout_retained": retained_heldout,
                "n_heldout_excluded": candidate_heldout - retained_heldout,
                "filter_enabled": int(threshold is not None),
                "active_threshold": "" if threshold is None else threshold,
                "selected_clusters": selection.selected_clusters,
                "cluster_sizes": "/".join(
                    str(int(count)) for count in sorted(cluster_counts, reverse=True)
                ),
                "required_cluster_size": selection.required_cluster_size,
                "best_valid_silhouette": selection.best_valid_silhouette,
                "silhouette_threshold": selection.silhouette_threshold,
                "distance_shift_for_silhouette": clustering.distance_shift,
                "median_pairwise_dtw": float(np.median(off_diagonal)),
                "q90_pairwise_dtw": float(np.quantile(off_diagonal, 0.90)),
                "n_candidates_z_ge_3": int(np.sum(outlier_score >= 3.0)),
                "n_candidates_z_ge_3_5": int(np.sum(outlier_score >= 3.5)),
            }
        )
    return filtered, metric_rows, summary_rows


def _metric_row(
    dataset: str,
    analysis: str,
    assignment: str,
    replicate: int,
    profile_a: Barycenter,
    profile_b: Barycenter,
) -> dict[str, Any]:
    metrics = align_profiles(
        profile_a.mean,
        profile_b.mean,
        supported_a=profile_a.supported,
        supported_b=profile_b.supported,
    )
    return {
        "dataset": dataset,
        "analysis": analysis,
        "assignment": assignment,
        "replicate": replicate,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "correlation": metrics.correlation,
        "normalized_l2_dtw": metrics.normalized_l2_dtw,
        "length_a": metrics.n_a,
        "length_b": metrics.n_b,
        "path_length": metrics.path_length,
        "mean_warp_deviation": metrics.mean_warp_deviation,
        "max_warp_deviation": metrics.max_warp_deviation,
    }


def _fit_profiles(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    channel: str,
    equal_depth: int | None,
    max_iter: int,
    tol: float,
    min_std: float,
) -> tuple[dict[str, Barycenter], list[dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, Barycenter] = {}
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    analysis = "published_split" if equal_depth is None else "equal_depth"
    for dataset in MOLECULES:
        training = [event for event in events_by_dataset[dataset] if event.is_training]
        if channel == "tail":
            training = [event for event in training if event.tail is not None]
        selected = training if equal_depth is None else _even_depth(training, equal_depth)
        profile, medoid_event = _fit_profile(
            selected, channel, max_iter=max_iter, tol=tol, min_std=min_std
        )
        profiles[dataset] = profile
        summaries.append(
            _profile_summary_row(
                dataset, f"{analysis}_{channel}", profile, medoid_event, len(selected)
            )
        )
        rows.extend(_profile_rows(dataset, f"{analysis}_{channel}", profile))
    return profiles, summaries, rows


def _task_event_grid(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    labels: Sequence[str],
    channel: str,
) -> tuple[list[PreparedEvent], list[SignalTrace], npt.NDArray[np.int64]]:
    events: list[PreparedEvent] = []
    traces: list[SignalTrace] = []
    truth: list[int] = []
    for class_index, dataset in enumerate(labels):
        for event in events_by_dataset[dataset]:
            if event.is_training:
                continue
            trace = cast(SignalTrace | None, getattr(event, channel))
            if trace is None:
                continue
            events.append(event)
            traces.append(trace)
            truth.append(class_index)
    return events, traces, np.asarray(truth, dtype=np.int64)


def _nuisance_scores(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]], labels: Sequence[str]
) -> tuple[list[PreparedEvent], npt.NDArray[np.int64], FloatArr]:
    training = [
        event
        for dataset in labels
        for event in events_by_dataset[dataset]
        if event.is_training
    ]
    test = [
        event
        for dataset in labels
        for event in events_by_dataset[dataset]
        if not event.is_training
    ]
    train_values = np.vstack([event.nuisance for event in training])
    shift = np.median(train_values, axis=0)
    scale = 1.4826 * np.median(np.abs(train_values - shift), axis=0)
    scale[scale <= 0.0] = 1.0
    standardized = (train_values - shift) / scale
    centroids = np.vstack(
        [
            np.mean(
                standardized[
                    np.asarray([event.dataset == label for event in training], dtype=bool)
                ],
                axis=0,
            )
            for label in labels
        ]
    )
    test_values = (np.vstack([event.nuisance for event in test]) - shift) / scale
    scores = np.sqrt(np.sum((test_values[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    truth = np.asarray([labels.index(event.dataset) for event in test], dtype=np.int64)
    return test, truth, scores


def _classification_rows(
    analysis: str,
    task: str,
    events: Sequence[PreparedEvent],
    labels: Sequence[str],
    truth: npt.NDArray[np.int64],
    scores: FloatArr,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metrics = classification_metrics(truth, scores, labels)
    inadmissible = np.isposinf(scores)
    boot = bootstrap_classification(
        truth, scores, labels, n_bootstrap=n_bootstrap, seed=seed
    )
    intervals = {name: percentile_interval(values) for name, values in boot.items()}
    summary = {
        "analysis": analysis,
        "task": task,
        "n_events": len(events),
        "n_classes": len(labels),
        "n_inadmissible_scores": int(np.sum(inadmissible)),
        "n_events_with_inadmissible_score": int(np.sum(np.any(inadmissible, axis=1))),
        "accuracy": metrics.accuracy,
        "accuracy_ci_low": intervals["accuracy"][0],
        "accuracy_ci_high": intervals["accuracy"][1],
        "balanced_accuracy": metrics.balanced_accuracy,
        "balanced_accuracy_ci_low": intervals["balanced_accuracy"][0],
        "balanced_accuracy_ci_high": intervals["balanced_accuracy"][1],
        "macro_f1": metrics.macro_f1,
        "macro_f1_ci_low": intervals["macro_f1"][0],
        "macro_f1_ci_high": intervals["macro_f1"][1],
        "chance_accuracy": 1.0 / len(labels),
    }
    per_class = [
        {
            "analysis": analysis,
            "task": task,
            "class": label,
            "count": int(metrics.counts[index]),
            "precision": metrics.precision[index],
            "recall": metrics.recall[index],
            "f1": metrics.f1[index],
        }
        for index, label in enumerate(labels)
    ]
    confusion = [
        {
            "analysis": analysis,
            "task": task,
            "true_class": true_label,
            "predicted_class": predicted_label,
            "count": int(metrics.confusion[i, j]),
            "row_fraction": (
                metrics.confusion[i, j] / metrics.counts[i]
                if metrics.counts[i]
                else float("nan")
            ),
        }
        for i, true_label in enumerate(labels)
        for j, predicted_label in enumerate(labels)
    ]
    event_rows: list[dict[str, Any]] = []
    for row, event in enumerate(events):
        base = {
            "analysis": analysis,
            "task": task,
            "dataset": event.dataset,
            "event_id": event.event_id,
            "true_class": labels[int(truth[row])],
            "predicted_class": labels[int(metrics.predicted_index[row])],
            "correct": int(metrics.predicted_index[row] == truth[row]),
            "top_two_margin": metrics.margin[row],
            "n_inadmissible_scores": int(np.sum(inadmissible[row])),
        }
        for column, label in enumerate(labels):
            base[f"score_{label}"] = scores[row, column]
        event_rows.append(base)
    return per_class, confusion, event_rows, summary


def _run_classification_set(
    analysis: str,
    profiles: Mapping[str, Barycenter],
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    channel: str,
    transform: str | None,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for task_index, (task, labels) in enumerate(TASKS.items()):
        events, traces, truth = _task_event_grid(events_by_dataset, labels, channel)
        if transform == "reversed":
            traces = [trace.reversed() for trace in traces]
        elif transform == "shuffled":
            traces = [
                trace.shuffled(seed + 100_000 * task_index + event.event_id)
                for trace, event in zip(traces, events)
            ]
        task_profiles = {label: profiles[label] for label in labels}
        score_labels, scores = fixed_profile_scores(traces, task_profiles)
        if score_labels != tuple(labels):
            raise RuntimeError("profile score label order changed")
        per_class, confusion, event_score, summary = _classification_rows(
            analysis,
            task,
            events,
            labels,
            truth,
            scores,
            n_bootstrap=n_bootstrap,
            seed=seed + task_index,
        )
        summaries.append(summary)
        per_class_rows.extend(per_class)
        confusion_rows.extend(confusion)
        event_rows.extend(event_score)
    return summaries, per_class_rows, confusion_rows, event_rows


def _run_nuisance_controls(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for task_index, (task, labels) in enumerate(TASKS.items()):
        events, truth, scores = _nuisance_scores(events_by_dataset, labels)
        per_class, confusion, event_score, summary = _classification_rows(
            "control_nuisance_only",
            task,
            events,
            labels,
            truth,
            scores,
            n_bootstrap=n_bootstrap,
            seed=seed + task_index,
        )
        summaries.append(summary)
        per_class_rows.extend(per_class)
        confusion_rows.extend(confusion)
        event_rows.extend(event_score)
    return summaries, per_class_rows, confusion_rows, event_rows


def _pair_rows(
    primary_profiles: Mapping[str, Barycenter],
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    bootstrap_profiles: Mapping[str, Sequence[tuple[Barycenter, Barycenter, float]]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targeted: list[dict[str, Any]] = []
    separations: list[dict[str, Any]] = []
    target_names = {(a, b): name for name, a, b in TARGETED_PAIRS}
    datasets = tuple(MOLECULES)
    for a_index, a in enumerate(datasets):
        for b in datasets[a_index + 1 :]:
            events, traces, truth = _task_event_grid(events_by_dataset, (a, b), "peptide")
            labels, scores = fixed_profile_scores(
                traces, {a: primary_profiles[a], b: primary_profiles[b]}
            )
            metrics = classification_metrics(truth, scores, labels)
            decision = scores[:, 0] - scores[:, 1]
            auroc, auprc = binary_rank_metrics(truth, decision)
            boot = bootstrap_classification(
                truth, scores, labels, n_bootstrap=n_bootstrap, seed=seed + a_index
            )
            balanced_ci = percentile_interval(boot["balanced_accuracy"])

            between = align_profiles(
                primary_profiles[a].mean,
                primary_profiles[b].mean,
                supported_a=primary_profiles[a].supported,
                supported_b=primary_profiles[b].supported,
            ).normalized_l2_dtw
            within_a = np.asarray([item[2] for item in bootstrap_profiles[a]], dtype=float)
            within_b = np.asarray([item[2] for item in bootstrap_profiles[b]], dtype=float)
            base_within_a = float(np.mean(within_a))
            base_within_b = float(np.mean(within_b))
            ratio = separation_ratio(between, base_within_a, base_within_b)
            repeat_count = min(len(bootstrap_profiles[a]), len(bootstrap_profiles[b]))
            boot_ratio = np.empty(repeat_count, dtype=float)
            for repeat in range(repeat_count):
                pa = bootstrap_profiles[a][repeat][0]
                pb = bootstrap_profiles[b][repeat][0]
                boot_between = align_profiles(
                    pa.mean,
                    pb.mean,
                    supported_a=pa.supported,
                    supported_b=pb.supported,
                ).normalized_l2_dtw
                boot_ratio[repeat] = separation_ratio(
                    boot_between,
                    bootstrap_profiles[a][repeat][2],
                    bootstrap_profiles[b][repeat][2],
                )
            ratio_ci = percentile_interval(boot_ratio)
            interval_contains_primary = ratio_ci[0] <= ratio <= ratio_ci[1]
            separations.append(
                {
                    "variant_a": a,
                    "variant_b": b,
                    "between_distance": between,
                    "within_a_distance": base_within_a,
                    "within_b_distance": base_within_b,
                    "separation_ratio": ratio,
                    "separation_ratio_ci_low": ratio_ci[0],
                    "separation_ratio_ci_high": ratio_ci[1],
                    "bootstrap_interval_contains_primary": int(
                        interval_contains_primary
                    ),
                    "n_profile_bootstrap": repeat_count,
                }
            )
            if (a, b) in target_names:
                targeted.append(
                    {
                        "contrast": target_names[(a, b)],
                        "variant_a": a,
                        "variant_b": b,
                        "n_events": len(events),
                        "accuracy": metrics.accuracy,
                        "balanced_accuracy": metrics.balanced_accuracy,
                        "balanced_accuracy_ci_low": balanced_ci[0],
                        "balanced_accuracy_ci_high": balanced_ci[1],
                        "auroc": auroc,
                        "auprc": auprc,
                        "median_margin": float(np.median(metrics.margin)),
                        "separation_ratio": ratio,
                        "separation_ratio_ci_low": ratio_ci[0],
                        "separation_ratio_ci_high": ratio_ci[1],
                        "bootstrap_interval_contains_primary": int(
                            interval_contains_primary
                        ),
                    }
                )
    return targeted, separations


def _published_rows(
    summaries: Sequence[Mapping[str, Any]],
    per_class: Sequence[Mapping[str, Any]],
    targeted: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summary_by_task = {
        (str(row["analysis"]), str(row["task"])): row for row in summaries
    }
    primary_class = [
        row for row in per_class if row["analysis"] == "published_split_peptide"
    ]
    nine_recall = [
        float(row["recall"])
        for row in primary_class
        if row["task"] == "nine_class_bcn"
    ]
    sulf_recall = [
        float(row["recall"])
        for row in primary_class
        if row["task"] == "four_sulfation"
    ]
    chemistry = [
        float(row["balanced_accuracy"])
        for row in targeted
        if row["contrast"] in {"chemistry_site_1", "chemistry_site_2"}
    ]
    peg = summary_by_task[("published_split_peptide", "four_peg8")]
    return [
        {
            "comparison": "Nine-class per-class recall range",
            "published_result": "80–100%",
            "cowler_estimate": f"{100*min(nine_recall):.1f}–{100*max(nine_recall):.1f}%",
            "ci_95": "per-class intervals not yet estimated",
            "absolute_difference": "range comparison",
            "method_notes": "DBA + Gaussian-DTW; see primary_split in run_summary.json",
        },
        {
            "comparison": "Same-site sulfate versus phosphate",
            "published_result": ">90%",
            "cowler_estimate": f"{100*min(chemistry):.1f}–{100*max(chemistry):.1f}% balanced accuracy",
            "ci_95": "see targeted_pair_metrics.csv",
            "absolute_difference": "see pair rows",
            "method_notes": "minimum/maximum over the two fixed-site contrasts",
        },
        {
            "comparison": "Four sulfation-location states",
            "published_result": "95–100%",
            "cowler_estimate": f"{100*min(sulf_recall):.1f}–{100*max(sulf_recall):.1f}% recall",
            "ci_95": "overall interval in classification_summary.csv",
            "absolute_difference": "range comparison",
            "method_notes": "same four constructs; signal-profile calls",
        },
        {
            "comparison": "Four PEG8 sulfation states",
            "published_result": "reported confusion matrix",
            "cowler_estimate": f"{100*float(peg['balanced_accuracy']):.1f}% balanced accuracy",
            "ci_95": f"{100*float(peg['balanced_accuracy_ci_low']):.1f}–{100*float(peg['balanced_accuracy_ci_high']):.1f}%",
            "absolute_difference": "not computable from rounded published summary",
            "method_notes": "full counts in confusion_matrices.csv",
        },
    ]


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    js445_dir = Path(args.js445_dir).resolve()
    representation = str(args.representation)
    if args.split == "random" and args.outlier_filter_scope == "training":
        raise ValueError("--split random requires --outlier-filter-scope upstream")
    # Preserve historical native artifacts when running the new default.
    output_path = args.out_dir
    if output_path is None:
        output_path = (
            DEFAULT_OUT_DIR.with_name(DEFAULT_OUT_DIR.name + "_fixed_100")
            if representation == "fixed_100" else DEFAULT_OUT_DIR
        )
        if args.split == "stored":
            output_path = output_path.with_name(output_path.name + "_stored_split")
    out_dir = Path(output_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dna_profile = predict_DNA_6mer_5_3(TEMPLATE_DNA)
    valid = dna_profile["mean"].notna() & dna_profile["std"].notna()
    expected_dna = dna_profile.loc[valid, "mean"].to_numpy(float)
    target_shift, target_scale = robust_params(expected_dna)

    events_by_dataset: dict[str, list[PreparedEvent]] = {}
    manifest: list[dict[str, Any]] = []
    split_manifest: list[dict[str, Any]] = []
    for dataset in MOLECULES:
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
        manifest.extend(rows)
        split_manifest.extend(
            {
                "dataset": dataset,
                "event_id": event.event_id,
                "analysis": "published_split",
                "replicate": -1,
                "assignment": "stored_isConsensus",
                "group": "training" if event.stored_is_consensus else "held_out",
                "draw_order": -1,
                "active": 0,
            }
            for event in events
        )
        print(
            f"   included {len(events)}/{len(rows)}; training "
            f"{sum(event.stored_is_consensus for event in events)}",
            flush=True,
        )
    write_csv(out_dir / "preprocessing_manifest.csv", manifest)

    included_event_counts = {
        dataset: len(events_by_dataset[dataset]) for dataset in MOLECULES
    }
    training_candidate_counts = {
        dataset: sum(
            event.stored_is_consensus for event in events_by_dataset[dataset]
        )
        for dataset in MOLECULES
    }
    outlier_scope = str(args.outlier_filter_scope)
    active_outlier_threshold = float(args.outlier_z) if args.outlier_filter else None
    print(
        f"2. Calibrating {outlier_scope} distance outliers"
        + (
            f" and removing z >= {active_outlier_threshold:g}..."
            if active_outlier_threshold is not None
            else " without removal (explicit opt-out)..."
        ),
        flush=True,
    )
    events_by_dataset, outlier_rows, outlier_summaries = (
        _apply_outlier_filter(
            events_by_dataset,
            scope=outlier_scope,
            threshold=active_outlier_threshold,
        )
    )
    training_retained_counts = {
        dataset: sum(
            event.stored_is_consensus for event in events_by_dataset[dataset]
        )
        for dataset in MOLECULES
    }
    heldout_candidate_counts = {
        dataset: included_event_counts[dataset] - training_candidate_counts[dataset]
        for dataset in MOLECULES
    }
    heldout_retained_counts = {
        dataset: sum(
            not event.stored_is_consensus for event in events_by_dataset[dataset]
        )
        for dataset in MOLECULES
    }
    split_manifest.extend(
        {
            "dataset": row["dataset"],
            "event_id": row["event_id"],
            "analysis": f"{outlier_scope}_outlier_filter",
            "replicate": -1,
            "assignment": f"{outlier_scope}_cluster_aware_robust_z",
            "group": (
                ("retained_" if int(row["retained"]) else "excluded_")
                + (
                    "training"
                    if int(row["stored_is_consensus"])
                    else "held_out"
                )
            ),
            "draw_order": -1,
            "active": 0,
        }
        for row in outlier_rows
    )
    representation_rows = [
        {
            "dataset": dataset,
            "event_id": event.event_id,
            "representation": representation,
            "source_length": int(event.peptide.mean.size),
            "target_length": 100 if representation == "fixed_100" else int(event.peptide.mean.size),
        }
        for dataset, events in events_by_dataset.items()
        for event in events
    ]
    events_by_dataset = peptide_representation(
        events_by_dataset, representation, min_std=float(args.min_std)
    )
    write_csv(out_dir / "representation_manifest.csv", representation_rows)
    if args.split == "random":
        events_by_dataset, primary_split_rows = stratified_random_split(
            events_by_dataset,
            train_fraction=float(args.train_fraction),
            seed=int(args.split_seed),
        )
        split_strategy = "post_filter_stratified_random"
    else:
        events_by_dataset, primary_split_rows = stored_split(events_by_dataset)
        split_strategy = "stored_isConsensus"
    split_manifest.extend(primary_split_rows)
    primary_training_counts = {
        dataset: sum(event.is_training for event in events)
        for dataset, events in events_by_dataset.items()
    }
    primary_heldout_counts = {
        dataset: sum(not event.is_training for event in events)
        for dataset, events in events_by_dataset.items()
    }
    print(
        f"3. Fitting peptide profiles with {split_strategy}...",
        flush=True,
    )
    primary_profiles, profile_summaries, profile_rows = _fit_profiles(
        events_by_dataset,
        channel="peptide",
        equal_depth=None,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        min_std=float(args.min_std),
    )
    training_counts = [
        sum(event.is_training for event in events_by_dataset[dataset])
        for dataset in MOLECULES
    ]
    equal_depth = min(training_counts)
    for dataset in MOLECULES:
        training = [
            event for event in events_by_dataset[dataset] if event.is_training
        ]
        selected_ids = {
            event.event_id for event in _even_depth(training, equal_depth)
        }
        split_manifest.extend(
            {
                "dataset": dataset,
                "event_id": event.event_id,
                "analysis": "equal_depth",
                "replicate": -1,
                "assignment": "evenly_spaced_event_order",
                "group": (
                    "training"
                    if event.event_id in selected_ids
                    else "unused_training"
                ),
                "draw_order": -1,
            }
            for event in training
        )
    print(f"4. Fitting equal-depth peptide profiles (K={equal_depth})...", flush=True)
    equal_profiles, equal_summaries, equal_rows = _fit_profiles(
        events_by_dataset,
        channel="peptide",
        equal_depth=equal_depth,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        min_std=float(args.min_std),
    )
    profile_summaries.extend(equal_summaries)
    profile_rows.extend(equal_rows)

    print("5. Running blocked split-profile reproducibility...", flush=True)
    reproducibility: list[dict[str, Any]] = []
    bootstrap_reproducibility: list[dict[str, Any]] = []
    bootstrap_profiles: dict[
        str, list[tuple[Barycenter, Barycenter, float]]
    ] = {dataset: [] for dataset in MOLECULES}
    rng = np.random.default_rng(int(args.seed))
    common_group_depth = equal_depth // 2
    for dataset in MOLECULES:
        training = [event for event in events_by_dataset[dataset] if event.is_training]
        for assignment in range(3):
            group_a, group_b, name = blocked_groups(training, assignment)
            for group_name, group in (("A", group_a), ("B", group_b)):
                split_manifest.extend(
                    {
                        "dataset": dataset,
                        "event_id": event.event_id,
                        "analysis": "blocked_split",
                        "replicate": assignment,
                        "assignment": name,
                        "group": group_name,
                        "draw_order": draw_order,
                    }
                    for draw_order, event in enumerate(group)
                )
            profile_a, medoid_a = _fit_profile(
                group_a,
                "peptide",
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                min_std=float(args.min_std),
            )
            profile_b, medoid_b = _fit_profile(
                group_b,
                "peptide",
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                min_std=float(args.min_std),
            )
            reproducibility.append(
                _metric_row(dataset, "blocked_split", name, assignment, profile_a, profile_b)
            )
            profile_summaries.extend(
                [
                    _profile_summary_row(
                        dataset,
                        "blocked_split",
                        profile_a,
                        medoid_a,
                        len(group_a),
                        replicate=assignment,
                        split_group="A",
                    ),
                    _profile_summary_row(
                        dataset,
                        "blocked_split",
                        profile_b,
                        medoid_b,
                        len(group_b),
                        replicate=assignment,
                        split_group="B",
                    ),
                ]
            )
            if assignment == 0:
                profile_rows.extend(
                    _profile_rows(
                        dataset,
                        "blocked_split",
                        profile_a,
                        replicate=assignment,
                        split_group="A",
                    )
                )
                profile_rows.extend(
                    _profile_rows(
                        dataset,
                        "blocked_split",
                        profile_b,
                        replicate=assignment,
                        split_group="B",
                    )
                )
        for repeat in range(int(args.profile_bootstrap)):
            group_a, group_b, name = blocked_groups(training, repeat)
            sample_a = list(
                rng.choice(np.asarray(group_a, dtype=object), common_group_depth, replace=True)
            )
            sample_b = list(
                rng.choice(np.asarray(group_b, dtype=object), common_group_depth, replace=True)
            )
            for group_name, sample in (("A", sample_a), ("B", sample_b)):
                split_manifest.extend(
                    {
                        "dataset": dataset,
                        "event_id": cast(PreparedEvent, event).event_id,
                        "analysis": "whole_event_bootstrap",
                        "replicate": repeat,
                        "assignment": name,
                        "group": group_name,
                        "draw_order": draw_order,
                    }
                    for draw_order, event in enumerate(sample)
                )
            profile_a, medoid_a = _fit_profile(
                cast(list[PreparedEvent], sample_a),
                "peptide",
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                min_std=float(args.min_std),
            )
            profile_b, medoid_b = _fit_profile(
                cast(list[PreparedEvent], sample_b),
                "peptide",
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                min_std=float(args.min_std),
            )
            row = _metric_row(
                dataset, "whole_event_bootstrap", name, repeat, profile_a, profile_b
            )
            row.update(
                {
                    "fixed_group_depth": common_group_depth,
                    "medoid_a_event_id": medoid_a,
                    "medoid_b_event_id": medoid_b,
                }
            )
            bootstrap_reproducibility.append(row)
            bootstrap_profiles[dataset].append(
                (profile_a, profile_b, float(row["normalized_l2_dtw"]))
            )

    print("6. Fitting DNA-window and tail control profiles...", flush=True)
    dna_profiles, dna_summaries, dna_rows = _fit_profiles(
        events_by_dataset,
        channel="dna_window",
        equal_depth=None,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        min_std=float(args.min_std),
    )
    tail_training_counts = [
        sum(
            event.is_training and event.tail is not None
            for event in events_by_dataset[dataset]
        )
        for dataset in MOLECULES
    ]
    tail_equal_depth = min(tail_training_counts)
    tail_profiles, tail_summaries, tail_rows = _fit_profiles(
        events_by_dataset,
        channel="tail",
        equal_depth=tail_equal_depth,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        min_std=float(args.min_std),
    )
    profile_summaries.extend(dna_summaries + tail_summaries)
    profile_rows.extend(dna_rows + tail_rows)

    classification_summaries: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    event_score_rows: list[dict[str, Any]] = []

    analyses = (
        ("published_split_peptide", primary_profiles, "peptide", None),
        ("equal_depth_peptide", equal_profiles, "peptide", None),
        ("control_dna_window", dna_profiles, "dna_window", None),
        ("control_tail_only", tail_profiles, "tail", None),
        ("control_reversed_peptide", primary_profiles, "peptide", "reversed"),
        ("control_shuffled_peptide", primary_profiles, "peptide", "shuffled"),
    )
    for analysis_index, (analysis, profiles, channel, transform) in enumerate(analyses):
        print(f"7. Scoring {analysis}...", flush=True)
        summaries, per_class, confusion, event_scores = _run_classification_set(
            analysis,
            profiles,
            events_by_dataset,
            channel=channel,
            transform=transform,
            n_bootstrap=int(args.classification_bootstrap),
            seed=int(args.seed) + 100 * analysis_index,
        )
        classification_summaries.extend(summaries)
        per_class_rows.extend(per_class)
        confusion_rows.extend(confusion)
        event_score_rows.extend(event_scores)

    print("8. Scoring nuisance-only controls...", flush=True)
    summaries, per_class, confusion, event_scores = _run_nuisance_controls(
        events_by_dataset,
        n_bootstrap=int(args.classification_bootstrap),
        seed=int(args.seed) + 900,
    )
    classification_summaries.extend(summaries)
    per_class_rows.extend(per_class)
    confusion_rows.extend(confusion)
    event_score_rows.extend(event_scores)

    print("9. Computing targeted pairs and separation ratios...", flush=True)
    targeted, separations = _pair_rows(
        primary_profiles,
        events_by_dataset,
        bootstrap_profiles,
        n_bootstrap=int(args.classification_bootstrap),
        seed=int(args.seed) + 1200,
    )
    published = _published_rows(classification_summaries, per_class_rows, targeted)

    method_comparison = [
        {
            "feature": "Input events",
            "published_method": "approximately random 50/50 within pure sample",
            "cowler_method": (
                "cluster-aware robust-z filtering "
                + (
                    f"upstream at >= {active_outlier_threshold:g}"
                    if active_outlier_threshold is not None and outlier_scope == "upstream"
                    else f"on training events at >= {active_outlier_threshold:g}"
                    if active_outlier_threshold is not None
                    else f"disabled after {outlier_scope} calibration"
                )
                + (
                    f", then seeded {100 * float(args.train_fraction):g}/"
                    f"{100 * (1.0 - float(args.train_fraction)):g} split within construct"
                    if args.split == "random"
                    else ", then stored isConsensus training/held-out assignment"
                )
            ),
        },
        {
            "feature": "Preprocessing",
            "published_method": "open-pore normalization + common-DNA calibration",
            "cowler_method": "raw formula + complete-event CPIC + robust common-DNA affine calibration",
        },
        {
            "feature": "Tail rule",
            "published_method": "terminal D15 ramp, 0.6 relative-current rule",
            "cowler_method": f"one global trailing-current rule > {float(args.tail_threshold):g}",
        },
        {
            "feature": "Initialization",
            "published_method": "4–6 manually selected traces",
            "cowler_method": "deterministic minimum-row-sum Gaussian-DTW medoid",
        },
        {
            "feature": "Consensus engine",
            "published_method": "MAP/Baum-Welch profile HMM",
            "cowler_method": "inverse-variance DBA on a medoid axis",
        },
        {
            "feature": "Transition topology",
            "published_method": "published profile-HMM topology",
            "cowler_method": "bounded monotone symmetric1 DTW, max_run=3",
        },
        {
            "feature": "Endpoint handling",
            "published_method": "terminal trimming then profile model",
            "cowler_method": "annotation-assisted, tail-trimmed, closed DTW fragments",
        },
        {
            "feature": "Held-out score",
            "published_method": "forward-backward log likelihood",
            "cowler_method": "path-normalized Gaussian-DTW distance (not a likelihood)",
        },
    ]

    for row in profile_rows:
        is_peptide = not str(row["analysis"]).endswith(("_dna_window", "_tail"))
        row["representation"] = representation if is_peptide else "native"
        if is_peptide and representation == "fixed_100":
            row["dwell"] = ""
            row["relative_position"] = int(row["position"]) / 99.0
    write_csv(out_dir / "profiles.csv", profile_rows)
    write_csv(out_dir / "split_manifest.csv", split_manifest)
    write_csv(out_dir / f"{outlier_scope}_outlier_metrics.csv", outlier_rows)
    write_csv(
        out_dir / f"{outlier_scope}_outlier_summary.csv", outlier_summaries
    )
    write_csv(out_dir / "profile_diagnostics.csv", profile_summaries)
    write_csv(out_dir / "reproducibility_blocked.csv", reproducibility)
    write_csv(out_dir / "reproducibility_bootstrap.csv", bootstrap_reproducibility)
    write_csv(out_dir / "classification_summary.csv", classification_summaries)
    write_csv(out_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(out_dir / "confusion_matrices.csv", confusion_rows)
    write_csv(out_dir / "heldout_event_scores.csv", event_score_rows)
    write_csv(out_dir / "targeted_pair_metrics.csv", targeted)
    write_csv(out_dir / "separation_ratios.csv", separations)
    write_csv(out_dir / "published_comparison.csv", published)
    write_csv(out_dir / "method_comparison.csv", method_comparison)

    primary_summary = {
        str(row["task"]): row
        for row in classification_summaries
        if row["analysis"] == "published_split_peptide"
    }
    control_flags = [
        {
            "analysis": row["analysis"],
            "task": row["task"],
            "balanced_accuracy": row["balanced_accuracy"],
            "chance_accuracy": row["chance_accuracy"],
            "above_chance_ci": bool(
                float(row["balanced_accuracy_ci_low"]) > float(row["chance_accuracy"])
            ),
        }
        for row in classification_summaries
        if str(row["analysis"]).startswith("control_")
    ]
    summary_json = {
        "scope": (
            "same-run test-informed upstream-filter evaluation; not independent validation"
            if outlier_scope == "upstream"
            else "same-run reproduction and exploratory robustness; not independent validation"
        ),
        "signal_output_only": True,
        "dtw_is_likelihood": False,
        "preprocessing": {
            "raw_conversion": "(raw + offset) * rng / digitisation",
            "step_finder": "CPIC on robust-normalized complete event",
            "sensitivity": float(args.sensitivity),
            "min_level_length": int(args.min_level_length),
            "region_policy": "step midpoint inside annotation half-open interval",
            "tail_threshold_source_relative_current": float(args.tail_threshold),
            "tail_control_window_steps": int(args.tail_control_steps),
            "dna_calibration": "event DNA median/MAD mapped to known-template LUT median/MAD",
            "consensus_normalize": False,
            "peptide_representation": representation,
            "target_length": 100 if representation == "fixed_100" else None,
            "length_interpolation": "linear mean and variance, endpoint-inclusive",
            "length_normalization_order": "after native outlier filtering; before active split assignment and all peptide fitting/scoring",
            "peptide_dwell": "omitted" if representation == "fixed_100" else "measured",

        },
        "random_seed": int(args.seed),
        "primary_split": {
            "strategy": split_strategy,
            "train_fraction": (
                float(args.train_fraction) if args.split == "random" else None
            ),
            "seed": int(args.split_seed) if args.split == "random" else None,
            "assignment_field": "is_training",
            "stored_label_field": "stored_is_consensus",
            "stored_label_role": (
                "active training/held-out assignment"
                if args.split == "stored" else "immutable audit provenance only"
            ),
            "manifest_analysis": "primary_split",
            "training_events": primary_training_counts,
            "heldout_events": primary_heldout_counts,
        },
        "outlier_filter": {
            "enabled": active_outlier_threshold is not None,
            "scope": outlier_scope,
            "threshold": active_outlier_threshold,
            "metrics_file": f"{outlier_scope}_outlier_metrics.csv",
            "summary_file": f"{outlier_scope}_outlier_summary.csv",
            "score": "max(cluster-local medoid robust z, cluster-local 5-NN robust z)",
            "calibration_events": (
                "all eligible events before consulting stored isConsensus"
                if outlier_scope == "upstream"
                else "stored isConsensus training events only"
            ),
            "test_informed": outlier_scope == "upstream",
            "heldout_events_filtered": (
                outlier_scope == "upstream" and active_outlier_threshold is not None
            ),
            "training_candidates": training_candidate_counts,
            "training_retained": training_retained_counts,
            "training_excluded": {
                dataset: training_candidate_counts[dataset]
                - training_retained_counts[dataset]
                for dataset in MOLECULES
            },
            "heldout_candidates": heldout_candidate_counts,
            "heldout_retained": heldout_retained_counts,
            "heldout_excluded": {
                dataset: heldout_candidate_counts[dataset]
                - heldout_retained_counts[dataset]
                for dataset in MOLECULES
            },
        },
        "profile_bootstrap_replicates": int(args.profile_bootstrap),
        "classification_bootstrap_replicates": int(args.classification_bootstrap),
        "equal_training_depth": equal_depth,
        "fixed_reproducibility_group_depth": common_group_depth,
        "included_events": {
            dataset: included_event_counts[dataset] for dataset in MOLECULES
        },
        "primary_classification": primary_summary,
        "control_flags": control_flags,
        "interpretation_gate": {
            "provisional_noninferiority_margin_percentage_points": 5,
            "key_pair_balanced_accuracy_gate": 0.90,
            "external_validation": False,
            "hmm_secondary_run": False,
            "hmm_reason": "deferred until convergence and supported-depth gates pass",
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary_json, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(f"10. Wrote machine-readable results to {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--js445-dir", default=DEFAULT_JS445_DIR)
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: representation-specific)")
    parser.add_argument(
        "--representation", choices=("fixed_100", "native"), default="fixed_100",
        help="peptide representation for DBA fitting and scoring (default: fixed_100)",
    )
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--min-level-length", type=int, default=2)
    parser.add_argument("--tail-threshold", type=float, default=0.6)
    parser.add_argument("--tail-control-steps", type=int, default=5)
    parser.add_argument("--dna-window-steps", type=int, default=30)
    parser.add_argument("--min-trace-steps", type=int, default=5)
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--profile-bootstrap", type=int, default=10)
    parser.add_argument("--classification-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--outlier-z",
        "--training-outlier-z",
        dest="outlier_z",
        type=float,
        default=DEFAULT_DBA_UPSTREAM_OUTLIER_Z,
        help="robust distance-z threshold (DBA upstream default: 3.5)",
    )
    parser.add_argument(
        "--outlier-filter-scope",
        choices=("upstream", "training"),
        default="upstream",
        help="events used to calibrate and apply outlier removal (default: upstream)",
    )
    parser.add_argument(
        "--outlier-filter",
        dest="outlier_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply outlier removal after calibration (default: enabled)",
    )
    parser.add_argument(
        "--no-training-outlier-filter",
        dest="outlier_filter",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--split", choices=("stored", "random"), default="stored",
        help="training/held-out assignment (default: stored isConsensus labels)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="training fraction for --split random only (default: 0.5)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="seed for --split random only",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
