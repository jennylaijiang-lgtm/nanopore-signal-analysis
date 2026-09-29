"""Evaluate the eight PSK preprocessing conditions on one frozen stored split.

This runner answers the preprocessing ablation question without changing the
training/held-out assignment between conditions.  For every eligible event,
``stored_is_consensus`` is the master assignment:

* ``True``: training;
* ``False``: held out.

The upstream outlier screen still uses every eligible trace within its labelled
construct, as required by the homogeneous-within-construct assumption.  It never
overwrites the stored assignment.  Native-space screening supplies the ``TO`` and
``TON`` conditions.  The ``TNO`` conditions independently recompute distances,
clusters, and robust z scores after p95 or fixed-100 interpolation.

Eight nine-class DBA pipelines are fitted:

* T: tail trimming only;
* TO: native-space upstream outlier removal;
* TN-p95 and TN-100: length normalization without outlier removal;
* TON-p95 and TON-100: native-space removal, then normalization;
* TNO-p95 and TNO-100: normalization, then normalized-space removal.

Each pipeline is reported on its own retained held-out cohort.  A second paired
analysis uses the intersection of stored held-out events retained by every
outlier screen.  The operational analysis measures the complete selective
pipeline; the common-cohort analysis holds test-event identity fixed.

Run from the repository root:

    uv run python -m scripts.evaluate_psk_preprocessing_ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from cowler.consensus.dba import Barycenter
from cowler.consensus.length_normalize import (
    resample_signal,
    select_percentile_max_target,
)
from cowler.eval.peptide_consensus import (
    DEFAULT_DBA_UPSTREAM_OUTLIER_Z,
    bootstrap_classification,
    fixed_profile_scores,
    percentile_interval,
)
from cowler.io.lut import predict_DNA_6mer_5_3
from cowler.io.normalize import robust_params
from scripts.evaluate_psk_dba_consensus import (
    DEFAULT_DATA_DIR,
    DEFAULT_JS445_DIR,
    TEMPLATE_DNA,
    PreparedEvent,
    SignalTrace,
    _apply_outlier_filter,  # pyright: ignore[reportPrivateUsage]
    _classification_rows,  # pyright: ignore[reportPrivateUsage]
    json_default,
    preprocess_dataset,
    write_csv,
)
from scripts.evaluate_psk_length_normalized_consensus import (
    FittedProfile,
    _diagnostic_row,  # pyright: ignore[reportPrivateUsage]
    _fit_profile,  # pyright: ignore[reportPrivateUsage]
    _profile_rows,  # pyright: ignore[reportPrivateUsage]
)

FloatArr = npt.NDArray[np.float64]
EventKey = tuple[str, int]
BCN_DATASETS = tuple(f"JS{number}" for number in range(445, 454))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "tmp" / "psk_preprocessing_ablation"
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


@dataclass(frozen=True)
class Condition:
    """One complete representation/filtering pipeline."""

    name: str
    order: str
    normalization: str
    outlier_space: str
    target_length: int | None
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]]


def freeze_stored_split(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
) -> dict[str, list[PreparedEvent]]:
    """Restore the immutable ``isConsensus`` master assignment."""
    return {
        dataset: [
            replace(event, is_training=event.stored_is_consensus)
            for event in events
        ]
        for dataset, events in events_by_dataset.items()
    }


def select_p95_target(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    percentile: float,
) -> tuple[int, dict[str, float], tuple[str, ...]]:
    """Select the p95 target from stored training events only."""
    lengths = {
        dataset: [
            int(event.peptide.mean.size)
            for event in events
            if event.stored_is_consensus
        ]
        for dataset, events in events_by_dataset.items()
    }
    result = select_percentile_max_target(lengths, percentile=percentile)
    return (
        result.target_length,
        dict(result.per_dataset_percentile),
        result.determining_datasets,
    )


def length_normalize_events(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    target_length: int,
    min_std: float,
) -> dict[str, list[PreparedEvent]]:
    """Return events whose peptide traces use one derived relative grid."""
    transformed: dict[str, list[PreparedEvent]] = {}
    for dataset, events in events_by_dataset.items():
        rows: list[PreparedEvent] = []
        for event in events:
            result = resample_signal(
                event.peptide.mean,
                event.peptide.std,
                target_length,
                min_std=min_std,
            )
            trace = SignalTrace(
                mean=result.mean,
                std=result.std,
                # Existing length-normalized DBA deliberately omits dwell.  The
                # generic DBA coercer represents an absent dwell channel as ones.
                dwell=np.ones(result.target_length, dtype=float),
            )
            rows.append(replace(event, peptide=trace))
        transformed[dataset] = rows
    return transformed


def event_keys(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    heldout_only: bool = False,
) -> set[EventKey]:
    """Return stable construct/event identities from one condition."""
    return {
        (dataset, event.event_id)
        for dataset, events in events_by_dataset.items()
        for event in events
        if not heldout_only or not event.stored_is_consensus
    }


def common_heldout_keys(conditions: Sequence[Condition]) -> set[EventKey]:
    """Return stored held-out identities present in every condition."""
    if not conditions:
        raise ValueError("at least one condition is required")
    common = event_keys(conditions[0].events_by_dataset, heldout_only=True)
    for condition in conditions[1:]:
        common &= event_keys(condition.events_by_dataset, heldout_only=True)
    if not common:
        raise RuntimeError("preprocessing conditions have no common held-out events")
    return common


def retained_events(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    training: bool,
    selected_keys: set[EventKey] | None = None,
) -> list[PreparedEvent]:
    """Return events in deterministic construct/event order."""
    return [
        event
        for dataset in BCN_DATASETS
        for event in sorted(
            events_by_dataset[dataset],
            key=lambda value: (value.event_order, value.event_id),
        )
        if event.stored_is_consensus is training
        and (selected_keys is None or (dataset, event.event_id) in selected_keys)
    ]


def _screen(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    threshold: float,
    representation: str,
    target_length: int | None,
) -> tuple[dict[str, list[PreparedEvent]], list[dict[str, Any]], list[dict[str, Any]]]:
    filtered, rows, summaries = _apply_outlier_filter(
        events_by_dataset,
        scope="upstream",
        threshold=threshold,
    )
    for row in [*rows, *summaries]:
        row["screening_representation"] = representation
        row["screening_target_length"] = "" if target_length is None else target_length
        row["split_assignment_changed"] = 0
    return freeze_stored_split(filtered), rows, summaries


def build_conditions(
    events_by_dataset: Mapping[str, Sequence[PreparedEvent]],
    *,
    outlier_z: float,
    p95_percentile: float,
    fixed_target: int,
    min_std: float,
) -> tuple[
    list[Condition],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Construct the eight ablation cohorts without changing the master split."""
    native = freeze_stored_split(events_by_dataset)
    p95_before, p95_before_values, p95_before_determining = select_p95_target(
        native, percentile=p95_percentile
    )
    p95_unfiltered = length_normalize_events(
        native, target_length=p95_before, min_std=min_std
    )
    fixed_unfiltered = length_normalize_events(
        native, target_length=fixed_target, min_std=min_std
    )

    native_filtered, native_rows, native_summaries = _screen(
        native,
        threshold=outlier_z,
        representation="native",
        target_length=None,
    )
    p95_after, p95_after_values, p95_after_determining = select_p95_target(
        native_filtered, percentile=p95_percentile
    )
    p95_ton = length_normalize_events(
        native_filtered, target_length=p95_after, min_std=min_std
    )
    fixed_ton = length_normalize_events(
        native_filtered, target_length=fixed_target, min_std=min_std
    )

    p95_tno, p95_rows, p95_summaries = _screen(
        p95_unfiltered,
        threshold=outlier_z,
        representation="p95",
        target_length=p95_before,
    )
    fixed_tno, fixed_rows, fixed_summaries = _screen(
        fixed_unfiltered,
        threshold=outlier_z,
        representation="fixed_100",
        target_length=fixed_target,
    )

    conditions = [
        Condition("T", "T", "none", "none", None, native),
        Condition("TO", "TO", "none", "native", None, native_filtered),
        Condition("TN_p95", "TN", "p95", "none", p95_before, p95_unfiltered),
        Condition(
            "TN_fixed_100", "TN", "fixed_100", "none", fixed_target, fixed_unfiltered
        ),
        Condition("TON_p95", "TON", "p95", "native", p95_after, p95_ton),
        Condition(
            "TON_fixed_100", "TON", "fixed_100", "native", fixed_target, fixed_ton
        ),
        Condition("TNO_p95", "TNO", "p95", "p95", p95_before, p95_tno),
        Condition(
            "TNO_fixed_100",
            "TNO",
            "fixed_100",
            "fixed_100",
            fixed_target,
            fixed_tno,
        ),
    ]
    target_rows: list[dict[str, Any]] = []
    for stage, target, values, determining in (
        ("before_outlier_removal", p95_before, p95_before_values, p95_before_determining),
        ("after_native_outlier_removal", p95_after, p95_after_values, p95_after_determining),
    ):
        target_rows.extend(
            {
                "normalization": "p95",
                "selection_stage": stage,
                "dataset": dataset,
                "percentile": p95_percentile,
                "dataset_percentile_length": value,
                "determines_target": int(dataset in determining),
                "target_length": target,
            }
            for dataset, value in values.items()
        )
    target_rows.append(
        {
            "normalization": "fixed_100",
            "selection_stage": "predeclared",
            "dataset": "ALL",
            "percentile": "",
            "dataset_percentile_length": "",
            "determines_target": 1,
            "target_length": fixed_target,
        }
    )
    return (
        conditions,
        [*native_rows, *p95_rows, *fixed_rows],
        [*native_summaries, *p95_summaries, *fixed_summaries],
        target_rows,
    )


def _trace_fingerprint(
    events: Sequence[PreparedEvent],
    *,
    max_iter: int,
    tol: float,
    min_std: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{max_iter}|{tol:.17g}|{min_std:.17g}".encode())
    for event in events:
        digest.update(f"{event.dataset}|{event.event_id}|".encode())
        digest.update(np.asarray(event.peptide.mean, dtype=np.float64).tobytes())
        digest.update(np.asarray(event.peptide.std, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _load_cached_profile(
    path: Path, *, fingerprint: str
) -> FittedProfile | None:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != fingerprint:
        return None
    with np.load(path, allow_pickle=False) as values:
        profile = Barycenter(
            mean=np.asarray(values["mean"], dtype=float),
            std=np.asarray(values["std"], dtype=float),
            dwell=np.asarray(values["dwell"], dtype=float),
            depth=np.asarray(values["depth"], dtype=np.int64),
            supported=np.asarray(values["supported"], dtype=bool),
            n_iter=int(values["n_iter"]),
            converged=bool(values["converged"]),
            medoid_index=int(values["medoid_index"]),
            objective=np.asarray(values["objective"], dtype=float),
            delta=np.asarray(values["delta"], dtype=float),
        )
    return FittedProfile(
        profile=profile,
        medoid_event_id=int(metadata["medoid_event_id"]),
        n_training=int(metadata["n_training"]),
    )


def _save_cached_profile(
    path: Path, fitted: FittedProfile, *, fingerprint: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = fitted.profile
    np.savez_compressed(
        path,
        mean=profile.mean,
        std=profile.std,
        dwell=profile.dwell,
        depth=profile.depth,
        supported=profile.supported,
        n_iter=np.asarray(profile.n_iter),
        converged=np.asarray(profile.converged),
        medoid_index=np.asarray(profile.medoid_index),
        objective=profile.objective,
        delta=profile.delta,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "medoid_event_id": fitted.medoid_event_id,
                "n_training": fitted.n_training,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def fit_condition_profiles(
    condition: Condition,
    *,
    cache_dir: Path,
    max_iter: int,
    tol: float,
    min_std: float,
) -> tuple[dict[str, FittedProfile], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit or load all nine profiles for one condition."""
    profiles: dict[str, FittedProfile] = {}
    diagnostics: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for dataset in BCN_DATASETS:
        training = [
            event
            for event in condition.events_by_dataset[dataset]
            if event.stored_is_consensus
        ]
        if len(training) < 2:
            raise RuntimeError(f"{condition.name}/{dataset} has fewer than two training events")
        fingerprint = _trace_fingerprint(
            training, max_iter=max_iter, tol=tol, min_std=min_std
        )
        cache_path = cache_dir / condition.name / f"{dataset}.npz"
        fitted = _load_cached_profile(cache_path, fingerprint=fingerprint)
        if fitted is None:
            fitted = _fit_profile(
                [event.peptide for event in training],
                [event.event_id for event in training],
                max_iter=max_iter,
                tol=tol,
                min_std=min_std,
            )
            _save_cached_profile(cache_path, fitted, fingerprint=fingerprint)
        profiles[dataset] = fitted
        source_lengths = [event.raw_peptide_steps for event in training]
        diagnostic = _diagnostic_row(
            condition.name,
            dataset,
            fitted,
            source_lengths=source_lengths,
        )
        diagnostic.update(
            {
                "condition": condition.name,
                "order": condition.order,
                "normalization": condition.normalization,
                "outlier_space": condition.outlier_space,
                "target_length": (
                    "" if condition.target_length is None else condition.target_length
                ),
            }
        )
        diagnostics.append(diagnostic)
        rows = _profile_rows(condition.name, dataset, fitted)
        for row in rows:
            row.update(
                {
                    "condition": condition.name,
                    "order": condition.order,
                    "normalization": condition.normalization,
                    "outlier_space": condition.outlier_space,
                }
            )
        profile_rows.extend(rows)
    failed = [
        dataset
        for dataset, fitted in profiles.items()
        if not fitted.profile.converged or np.mean(fitted.profile.supported) < 0.8
    ]
    if failed:
        raise RuntimeError(
            f"{condition.name} profiles failed convergence/support: " + ", ".join(failed)
        )
    return profiles, diagnostics, profile_rows


def score_condition(
    condition: Condition,
    profiles: Mapping[str, FittedProfile],
    *,
    selected_keys: set[EventKey] | None,
    analysis: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[
    list[PreparedEvent],
    npt.NDArray[np.int64],
    FloatArr,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Score one condition on its own or a supplied common held-out cohort."""
    events = retained_events(
        condition.events_by_dataset,
        training=False,
        selected_keys=selected_keys,
    )
    if not events:
        raise RuntimeError(f"{condition.name} has no held-out events to score")
    truth = np.asarray(
        [BCN_DATASETS.index(event.dataset) for event in events], dtype=np.int64
    )
    labels, scores = fixed_profile_scores(
        [event.peptide for event in events],
        {dataset: profiles[dataset].profile for dataset in BCN_DATASETS},
        max_run=3,
    )
    if labels != BCN_DATASETS:
        raise RuntimeError(f"{condition.name} score-label order changed")
    per_class, confusion, event_rows, summary = _classification_rows(
        analysis,
        "nine_class_bcn",
        events,
        BCN_DATASETS,
        truth,
        scores,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    for row in [summary, *per_class, *confusion, *event_rows]:
        row.update(
            {
                "condition": condition.name,
                "order": condition.order,
                "normalization": condition.normalization,
                "outlier_space": condition.outlier_space,
                "target_length": (
                    "" if condition.target_length is None else condition.target_length
                ),
            }
        )
    for row, event in zip(event_rows, events):
        row["stored_is_consensus"] = int(event.stored_is_consensus)
        row["scoring_length"] = int(event.peptide.mean.size)
    return events, truth, np.asarray(scores, dtype=float), summary, per_class, confusion, event_rows


def condition_manifest(
    source_events: Mapping[str, Sequence[PreparedEvent]],
    conditions: Sequence[Condition],
) -> list[dict[str, Any]]:
    """Record retention and the unchanged stored assignment in every condition."""
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        retained = event_keys(condition.events_by_dataset)
        lengths = {
            (dataset, event.event_id): int(event.peptide.mean.size)
            for dataset, events in condition.events_by_dataset.items()
            for event in events
        }
        for dataset in BCN_DATASETS:
            for event in source_events[dataset]:
                key = (dataset, event.event_id)
                rows.append(
                    {
                        "condition": condition.name,
                        "order": condition.order,
                        "normalization": condition.normalization,
                        "outlier_space": condition.outlier_space,
                        "target_length": (
                            "" if condition.target_length is None else condition.target_length
                        ),
                        "dataset": dataset,
                        "event_id": event.event_id,
                        "stored_is_consensus": int(event.stored_is_consensus),
                        "master_group": (
                            "training" if event.stored_is_consensus else "held_out"
                        ),
                        "retained": int(key in retained),
                        "excluded": int(key not in retained),
                        "source_length": int(event.peptide.mean.size),
                        "condition_length": lengths.get(key, ""),
                        "assignment_overridden": 0,
                    }
                )
    return rows


def outlier_overlap_rows(
    outlier_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare identities rejected in native, p95, and fixed-100 spaces."""
    rejected: dict[str, set[EventKey]] = {}
    for row in outlier_rows:
        representation = str(row["screening_representation"])
        if int(row["excluded"]):
            rejected.setdefault(representation, set()).add(
                (str(row["dataset"]), int(row["event_id"]))
            )
        else:
            rejected.setdefault(representation, set())
    output: list[dict[str, Any]] = []
    representations = ("native", "p95", "fixed_100")
    for index, first in enumerate(representations):
        for second in representations[index + 1 :]:
            intersection = rejected[first] & rejected[second]
            union = rejected[first] | rejected[second]
            output.append(
                {
                    "representation_a": first,
                    "representation_b": second,
                    "n_rejected_a": len(rejected[first]),
                    "n_rejected_b": len(rejected[second]),
                    "n_rejected_both": len(intersection),
                    "n_rejected_either": len(union),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                }
            )
    return output


def paired_difference_rows(
    truth: npt.NDArray[np.int64],
    scores_by_condition: Mapping[str, FloatArr],
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return predeclared paired contrasts on the common held-out cohort."""
    comparisons = (
        ("TO", "T", "outlier removal on native traces"),
        ("TN_p95", "T", "p95 normalization without outlier removal"),
        ("TN_fixed_100", "T", "fixed-100 normalization without outlier removal"),
        ("TON_p95", "TO", "p95 normalization after native outlier removal"),
        ("TON_fixed_100", "TO", "fixed-100 normalization after native outlier removal"),
        ("TNO_p95", "TON_p95", "p95 order sensitivity"),
        ("TNO_fixed_100", "TON_fixed_100", "fixed-100 order sensitivity"),
    )
    boot = {
        condition: bootstrap_classification(
            truth,
            scores,
            BCN_DATASETS,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for condition, scores in scores_by_condition.items()
    }
    rows: list[dict[str, Any]] = []
    for first, second, interpretation in comparisons:
        for metric in METRICS:
            differences = boot[first][metric] - boot[second][metric]
            low, high = percentile_interval(differences)
            rows.append(
                {
                    "condition_a": first,
                    "condition_b": second,
                    "contrast": interpretation,
                    "metric": metric,
                    "difference_definition": "condition_a - condition_b",
                    "estimate": float(summaries[first][metric])
                    - float(summaries[second][metric]),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": n_bootstrap,
                    "bootstrap_seed": seed,
                }
            )
    return rows


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dna_profile = predict_DNA_6mer_5_3(TEMPLATE_DNA)
    valid = dna_profile["mean"].notna() & dna_profile["std"].notna()
    target_shift, target_scale = robust_params(
        dna_profile.loc[valid, "mean"].to_numpy(float)
    )

    events_by_dataset: dict[str, list[PreparedEvent]] = {}
    preprocessing_rows: list[dict[str, Any]] = []
    for dataset in BCN_DATASETS:
        print(f"1. Preprocessing {dataset}...", flush=True)
        events, rows = preprocess_dataset(
            dataset,
            data_dir=Path(args.data_dir).resolve(),
            js445_dir=Path(args.js445_dir).resolve(),
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
    events_by_dataset = freeze_stored_split(events_by_dataset)
    write_csv(out_dir / "preprocessing_manifest.csv", preprocessing_rows)

    print("2. Building native and normalized upstream screens...", flush=True)
    conditions, outlier_rows, outlier_summaries, target_rows = build_conditions(
        events_by_dataset,
        outlier_z=float(args.outlier_z),
        p95_percentile=float(args.p95_percentile),
        fixed_target=int(args.fixed_target),
        min_std=float(args.min_std),
    )
    write_csv(out_dir / "outlier_metrics.csv", outlier_rows)
    write_csv(out_dir / "outlier_summary.csv", outlier_summaries)
    write_csv(out_dir / "outlier_overlap.csv", outlier_overlap_rows(outlier_rows))
    write_csv(out_dir / "target_length_selection.csv", target_rows)
    manifest_rows = condition_manifest(events_by_dataset, conditions)
    write_csv(out_dir / "condition_manifest.csv", manifest_rows)

    common_keys = common_heldout_keys(conditions)
    write_csv(
        out_dir / "common_heldout_manifest.csv",
        [
            {
                "dataset": dataset,
                "event_id": event_id,
                "master_group": "held_out",
                "retained_by_all_conditions": 1,
            }
            for dataset, event_id in sorted(common_keys)
        ],
    )

    profiles_by_condition: dict[str, dict[str, FittedProfile]] = {}
    profile_diagnostics: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    operational_summaries: list[dict[str, Any]] = []
    operational_per_class: list[dict[str, Any]] = []
    operational_confusion: list[dict[str, Any]] = []
    operational_events: list[dict[str, Any]] = []
    common_summaries: list[dict[str, Any]] = []
    common_per_class: list[dict[str, Any]] = []
    common_confusion: list[dict[str, Any]] = []
    common_events: list[dict[str, Any]] = []
    common_scores: dict[str, FloatArr] = {}
    common_truth: npt.NDArray[np.int64] | None = None
    common_event_order: list[EventKey] | None = None

    for index, condition in enumerate(conditions, start=1):
        print(
            f"3. [{index}/{len(conditions)}] Fitting and scoring {condition.name}...",
            flush=True,
        )
        profiles, diagnostics, rows = fit_condition_profiles(
            condition,
            cache_dir=out_dir / "profile_cache",
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            min_std=float(args.min_std),
        )
        profiles_by_condition[condition.name] = profiles
        profile_diagnostics.extend(diagnostics)
        profile_rows.extend(rows)

        operational = score_condition(
            condition,
            profiles,
            selected_keys=None,
            analysis="operational_retained_cohort",
            n_bootstrap=int(args.classification_bootstrap),
            seed=int(args.bootstrap_seed),
        )
        operational_summaries.append(operational[3])
        operational_per_class.extend(operational[4])
        operational_confusion.extend(operational[5])
        operational_events.extend(operational[6])

        common = score_condition(
            condition,
            profiles,
            selected_keys=common_keys,
            analysis="paired_common_heldout_cohort",
            n_bootstrap=int(args.classification_bootstrap),
            seed=int(args.bootstrap_seed),
        )
        keys = [(event.dataset, event.event_id) for event in common[0]]
        if common_event_order is None:
            common_event_order = keys
            common_truth = common[1]
        else:
            assert common_truth is not None
            if keys != common_event_order or not np.array_equal(common[1], common_truth):
                raise RuntimeError("common held-out event order differs between conditions")
        common_scores[condition.name] = common[2]
        common_summaries.append(common[3])
        common_per_class.extend(common[4])
        common_confusion.extend(common[5])
        common_events.extend(common[6])

    write_csv(out_dir / "profile_diagnostics.csv", profile_diagnostics)
    write_csv(out_dir / "profiles.csv", profile_rows)
    write_csv(out_dir / "operational_classification_summary.csv", operational_summaries)
    write_csv(out_dir / "operational_per_class.csv", operational_per_class)
    write_csv(out_dir / "operational_confusion_matrix.csv", operational_confusion)
    write_csv(out_dir / "operational_heldout_event_scores.csv", operational_events)
    write_csv(out_dir / "common_classification_summary.csv", common_summaries)
    write_csv(out_dir / "common_per_class.csv", common_per_class)
    write_csv(out_dir / "common_confusion_matrix.csv", common_confusion)
    write_csv(out_dir / "common_heldout_event_scores.csv", common_events)

    if common_truth is None:
        raise RuntimeError("common held-out scoring did not run")
    common_summary_by_condition = {
        str(row["condition"]): row for row in common_summaries
    }
    paired_rows = paired_difference_rows(
        common_truth,
        common_scores,
        common_summary_by_condition,
        n_bootstrap=int(args.classification_bootstrap),
        seed=int(args.bootstrap_seed),
    )
    write_csv(out_dir / "paired_classification_differences.csv", paired_rows)

    counts = {
        condition.name: {
            "training": len(
                retained_events(condition.events_by_dataset, training=True)
            ),
            "held_out": len(
                retained_events(condition.events_by_dataset, training=False)
            ),
        }
        for condition in conditions
    }
    summary = {
        "scope": "eight-condition DBA preprocessing ablation",
        "master_split": {
            "training": "stored isConsensus=True",
            "held_out": "stored isConsensus=False",
            "assignment_overridden": False,
        },
        "tail_threshold": float(args.tail_threshold),
        "outlier_filter": {
            "scope": "upstream within each labelled construct",
            "threshold": float(args.outlier_z),
            "homogeneity_assumption": "one genuine signal species per construct",
            "test_informed": True,
        },
        "targets": target_rows,
        "condition_counts": counts,
        "common_heldout_events": len(common_keys),
        "operational_classification": operational_summaries,
        "common_classification": common_summaries,
        "paired_differences": paired_rows,
        "outputs": {
            "operational": "each condition's own retained stored-held-out cohort",
            "paired": "intersection of stored-held-out events retained by every screen",
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(f"4. Wrote preprocessing ablation to {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--js445-dir", default=DEFAULT_JS445_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--min-level-length", type=int, default=2)
    parser.add_argument("--tail-threshold", type=float, default=0.6)
    parser.add_argument("--tail-control-steps", type=int, default=5)
    parser.add_argument("--dna-window-steps", type=int, default=30)
    parser.add_argument("--min-trace-steps", type=int, default=5)
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--outlier-z", type=float, default=DEFAULT_DBA_UPSTREAM_OUTLIER_Z)
    parser.add_argument("--p95-percentile", type=float, default=95.0)
    parser.add_argument("--fixed-target", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--classification-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
