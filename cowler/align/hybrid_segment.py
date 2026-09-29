"""Known-profile segmentation of hybrid DNA/non-DNA nanopore reads.

The decoder has three semantic regions: ``pre_dna``, ``dna``, and ``non_dna``.
The pre-DNA signal likely arises from threading-tail--pore interactions before
the DNA reaches the constriction. The decoder does not define or infer any
intermediate or molecule-specific downstream region.

The DNA region is the selected known-profile state chain itself. There are no
generic DNA prefix or suffix states: accepted DNA coordinates are exactly the
observed interval assigned to that chain. Separate ``pre_dna`` and ``non_dna``
models describe signal before the DNA and after the DNA-to-non-DNA boundary.

Caller contract
---------------
Callers must currently provide:

* one already isolated event as segmented ``Step`` observations with raw-sample
  bounds;
* the unfiltered known-DNA profile from ``predict_DNA_6mer_5_3``; and
* calibrated ``pre_dna``, ``dna``, and ``non_dna`` emission models.

Step levels, profile levels, and region models must share one current scale.
This module does not isolate events, segment raw signal, or fit region models.

The implemented topology is exactly
``PRE_DNA -> DNA[h] -> NON_DNA``. Only the ``dna_to_non_dna`` orientation is
implemented. Anchor selection remains coordinate-aware at either profile edge
so the reverse topology can be added without changing reference bookkeeping.

Setting ``dna_only=True`` (or calling ``segment_dna_only``) drops the terminal
state, decoding ``PRE_DNA -> DNA[h]`` with a free end for events that contain no
non-DNA region.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Literal, Protocol, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .dp import emission
from .segment import Step
from .transition import build_transition

NEG_INF = -np.inf
_LOG_2PI = float(np.log(2.0 * np.pi))
DEFAULT_BOUNDARY_TRIM_STEPS = 4

BoundaryEdge = Literal["profile_start", "profile_end"]
Orientation = Literal["dna_to_non_dna", "non_dna_to_dna"]


class LevelEmissionModel(Protocol):
    """Interface for one semantic region's step-level emission model."""

    @property
    def label(self) -> str:
        """Semantic label represented by this model."""
        ...

    def log_prob(
        self,
        mean: npt.NDArray[np.float64],
        std: npt.NDArray[np.float64],
        dwell: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return one log-emission per observed step."""
        ...


@dataclass(frozen=True)
class GaussianLevelModel:
    """Simple control-fitted level model for one semantic signal region.

    ``mean`` and ``std`` describe the region's level distribution. Observed step
    uncertainty is convolved with the model variance, matching ``align.dp.emission``.
    Dwell is accepted for interface compatibility but is not modelled by this MVP.
    """

    mean: float
    std: float
    label: str = "non_dna"

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("GaussianLevelModel.mean must be finite")
        if not np.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("GaussianLevelModel.std must be finite and > 0")
        if not self.label:
            raise ValueError("GaussianLevelModel.label must be non-empty")

    def log_prob(
        self,
        mean: npt.NDArray[np.float64],
        std: npt.NDArray[np.float64],
        dwell: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        del dwell  # reserved for later dwell-aware region models
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
        variance = std**2 + self.std**2
        return -0.5 * (
            _LOG_2PI + np.log(variance) + (mean - self.mean) ** 2 / variance
        )


@dataclass(frozen=True)
class HybridRegionModels:
    """Calibrated emissions for the pre-DNA, DNA, and non-DNA regions."""

    pre_dna: LevelEmissionModel
    dna: LevelEmissionModel
    non_dna: LevelEmissionModel

    def __post_init__(self) -> None:
        if self.pre_dna.label != "pre_dna":
            raise ValueError("the pre-DNA emission model must have label 'pre_dna'")
        if self.dna.label != "dna":
            raise ValueError("the DNA emission model must have label 'dna'")
        if self.non_dna.label != "non_dna":
            raise ValueError("the non-DNA emission model must have label 'non_dna'")


@dataclass(frozen=True)
class HybridSegmentParams:
    """Reference selection, chemistry, topology, and rejection parameters."""

    orientation: Orientation = "dna_to_non_dna"
    dna_only: bool = False
    boundary_edge: BoundaryEdge = "profile_end"
    boundary_trim_steps: int = DEFAULT_BOUNDARY_TRIM_STEPS
    anchor_fraction: float = 1.0
    min_anchor_coverage: float = 0.75
    min_anchor_observed_fraction: float = 0.50
    anchor_end_tolerance_states: int = 1
    min_non_dna_steps: int = 5
    min_score_margin: float = 0.0
    min_anchor_emission_margin: float = 0.0
    log_anchor_entry_prior: float = 0.0
    log_anchor_exit_prior: float = 0.0
    log_boundary_prior: float = 0.0
    pstep: float = 0.9
    phold: float = 0.05
    pmiss: float = 0.1
    kmax: int = 10

    def __post_init__(self) -> None:
        if self.orientation not in ("dna_to_non_dna", "non_dna_to_dna"):
            raise ValueError(f"unsupported orientation: {self.orientation!r}")
        if self.boundary_edge not in ("profile_start", "profile_end"):
            raise ValueError(f"unsupported boundary edge: {self.boundary_edge!r}")
        if self.boundary_trim_steps < 0:
            raise ValueError("boundary_trim_steps must be >= 0")
        if self.anchor_fraction != 1.0:
            raise ValueError(
                "the current three-region topology requires anchor_fraction == 1.0"
            )
        if not 0.0 < self.min_anchor_coverage <= 1.0:
            raise ValueError("min_anchor_coverage must be in (0, 1]")
        if not 0.0 < self.min_anchor_observed_fraction <= 1.0:
            raise ValueError("min_anchor_observed_fraction must be in (0, 1]")
        if self.anchor_end_tolerance_states < 0:
            raise ValueError("anchor_end_tolerance_states must be >= 0")
        if self.min_non_dna_steps < 1:
            raise ValueError("min_non_dna_steps must be >= 1")
        if self.kmax < 1:
            raise ValueError("kmax must be >= 1")


@dataclass(frozen=True)
class TrustedAnchor:
    """Selected DNA profile states with explicit original-reference coordinates."""

    table: pd.DataFrame
    source_length_bases: int
    boundary_edge: BoundaryEdge
    trimmed_profile_rows: tuple[int, ...]
    anchor_profile_rows: tuple[int, ...]
    anchor_ref_positions: tuple[int, ...]

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        return self.table["mean"].to_numpy(dtype=float)

    @property
    def std(self) -> npt.NDArray[np.float64]:
        return self.table["std"].to_numpy(dtype=float)

    @property
    def ref_pos(self) -> npt.NDArray[np.int64]:
        return self.table["ref_pos_1based"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.table)


@dataclass(frozen=True)
class HybridSegmentResult:
    """MAP DNA/non-DNA segmentation; all step intervals are half-open."""

    status: Literal["pass", "rejected"]
    step_labels: npt.NDArray[np.str_]
    dna_path: npt.NDArray[np.int64]
    dna_start_step: int
    dna_end_step: int
    dna_start_sample: int
    dna_end_sample: int
    boundary_step: int
    boundary_sample: int
    anchor_start_step: int
    anchor_end_step: int
    anchor_start_sample: int
    anchor_end_sample: int
    anchor_ref_start: int
    anchor_ref_end: int
    anchor_ref_coverage: float
    anchor_observed_fraction: float
    orientation: Orientation
    total_score: float
    null_score: float
    score_margin: float
    anchor_emission_margin: float
    anchor_confidence: float | None = None
    boundary_confidence: float | None = None
    state_posteriors: npt.NDArray[np.float64] | None = None


@dataclass(frozen=True)
class _ObservedArrays:
    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    dwell: npt.NDArray[np.float64]
    start_sample: npt.NDArray[np.int64]
    end_sample: npt.NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.mean)


@dataclass(frozen=True)
class _DecodedPath:
    state_path: npt.NDArray[np.int64]
    total_score: float
    null_score: float
    score_margin: float
    pre_dna_state: int
    dna_start_state: int
    dna_stop_state: int
    non_dna_state: int


def select_trusted_anchor(
    full_dna_profile: pd.DataFrame,
    *,
    boundary_edge: BoundaryEdge = "profile_end",
    boundary_trim_steps: int = DEFAULT_BOUNDARY_TRIM_STEPS,
) -> TrustedAnchor:
    """Select coordinate-aware DNA states near the expected region boundary.

    ``full_dna_profile`` must be the unfiltered output of
    ``predict_DNA_6mer_5_3``. Original one-based input coordinates are recovered
    before invalid boundary rows are dropped:

    ``ref_pos_1based = sequence_length - profile_row // 2``.

    The eligible source profile first retains only finite states belonging to a
    complete pre/post pair. ``boundary_trim_steps`` then removes that many expected
    half-step states from the selected profile edge. Odd values deliberately allow
    a single pre/post phase to be removed because the JS445 sweep tests 0--9 states,
    not only whole nucleotides. All remaining expected states are retained. The
    current topology has no generic DNA prefix state that could absorb DNA omitted
    by fractional profile selection.

    The helper is shared by ``segment_dna_non_dna`` and evaluation/window-sweep
    code. Call it directly when the exact retained/trimmed source-profile rows
    or reference coordinates are needed; ordinary segmentation callers can let
    ``segment_dna_non_dna`` call it internally.

    ``boundary_trim_steps`` is tunable; four is the shipped default, selected by a
    JS445 within-run sweep on the decoder's own uncalibrated boundary (trim 4 gave
    the best exact/within-one boundary accuracy). It is an empirical same-run value,
    not an independently validated chemistry constant, and remains pending
    independent-run confirmation. See ``docs/dna-non-dna-segmentation.md``.
    """

    required = {"mean", "std", "mode"}
    missing = required.difference(full_dna_profile.columns)
    if missing:
        raise ValueError(f"DNA profile is missing columns: {sorted(missing)}")
    if boundary_edge not in ("profile_start", "profile_end"):
        raise ValueError(f"unsupported boundary edge: {boundary_edge!r}")
    if boundary_trim_steps < 0:
        raise ValueError("boundary_trim_steps must be >= 0")
    if len(full_dna_profile) == 0 or len(full_dna_profile) % 2:
        raise ValueError("full DNA profile must contain two rows per input base")

    source_length = len(full_dna_profile) // 2
    profile = full_dna_profile.copy()
    profile["profile_row"] = np.arange(len(profile), dtype=np.int64)
    profile["ref_pos_1based"] = (
        source_length - profile["profile_row"].to_numpy(dtype=np.int64) // 2
    )

    finite = np.isfinite(profile["mean"].to_numpy(dtype=float)) & np.isfinite(
        profile["std"].to_numpy(dtype=float)
    )
    valid = profile.loc[finite].copy()
    if valid.empty:
        raise ValueError("DNA profile has no finite expected levels")

    ordered_positions = list(dict.fromkeys(valid["ref_pos_1based"].astype(int)))
    complete_positions: list[int] = []
    for pos in ordered_positions:
        rows = valid.loc[valid["ref_pos_1based"] == pos]
        if len(rows) == 2 and set(rows["mode"].astype(str)) == {"pre", "post"}:
            complete_positions.append(pos)

    complete = valid.loc[valid["ref_pos_1based"].isin(complete_positions)].copy()
    if len(complete) <= boundary_trim_steps:
        raise ValueError(
            "DNA profile is too short after requiring complete pre/post pairs "
            f"and trimming {boundary_trim_steps} boundary-proximal steps"
    )

    if boundary_edge == "profile_end":
        trimmed = (
            complete.iloc[-boundary_trim_steps:]
            if boundary_trim_steps
            else complete.iloc[0:0]
        )
        remaining = (
            complete.iloc[:-boundary_trim_steps]
            if boundary_trim_steps
            else complete
        )
        anchor = remaining.copy()
    else:
        trimmed = complete.iloc[:boundary_trim_steps]
        remaining = complete.iloc[boundary_trim_steps:]
        anchor = remaining.copy()

    anchor_positions = list(dict.fromkeys(anchor["ref_pos_1based"].astype(int)))
    anchor["anchor_state"] = np.arange(len(anchor), dtype=np.int64)
    anchor.reset_index(drop=True, inplace=True)

    return TrustedAnchor(
        table=anchor,
        source_length_bases=source_length,
        boundary_edge=boundary_edge,
        trimmed_profile_rows=tuple(int(x) for x in trimmed["profile_row"]),
        anchor_profile_rows=tuple(int(x) for x in anchor["profile_row"]),
        anchor_ref_positions=tuple(int(x) for x in anchor_positions),
    )


def _coerce_observed_steps(steps: Sequence[Step] | object) -> _ObservedArrays:
    """Accept ``list[Step]`` or a StepRead-like struct-of-arrays object."""

    if hasattr(steps, "mean") and not isinstance(steps, (list, tuple)):
        mean = np.asarray(getattr(steps, "mean"), dtype=float)
        std = np.asarray(getattr(steps, "std"), dtype=float)
        dwell = np.asarray(getattr(steps, "dwell"), dtype=float)
        start_sample = np.asarray(getattr(steps, "start_sample"), dtype=np.int64)
        end_sample = np.asarray(getattr(steps, "end_sample"), dtype=np.int64)
    else:
        seq = list(steps)  # type: ignore[arg-type]
        mean = np.asarray([step.mean for step in seq], dtype=float)
        std = np.asarray([step.std for step in seq], dtype=float)
        dwell = np.asarray([step.dwell for step in seq], dtype=float)
        start_sample = np.asarray([step.start_sample for step in seq], dtype=np.int64)
        end_sample = np.asarray([step.end_sample for step in seq], dtype=np.int64)

    arrays = (mean, std, dwell, start_sample, end_sample)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("observed step fields must be one-dimensional")
    if len(mean) == 0:
        raise ValueError("at least one observed step is required")
    if any(len(array) != len(mean) for array in arrays[1:]):
        raise ValueError("observed step fields must have the same length")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("observed step mean/std values must be finite")
    if np.any(std < 0.0):
        raise ValueError("observed step std values must be >= 0")
    if np.any(end_sample <= start_sample):
        raise ValueError("each observed step must have a positive sample span")
    if np.any(start_sample[1:] < end_sample[:-1]):
        raise ValueError("observed step sample spans must not overlap")

    return _ObservedArrays(mean, std, dwell, start_sample, end_sample)


def _validated_model_scores(
    model: LevelEmissionModel,
    observed: _ObservedArrays,
) -> npt.NDArray[np.float64]:
    score = np.asarray(
        model.log_prob(observed.mean, observed.std, observed.dwell), dtype=float
    )
    if score.shape != observed.mean.shape:
        raise ValueError(
            f"emission model {model.label!r} returned shape {score.shape}; "
            f"expected {observed.mean.shape}"
        )
    if not np.all(np.isfinite(score)):
        raise ValueError(f"emission model {model.label!r} returned non-finite scores")
    return score


def _viterbi_dna_to_non_dna(
    dna_profile_emission: npt.NDArray[np.float64],
    log_dna_transition: npt.NDArray[np.float64],
    pre_dna_emission: npt.NDArray[np.float64],
    dna_emission: npt.NDArray[np.float64],
    non_dna_emission: npt.NDArray[np.float64],
    params: HybridSegmentParams,
) -> _DecodedPath:
    """Decode ``pre-DNA -> known-profile DNA -> non-DNA``."""

    n_steps, n_dna = dna_profile_emission.shape
    if log_dna_transition.shape != (n_dna, n_dna):
        raise ValueError("DNA transition matrix shape does not match DNA profile")
    if any(
        len(score) != n_steps
        for score in (pre_dna_emission, dna_emission, non_dna_emission)
    ):
        raise ValueError("all region emissions must match the observed step count")

    required_span = ceil(params.min_anchor_coverage * n_dna)
    min_exit = max(0, n_dna - 1 - params.anchor_end_tolerance_states)
    max_entry = min_exit - required_span + 1
    if max_entry < 0:
        raise ValueError(
            "DNA profile is too short for min_anchor_coverage and "
            "anchor_end_tolerance_states"
        )
    entry_states = np.arange(max_entry + 1, dtype=np.int64)
    exit_states = np.arange(min_exit, n_dna, dtype=np.int64)

    pre_dna_state = 0
    dna_start_state = 1
    dna_stop_state = dna_start_state + n_dna
    non_dna_state = dna_stop_state
    n_states = non_dna_state + 1

    emit = np.full((n_steps, n_states), NEG_INF)
    emit[:, pre_dna_state] = pre_dna_emission
    emit[:, dna_start_state:dna_stop_state] = dna_profile_emission
    emit[:, non_dna_state] = non_dna_emission

    transition = np.full((n_states, n_states), NEG_INF)
    transition[pre_dna_state, pre_dna_state] = 0.0
    entry_logp = params.log_anchor_entry_prior - float(np.log(len(entry_states)))
    dna_entry_columns = dna_start_state + entry_states
    transition[pre_dna_state, dna_entry_columns] = entry_logp
    transition[
        dna_start_state:dna_stop_state,
        dna_start_state:dna_stop_state,
    ] = log_dna_transition

    dna_exit_rows = dna_start_state + exit_states
    transition[dna_exit_rows, non_dna_state] = (
        params.log_anchor_exit_prior + params.log_boundary_prior
    )
    transition[non_dna_state, non_dna_state] = 0.0

    value = np.full((n_steps, n_states), NEG_INF)
    backptr = np.full((n_steps, n_states), -1, dtype=np.int64)
    value[0, pre_dna_state] = emit[0, pre_dna_state]
    value[0, dna_entry_columns] = entry_logp + emit[0, dna_entry_columns]

    for t in range(1, n_steps):
        candidates = value[t - 1][:, None] + transition
        if n_steps - t < params.min_non_dna_steps:
            candidates[:, non_dna_state] = NEG_INF
            candidates[non_dna_state, non_dna_state] = value[
                t - 1, non_dna_state
            ]
        backptr[t] = np.argmax(candidates, axis=0)
        best = candidates[backptr[t], np.arange(n_states)]
        value[t] = best + emit[t]

    total_score = float(value[-1, non_dna_state])
    state_path = np.full(n_steps, -1, dtype=np.int64)
    if np.isfinite(total_score):
        state_path[-1] = non_dna_state
        for t in range(n_steps - 1, 0, -1):
            state_path[t - 1] = backptr[t, state_path[t]]

    null_score = float(
        max(
            np.sum(pre_dna_emission),
            np.sum(dna_emission),
            np.sum(non_dna_emission),
        )
    )
    return _DecodedPath(
        state_path=state_path,
        total_score=total_score,
        null_score=null_score,
        score_margin=total_score - null_score,
        pre_dna_state=pre_dna_state,
        dna_start_state=dna_start_state,
        dna_stop_state=dna_stop_state,
        non_dna_state=non_dna_state,
    )


def _viterbi_dna_only(
    dna_profile_emission: npt.NDArray[np.float64],
    log_dna_transition: npt.NDArray[np.float64],
    pre_dna_emission: npt.NDArray[np.float64],
    dna_emission: npt.NDArray[np.float64],
    non_dna_emission: npt.NDArray[np.float64],
    params: HybridSegmentParams,
) -> _DecodedPath:
    """Decode ``pre-DNA -> known-profile DNA`` with a free end and no non-DNA state.

    The MAP path may stop in any DNA state, so the whole post-pre-DNA tail is DNA.
    ``non_dna_emission`` is accepted for a uniform decoder signature but is unused;
    the null score also drops the non-DNA term.
    """

    n_steps, n_dna = dna_profile_emission.shape
    if log_dna_transition.shape != (n_dna, n_dna):
        raise ValueError("DNA transition matrix shape does not match DNA profile")
    if any(
        len(score) != n_steps
        for score in (pre_dna_emission, dna_emission, non_dna_emission)
    ):
        raise ValueError("all region emissions must match the observed step count")
    del non_dna_emission  # no non-DNA state in dna_only decoding

    required_span = ceil(params.min_anchor_coverage * n_dna)
    max_entry = n_dna - required_span
    if max_entry < 0:
        raise ValueError("DNA profile is too short for min_anchor_coverage")
    entry_states = np.arange(max_entry + 1, dtype=np.int64)

    pre_dna_state = 0
    dna_start_state = 1
    dna_stop_state = dna_start_state + n_dna
    n_states = dna_stop_state  # no non-DNA state

    emit = np.full((n_steps, n_states), NEG_INF)
    emit[:, pre_dna_state] = pre_dna_emission
    emit[:, dna_start_state:dna_stop_state] = dna_profile_emission

    transition = np.full((n_states, n_states), NEG_INF)
    transition[pre_dna_state, pre_dna_state] = 0.0
    entry_logp = params.log_anchor_entry_prior - float(np.log(len(entry_states)))
    dna_entry_columns = dna_start_state + entry_states
    transition[pre_dna_state, dna_entry_columns] = entry_logp
    transition[
        dna_start_state:dna_stop_state,
        dna_start_state:dna_stop_state,
    ] = log_dna_transition

    value = np.full((n_steps, n_states), NEG_INF)
    backptr = np.full((n_steps, n_states), -1, dtype=np.int64)
    value[0, pre_dna_state] = emit[0, pre_dna_state]
    value[0, dna_entry_columns] = entry_logp + emit[0, dna_entry_columns]

    for t in range(1, n_steps):
        candidates = value[t - 1][:, None] + transition
        backptr[t] = np.argmax(candidates, axis=0)
        best = candidates[backptr[t], np.arange(n_states)]
        value[t] = best + emit[t]

    # Free end, but DNA must be present: the path must terminate in a DNA state.
    dna_final = value[-1, dna_start_state:dna_stop_state]
    total_score = float(np.max(dna_final))
    state_path = np.full(n_steps, -1, dtype=np.int64)
    if np.isfinite(total_score):
        state_path[-1] = dna_start_state + int(np.argmax(dna_final))
        for t in range(n_steps - 1, 0, -1):
            state_path[t - 1] = backptr[t, state_path[t]]

    null_score = float(max(np.sum(pre_dna_emission), np.sum(dna_emission)))
    return _DecodedPath(
        state_path=state_path,
        total_score=total_score,
        null_score=null_score,
        score_margin=total_score - null_score,
        pre_dna_state=pre_dna_state,
        dna_start_state=dna_start_state,
        dna_stop_state=dna_stop_state,
        non_dna_state=-1,
    )


def _rejected_result(
    observed: _ObservedArrays,
    params: HybridSegmentParams,
    decoded: _DecodedPath,
    anchor_emission_margin: float = NEG_INF,
) -> HybridSegmentResult:
    return HybridSegmentResult(
        status="rejected",
        step_labels=np.full(len(observed), "pre_dna", dtype="<U8"),
        dna_path=np.full(len(observed), -1, dtype=np.int64),
        dna_start_step=-1,
        dna_end_step=-1,
        dna_start_sample=-1,
        dna_end_sample=-1,
        boundary_step=-1,
        boundary_sample=-1,
        anchor_start_step=-1,
        anchor_end_step=-1,
        anchor_start_sample=-1,
        anchor_end_sample=-1,
        anchor_ref_start=-1,
        anchor_ref_end=-1,
        anchor_ref_coverage=0.0,
        anchor_observed_fraction=0.0,
        orientation=params.orientation,
        total_score=decoded.total_score,
        null_score=decoded.null_score,
        score_margin=decoded.score_margin,
        anchor_emission_margin=anchor_emission_margin,
    )


def segment_dna_non_dna(
    steps: Sequence[Step] | object,
    full_dna_profile: pd.DataFrame,
    region_models: HybridRegionModels,
    params: HybridSegmentParams = HybridSegmentParams(),
) -> HybridSegmentResult:
    """Locate one known DNA block and its transition to non-DNA signal.

    ``steps`` must represent one already isolated and segmented event. Each step
    must carry mean, standard deviation, dwell, and non-overlapping raw-sample
    bounds. ``full_dna_profile`` must be the unfiltered known-DNA profile.
    ``region_models`` must contain calibrated models labelled exactly
    ``pre_dna``, ``dna``, and ``non_dna``.

    Step levels, profile levels, and region models must already share a current
    scale. Do not fit scaling over the whole hybrid event because its non-DNA
    levels would contaminate DNA calibration. This function does not isolate
    events, segment raw signal, or fit the region models.

    The decoded topology is exactly
    ``PRE_DNA -> DNA[h] -> NON_DNA``. ``boundary_step`` is the observed-step
    index of the first non-DNA step and equals the half-open ``dna_end_step``.
    ``boundary_sample`` is that step's raw-sample start; it is a global source
    coordinate when the input step bounds are global.

    Accepted labels are ``pre_dna`` before the known-profile DNA block, ``dna``
    for that block, and ``non_dna`` after the boundary. Rejected results use
    ``-1`` for inferred coordinates and must not be interpreted as boundaries.
    """

    if params.orientation != "dna_to_non_dna":
        raise NotImplementedError(
            "the mechanical MVP currently implements only dna_to_non_dna; "
            "anchor selection already supports either boundary edge"
        )

    observed = _coerce_observed_steps(steps)
    anchor = select_trusted_anchor(
        full_dna_profile,
        boundary_edge=params.boundary_edge,
        boundary_trim_steps=params.boundary_trim_steps,
    )
    pre_dna_score = _validated_model_scores(region_models.pre_dna, observed)
    dna_score = _validated_model_scores(region_models.dna, observed)
    non_dna_score = _validated_model_scores(region_models.non_dna, observed)
    anchor_score = emission(observed.mean, observed.std, anchor.mean, anchor.std)
    log_transition = build_transition(
        len(anchor),
        pstep=params.pstep,
        phold=params.phold,
        pmiss=params.pmiss,
        kmax=params.kmax,
    )

    decoder = _viterbi_dna_only if params.dna_only else _viterbi_dna_to_non_dna
    decoded = decoder(
        anchor_score,
        log_transition,
        pre_dna_score,
        dna_score,
        non_dna_score,
        params,
    )
    if not np.isfinite(decoded.total_score):
        return _rejected_result(observed, params, decoded)

    path = decoded.state_path
    dna_mask = (path >= decoded.dna_start_state) & (
        path < decoded.dna_stop_state
    )
    non_dna_mask = path == decoded.non_dna_state  # all-False in dna_only (state -1)
    if not np.any(dna_mask) or (
        not params.dna_only and not np.any(non_dna_mask)
    ):
        return _rejected_result(observed, params, decoded)

    dna_indices = np.flatnonzero(dna_mask)
    non_dna_indices = np.flatnonzero(non_dna_mask)
    dna_profile_path = path[dna_mask] - decoded.dna_start_state
    coverage = float(
        (np.max(dna_profile_path) - np.min(dna_profile_path) + 1) / len(anchor)
    )
    observed_fraction = float(len(np.unique(dna_profile_path)) / len(anchor))
    dna_profile_model_score = float(
        np.sum(anchor_score[dna_indices, dna_profile_path])
    )
    best_region_score = np.maximum.reduce(
        [
            pre_dna_score[dna_indices],
            dna_score[dna_indices],
            non_dna_score[dna_indices],
        ]
    )
    anchor_emission_margin = dna_profile_model_score - float(
        np.sum(best_region_score)
    )
    rejected = (
        coverage < params.min_anchor_coverage
        or observed_fraction < params.min_anchor_observed_fraction
        or decoded.score_margin < params.min_score_margin
        or anchor_emission_margin < params.min_anchor_emission_margin
    )
    if not params.dna_only:
        rejected = rejected or len(non_dna_indices) < params.min_non_dna_steps
    if rejected:
        return _rejected_result(
            observed, params, decoded, anchor_emission_margin
        )

    labels = np.full(len(observed), "pre_dna", dtype="<U8")
    labels[dna_mask] = "dna"
    labels[non_dna_mask] = "non_dna"
    dna_path = np.full(len(observed), -1, dtype=np.int64)
    dna_path[dna_mask] = dna_profile_path
    dna_start = int(dna_indices[0])
    used_ref_pos = anchor.ref_pos[dna_profile_path]

    if params.dna_only:
        # Free-end DNA runs to the read end; there is no non-DNA boundary.
        dna_end = len(observed)
        boundary_step = -1
        boundary_sample = -1
        dna_end_sample = int(observed.end_sample[-1])
    else:
        boundary = int(non_dna_indices[0])
        dna_end = boundary
        boundary_step = boundary
        boundary_sample = int(observed.start_sample[boundary])
        dna_end_sample = int(observed.start_sample[boundary])

    return HybridSegmentResult(
        status="pass",
        step_labels=labels,
        dna_path=dna_path,
        dna_start_step=dna_start,
        dna_end_step=dna_end,
        dna_start_sample=int(observed.start_sample[dna_start]),
        dna_end_sample=dna_end_sample,
        boundary_step=boundary_step,
        boundary_sample=boundary_sample,
        anchor_start_step=dna_start,
        anchor_end_step=dna_end,
        anchor_start_sample=int(observed.start_sample[dna_start]),
        anchor_end_sample=dna_end_sample,
        anchor_ref_start=int(np.min(used_ref_pos)),
        anchor_ref_end=int(np.max(used_ref_pos)),
        anchor_ref_coverage=coverage,
        anchor_observed_fraction=observed_fraction,
        orientation=params.orientation,
        total_score=decoded.total_score,
        null_score=decoded.null_score,
        score_margin=decoded.score_margin,
        anchor_emission_margin=anchor_emission_margin,
    )


def segment_dna_only(
    steps: Sequence[Step] | object,
    full_dna_profile: pd.DataFrame,
    region_models: HybridRegionModels,
    params: HybridSegmentParams = HybridSegmentParams(),
) -> HybridSegmentResult:
    """Segment an event assumed to contain no non-DNA region.

    Convenience wrapper that forces ``dna_only=True``. The decoded topology is
    ``PRE_DNA -> DNA[h]`` with a free end: the whole post-pre-DNA tail is labelled
    ``dna``, ``boundary_step``/``boundary_sample`` are ``-1`` (no boundary), and
    ``dna_end_step`` equals the observed step count. The coverage, observed-fraction,
    score-margin, and anchor-emission-margin rejections still apply, so a genuinely
    non-DNA event is rejected. See ``segment_dna_non_dna`` for the input contract.
    """

    if params.orientation != "dna_to_non_dna":
        raise NotImplementedError(
            "dna_only currently supports only the dna_to_non_dna anchor orientation"
        )
    return segment_dna_non_dna(
        steps,
        full_dna_profile,
        region_models,
        replace(params, dna_only=True),
    )
