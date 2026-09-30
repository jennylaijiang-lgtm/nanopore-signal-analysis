"""Length-normalize step-level signal traces before consensus fitting.

Each trace is linearly interpolated onto a shared, endpoint-inclusive
relative-position grid before consensus fitting. The target length may be fixed
or selected from training-trace lengths. Interpolated positions are correlated
within a source trace and are not physical enzyme steps.

Only signal mean and uncertainty are transformed.  Dwell is deliberately absent:
ordinary interpolation would not conserve total event duration.  Current values
must already share the desired physical scale; this module performs no affine or
robust current normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TargetLengthSelection:
    """Auditable result of selecting one shared interpolation length."""

    method: str
    target_length: int
    percentile: float | None
    per_dataset_percentile: Mapping[str, float]
    determining_datasets: tuple[str, ...]


@dataclass(frozen=True)
class ResampledSignal:
    """Mean and uncertainty on a derived relative-position grid."""

    mean: FloatArr
    std: FloatArr
    relative_position: FloatArr
    source_length: int
    target_length: int


def select_percentile_max_target(
    lengths_by_dataset: Mapping[str, Sequence[int] | npt.NDArray[np.integer]],
    *,
    percentile: float = 95.0,
) -> TargetLengthSelection:
    """Select ``ceil(max(per-dataset percentile))`` from training lengths.

    Calculating the percentile independently within each dataset prevents a
    dataset with more training traces from receiving more weight.  Callers own
    cohort selection: excluded and held-out events must be removed before this
    function is called.
    """
    if not lengths_by_dataset:
        raise ValueError("lengths_by_dataset must be non-empty")
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be finite and within [0, 100]")

    quantiles: dict[str, float] = {}
    for dataset, values in lengths_by_dataset.items():
        lengths = np.asarray(values, dtype=float)
        if lengths.ndim != 1 or lengths.size == 0:
            raise ValueError(f"{dataset} training lengths must be a non-empty 1-D array")
        if not np.all(np.isfinite(lengths)) or np.any(lengths < 2.0):
            raise ValueError(f"{dataset} training lengths must be finite and >= 2")
        if not np.all(lengths == np.floor(lengths)):
            raise ValueError(f"{dataset} training lengths must be integers")
        quantiles[str(dataset)] = float(
            np.percentile(lengths, percentile, method="linear")
        )

    maximum = max(quantiles.values())
    target_length = int(np.ceil(maximum))
    tolerance = np.finfo(float).eps * max(1.0, abs(maximum)) * 8.0
    determining = tuple(
        dataset
        for dataset, value in quantiles.items()
        if abs(value - maximum) <= tolerance
    )
    return TargetLengthSelection(
        method="per_dataset_percentile_max",
        target_length=target_length,
        percentile=float(percentile),
        per_dataset_percentile=quantiles,
        determining_datasets=determining,
    )


def select_fixed_target(target_length: int) -> TargetLengthSelection:
    """Return an auditable fixed-length selection."""
    target = int(target_length)
    if target < 2 or target != target_length:
        raise ValueError("target_length must be an integer >= 2")
    return TargetLengthSelection(
        method="fixed",
        target_length=target,
        percentile=None,
        per_dataset_percentile={},
        determining_datasets=(),
    )


def resample_signal(
    mean: npt.ArrayLike,
    std: npt.ArrayLike,
    target_length: int,
    *,
    min_std: float = 1e-3,
) -> ResampledSignal:
    """Linearly interpolate mean and variance onto a shared relative axis.

    ``numpy.interp`` treats source levels as point samples, so transitions contain
    synthetic intermediate values.  Variance is interpolated before converting
    back to standard deviation.  The first and last source values are preserved
    exactly because both grids include 0 and 1.
    """
    source_mean = np.asarray(mean, dtype=float)
    source_std = np.asarray(std, dtype=float)
    target = int(target_length)
    if source_mean.ndim != 1 or source_mean.size < 2:
        raise ValueError("mean must be a one-dimensional array with at least two values")
    if source_std.shape != source_mean.shape:
        raise ValueError("mean and std must have matching shapes")
    if not np.all(np.isfinite(source_mean)) or not np.all(np.isfinite(source_std)):
        raise ValueError("mean and std must contain only finite values")
    if np.any(source_std < 0.0):
        raise ValueError("std must be non-negative")
    if target < 2 or target != target_length:
        raise ValueError("target_length must be an integer >= 2")
    if not np.isfinite(min_std) or min_std <= 0.0:
        raise ValueError("min_std must be finite and positive")

    source_position = np.linspace(0.0, 1.0, source_mean.size, dtype=float)
    target_position = np.linspace(0.0, 1.0, target, dtype=float)
    target_mean = np.interp(target_position, source_position, source_mean)
    target_variance = np.interp(
        target_position, source_position, np.square(source_std)
    )
    target_std = np.sqrt(np.maximum(target_variance, min_std**2))
    return ResampledSignal(
        mean=np.asarray(target_mean, dtype=float),
        std=np.asarray(target_std, dtype=float),
        relative_position=target_position,
        source_length=int(source_mean.size),
        target_length=target,
    )
