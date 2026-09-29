"""Internal coercion and normalization helpers for signal consensus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from ..io.normalize import robust_params

FloatArr = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SignalSteps:
    mean: FloatArr
    std: FloatArr
    dwell: FloatArr


def signal_steps(read: Any) -> SignalSteps:
    """Coerce a StepRead-like object or ``(mean, std[, dwell])`` tuple."""
    if hasattr(read, "mean") and hasattr(read, "std"):
        mean = np.asarray(read.mean, dtype=float)
        std = np.asarray(read.std, dtype=float)
        dwell_value = getattr(read, "dwell", np.ones(mean.size, dtype=float))
        dwell = np.asarray(dwell_value, dtype=float)
    else:
        values = tuple(read)
        if len(values) not in (2, 3):
            raise ValueError("a read tuple must contain (mean, std[, dwell])")
        mean = np.asarray(values[0], dtype=float)
        std = np.asarray(values[1], dtype=float)
        dwell = (
            np.asarray(values[2], dtype=float)
            if len(values) == 3
            else np.ones(mean.size, dtype=float)
        )

    if mean.ndim != 1 or mean.size == 0:
        raise ValueError("read mean must be a non-empty one-dimensional array")
    if std.shape != mean.shape or dwell.shape != mean.shape:
        raise ValueError("read mean, std, and dwell must have matching shapes")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("read mean/std contains NaN or infinity")
    if not np.all(np.isfinite(dwell)):
        raise ValueError("read dwell contains NaN or infinity")
    if np.any(std < 0.0) or np.any(dwell < 0.0):
        raise ValueError("read std and dwell must be non-negative")
    return SignalSteps(mean=mean, std=std, dwell=dwell)


def prepare_reads(
    reads: Sequence[Any], *, normalize: bool, min_std: float
) -> list[SignalSteps]:
    """Validate reads and optionally robust-normalize each one."""
    if len(reads) == 0:
        raise ValueError("reads must be non-empty")
    prepared = [signal_steps(read) for read in reads]
    if not normalize:
        return [
            SignalSteps(
                mean=read.mean,
                std=np.maximum(read.std, min_std),
                dwell=read.dwell,
            )
            for read in prepared
        ]

    normalized: list[SignalSteps] = []
    for read in prepared:
        shift, scale = robust_params(read.mean)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        normalized.append(
            SignalSteps(
                mean=(read.mean - shift) / scale,
                std=np.maximum(read.std / scale, min_std),
                dwell=read.dwell.copy(),
            )
        )
    return normalized


def validate_medoid_index(medoid_index: int, n_reads: int) -> int:
    index = int(medoid_index)
    if not 0 <= index < n_reads:
        raise ValueError("medoid_index is outside the read collection")
    return index
