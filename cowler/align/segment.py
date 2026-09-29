"""Step finding: raw signal -> observed steps (Component 1 front-end).

Each read's normalized signal x[t] is segmented into piecewise-constant levels;
each level = one observed step = (mean, std, dwell_samples). These steps -- not
raw samples -- are the observation sequence the aligner consumes, matching the
enzyme-step transition model (pstep/phold/pmiss) and the proven MATLAB workflow,
where alignment input was already `level_mean` / `level_std` per step.

Algorithm: CPIC (Change-point Penalized Information Criterion) recursive change
detection -- the level detector from the proven poreFlow workflow (ported from
`tmp/find_steps.py`, not line-reimplemented). A greedy recursive split proposes
transitions where the CPIC drops below zero, then a merge pass removes any split
whose removal raises the CPIC. Oversegmentation is absorbed downstream by the
aligner's phold term, so `sensitivity`/`min_level_length` bias slightly toward
over- rather than under-segmentation.

Public API:
    Step                   -- dataclass: mean, std, dwell, start_sample, end_sample
    find_steps(x, params)  -- normalized signal -> list[Step]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_MIN_VAR = 3e-4  # variance floor: guards log(var) for near-flat levels


def _cpic_penalty(n: float) -> float:
    """Per-transition information penalty p_CPIC as a function of segment length n."""
    if n > 1000.0:
        x = min(n, 1e6)
        a, b, c = 2.456, 1.187, 2.73
        return a * np.log(np.log(x)) + b * np.log(np.log(np.log(x))) + c
    a, b, c = 1.239, 0.9872, 1.999
    p3, p4, p5, ph = 5.913e-10, -1.876e-06, 0.004354, -0.1906
    return (
        a * np.log(np.log(n))
        + b * np.log(n)
        + p3 * n**3
        + p4 * n**2
        + p5 * n
        + ph * np.abs(n) ** 0.5
        + c
    )


def _step_finder_cpic(
    data: npt.NDArray[np.float64],
    sensitivity: float = 1.0,
    min_level_length: int = 2,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Segment `data` into levels. Returns (transitions, features).

    transitions: sample indices [0, ..., len(data)] bounding each level.
    features:    [2, n_levels] -> row 0 = mean, row 1 = std of each level.
    NaNs are dropped and transitions mapped back to original sample indices.
    """
    original_mapping = np.arange(len(data))
    data_length = len(data)
    keep = ~np.isnan(data)
    original_mapping = original_mapping[keep]
    data = data[keep]

    if len(data) < 2 * min_level_length:
        feats = np.array([[np.mean(data)], [np.std(data)]]) if len(data) else np.zeros((2, 0))
        return np.array([0, data_length]), feats

    # 1-indexed prefix sums (leading 0) so slices match the MATLAB workflow.
    x = np.concatenate(([0.0], np.cumsum(data)))
    xsq = np.concatenate(([0.0], np.cumsum(data**2)))

    def _cpic(left: int, N_L, right: int):
        N_T = right - left + 1
        N_R = N_T - N_L
        x_mean_L = (x[left + N_L - 1] - x[left - 1]) / N_L
        x_mean_R = (x[right] - x[right - N_R]) / N_R
        x_mean_T = (x[right] - x[left - 1]) / N_T
        xsq_mean_L = (xsq[left + N_L - 1] - xsq[left - 1]) / N_L
        xsq_mean_R = (xsq[right] - xsq[right - N_R]) / N_R
        xsq_mean_T = (xsq[right] - xsq[left - 1]) / N_T
        var_L = np.maximum(xsq_mean_L - x_mean_L**2, _MIN_VAR)
        var_R = np.maximum(xsq_mean_R - x_mean_R**2, _MIN_VAR)
        var_T = np.maximum(xsq_mean_T - x_mean_T**2, _MIN_VAR)
        p = _cpic_penalty(min(N_T, 1e6) if N_T > 1000 else N_T)
        return (
            0.5 * (N_L * np.log(var_L) + N_R * np.log(var_R) - N_T * np.log(var_T))
            + 1
            + sensitivity * p
        )

    def find_transitions(left: int, right: int) -> list[int]:
        N_T = right - left + 1
        if N_T < 2 * min_level_length:
            return []
        N_L = np.arange(min_level_length, N_T - min_level_length + 1)
        cpic = _cpic(left, N_L, right)
        i = int(np.argmin(cpic))
        if cpic[i] >= 0:
            return []
        split = i + min_level_length + left - 1
        return (
            [split]
            + find_transitions(left, split)
            + find_transitions(split + 1, right)
        )

    transitions = sorted(find_transitions(1, len(data)))
    transitions = [
        t for t in transitions if min_level_length < t < len(data) - min_level_length
    ]
    bounds = [0] + transitions + [len(data)]

    # Merge pass: drop the transition whose removal most improves (raises) the CPIC.
    changed = True
    while changed:
        changed = False
        gain = np.full(len(bounds), -np.inf)
        for ii in range(1, len(bounds) - 1):
            left = max(bounds[ii - 1], 1)
            right = bounds[ii + 1]
            if right - left + 1 < 2 * min_level_length:
                continue
            gain[ii] = _cpic(left, bounds[ii] - left + 1, right)
        j = int(np.argmax(gain))
        if gain[j] > 0:
            bounds.pop(j)
            changed = True

    n_levels = len(bounds) - 1
    features = np.zeros((2, n_levels))
    for ct in range(n_levels):
        seg = data[bounds[ct] : bounds[ct + 1]]
        features[:, ct] = [np.mean(seg), np.std(seg)]

    transitions = np.array(
        [0] + [int(original_mapping[t]) for t in bounds[1:-1]] + [data_length]
    )
    return transitions, features


@dataclass
class Step:
    mean: float          # level mean (pA, normalized units)
    std: float           # within-level sample spread
    dwell: int           # level length in samples
    start_sample: int
    end_sample: int


def find_steps(
    x: npt.NDArray[np.float64],
    sensitivity: float = 1.0,
    min_level_length: int = 2,
) -> list[Step]:
    """Segment a normalized signal into observed steps (CPIC level detector).

    `sensitivity` scales the change-point penalty (lower -> more, shorter levels);
    `min_level_length` is the smallest resolvable level in samples. Bias toward
    slight oversegmentation -- the aligner's phold term absorbs it.
    """
    x = np.asarray(x, float)
    transitions, features = _step_finder_cpic(x, sensitivity, min_level_length)
    steps: list[Step] = []
    for i in range(len(transitions) - 1):
        a, b = int(transitions[i]), int(transitions[i + 1])
        steps.append(
            Step(
                mean=float(features[0, i]),
                std=float(features[1, i]),
                dwell=b - a,
                start_sample=a,
                end_sample=b,
            )
        )
    return steps
