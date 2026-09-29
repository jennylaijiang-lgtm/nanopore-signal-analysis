"""Generate variable-length noisy traces, resample them, and fit a DBA profile.

Run from the repository root: python -m examples.synthetic_consensus
All signals are artificial and in arbitrary units. This is an API demonstration,
not a physical nanopore simulation or a reproduction of experimental results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from cowler.consensus.dba import dba
from cowler.consensus.length_normalize import resample_signal


def template(position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A smooth artificial profile with two distinct features."""
    return (
        0.6 * np.sin(2.0 * np.pi * position)
        + 0.35 * np.exp(-((position - 0.30) / 0.08) ** 2)
        - 0.25 * np.exp(-((position - 0.72) / 0.06) ** 2)
    )


def run(out_dir: Path, seed: int = 42) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    target_length = 64
    reads: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    source_lengths: list[int] = []
    for index in range(16):
        length = int(rng.integers(48, 81))
        position = np.linspace(0.0, 1.0, length)
        # Different noise levels let inverse-variance weighting affect the fit.
        noise_std = 0.04 + 0.01 * (index % 4)
        mean = template(position) + rng.normal(0.0, noise_std, length)
        resampled = resample_signal(mean, np.full(length, noise_std), target_length)
        reads.append((resampled.mean, resampled.std))
        source_lengths.append(length)

    # Traces already share a scale; avoid normalising away their amplitude.
    profile = dba(reads, normalize=False, max_iter=50, tol=1e-6)
    position = np.linspace(0.0, 1.0, target_length)
    truth = template(position)
    rmse = lambda values: float(np.sqrt(np.mean((values - truth) ** 2)))
    summary: dict[str, object] = {
        "data": "synthetic demonstration; arbitrary signal units",
        "seed": seed,
        "n_reads": len(reads),
        "source_lengths": source_lengths,
        "target_length": target_length,
        "medoid_index": profile.medoid_index,
        "iterations": profile.n_iter,
        "converged": bool(profile.converged),
        "medoid_rmse": rmse(reads[profile.medoid_index][0]),
        "consensus_rmse": rmse(profile.mean),
        "minimum_read_depth": int(profile.depth.min()),
        "all_positions_supported": bool(profile.supported.all()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_dir / "consensus.csv",
        np.column_stack([position, truth, profile.mean, profile.std, profile.depth]),
        delimiter=",",
        header="relative_position,synthetic_truth,consensus_mean,consensus_std,read_depth",
        comments="",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
