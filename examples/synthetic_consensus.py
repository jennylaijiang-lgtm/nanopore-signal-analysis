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

from cowler.consensus.dba import Barycenter, dba
from cowler.consensus.length_normalize import resample_signal


def template(position: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A smooth artificial profile with two distinct features."""
    return (
        0.6 * np.sin(2.0 * np.pi * position)
        + 0.35 * np.exp(-((position - 0.30) / 0.08) ** 2)
        - 0.25 * np.exp(-((position - 0.72) / 0.06) ** 2)
    )


def plot_workflow(
    out_dir: Path,
    native_reads: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    reads: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    profile: Barycenter,
    seed: int,
) -> None:
    """Show every trace on common y limits; no uncertainty band is implied."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    position = np.linspace(0.0, 1.0, profile.mean.size)
    with plt.rc_context({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.facecolor": "white",
    }):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, layout="constrained")
        for index, ((native, _), (resampled, _)) in enumerate(zip(native_reads, reads)):
            label = "Generated traces" if index == 0 else None
            axes[0].plot(np.arange(native.size), native, color="#777777", lw=0.8, label=label)
            axes[1].plot(position, resampled, color="#777777", lw=0.8,
                         label="Resampled traces" if index == 0 else None)
        axes[2].plot(position, profile.mean, color="#0072B2", lw=2.3, label="DBA consensus")
        axes[2].plot(position, template(position), color="#222222", lw=1.6,
                     linestyle="--", label="Artificial reference")
        titles = ["1  Variable-length traces", "2  Shared 64-point grid", "3  Consensus profile"]
        for axis, title in zip(axes, titles):
            axis.set_title(title, loc="left", fontsize=12, pad=12)
            axis.grid(axis="y", color="#e5e5e5", lw=0.6)
            axis.legend(loc="lower left", fontsize=9, frameon=False)
            axis.set_xlabel("Relative position")
        axes[0].set_xlabel("Source point index")
        axes[0].set_ylabel("Signal (arbitrary units)")
        fig.suptitle(f"Synthetic workflow · {len(reads)} traces · seed {seed}", fontsize=14)
        fig.savefig(out_dir / "synthetic-workflow.png", dpi=160, facecolor="white")
        plt.close(fig)
    metadata = {
        "purpose": "GitHub portfolio illustration, not experimental validation",
        "seed": seed, "n_traces": len(reads), "excluded_traces": 0,
        "transformations": ["endpoint-inclusive linear mean/variance interpolation",
                            "inverse-variance weighted DBA; see summary.json for convergence"],
        "uncertainty": "No uncertainty band plotted; profile std in consensus.csv is not a CI",
        "source_data": ["native_traces.csv", "resampled_traces.csv", "consensus.csv"],
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "size_pixels": [1920, 672], "dpi": 160,
    }
    (out_dir / "figure-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run(out_dir: Path, seed: int = 42) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    target_length = 64
    reads: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    source_lengths: list[int] = []
    native_reads: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for index in range(16):
        length = int(rng.integers(48, 81))
        position = np.linspace(0.0, 1.0, length)
        # Different noise levels let inverse-variance weighting affect the fit.
        noise_std = 0.04 + 0.01 * (index % 4)
        mean = template(position) + rng.normal(0.0, noise_std, length)
        native_std = np.full(length, noise_std)
        native_reads.append((mean, native_std))
        resampled = resample_signal(mean, native_std, target_length)
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
    for name, traces in [("native_traces", native_reads), ("resampled_traces", reads)]:
        rows = np.vstack([
            np.column_stack([np.full(mean.size, index), np.arange(mean.size),
                             np.linspace(0.0, 1.0, mean.size), mean, std])
            for index, (mean, std) in enumerate(traces)
        ])
        np.savetxt(out_dir / f"{name}.csv", rows, delimiter=",",
                   header="trace_id,point_index,relative_position,mean,std", comments="")
    plot_workflow(out_dir, native_reads, reads, profile, seed)
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
