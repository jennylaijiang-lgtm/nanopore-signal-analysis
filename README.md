# Nanopore signal analysis

Selected Python implementations for nanopore signal processing, alignment and consensus analysis.

A research-code portfolio by **Jenny Jiang**, drawn from a research project in the Cees Dekker Lab at TU Delft, supervised by **Dr Xiuqi Chen**. The examples focus on consensus signal estimation and the experiments used to evaluate preprocessing, classification and robustness.

## Start with these three files

| Code | What to look for |
| --- | --- |
| [Uncertainty-weighted DBA](cowler/consensus/dba.py) | Jenny's consensus implementation, developed from an existing documented scaffold: medoid initialisation, inverse-variance weighting, convergence tracking and distinct-read depth. |
| [Length normalisation](cowler/consensus/length_normalize.py) | Jenny's preprocessing implementation: a shared relative-position grid, variance interpolation, training-derived target lengths and input validation. |
| [Preprocessing ablation](scripts/evaluate_psk_preprocessing_ablation.py) | A frozen training/test assignment across eight preprocessing conditions, with both retained-cohort and common-cohort comparisons. Requires external experimental data. |

The underlying [DTW implementation](cowler/align/dtw.py) was written primarily by the supervisor and is included as a supporting dependency. See [Contributions and attribution](CONTRIBUTIONS.md) for the distinction between Jenny's work, inherited code and portfolio preparation.

## Run a self-contained example

Use Python 3.11 or newer (tested with Python 3.12). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m examples.synthetic_consensus
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.

The example generates 16 noisy traces of varying lengths, interpolates their means and variances to 64 points, and fits an uncertainty-weighted DBA consensus. It writes `results/synthetic/consensus.csv` and `results/synthetic/summary.json`, including convergence, read depth and error against the artificial reference. Use `--seed` or `--out-dir` to change the seed or output directory. The first run may take longer while Numba compiles DTW.

All example signals are generated in arbitrary units. They are not experimental traces or a validated physical simulation. The consensus is a **signal profile**, not an amino-acid sequence; interpolated positions are not physical enzyme steps. The reported profile spread is not a confidence interval.

## Selected research analyses — external data required

| Runner | Purpose |
| --- | --- |
| [DBA consensus and classification](scripts/evaluate_psk_dba_consensus.py) | Preprocess events, fit profiles and classify peptide variants, with controls and resampling analyses. |
| [Preprocessing ablation](scripts/evaluate_psk_preprocessing_ablation.py) | Compare trimming, outlier removal and length normalisation on a fixed stored split. |
| [Boundary transfer](scripts/evaluate_boundary_constructs.py) | Compare within-construct fits with a frozen model transferred across constructs. |
| [Medoid robustness, Stage 1](scripts/evaluate_medoid_robustness.py) | Identify alternative initialisation candidates and preserve input provenance. |
| [Stage 2](scripts/evaluate_medoid_robustness_stage2.py) · [Stage 2 follow-up](scripts/evaluate_medoid_robustness_stage2_followup.py) | Refit alternative consensuses after reproducing the baseline; compare candidates and fitted profiles. |
| [Stage 3](scripts/evaluate_medoid_robustness_stage3.py) · [Stage 4](scripts/evaluate_medoid_robustness_stage4.py) | Measure classification sensitivity and assemble summaries from validated stage outputs. |

The JS445 segmentation and length-normalised classification runners are also included because the selected analyses import their helpers. The `cowler/` package preserves the supporting import structure; it is a selected subset of the research project, not a full basecalling toolkit.

**Experimental data is not included.** Research runners additionally require FAST5 files, annotation tables and the laboratory DNA calibration CSV. Medoid analyses need frozen baseline outputs and cached traces/distances; Stage 4 also needs construct-icon PNGs. See [Experimental setup and execution order](docs/experimental-runners.md). A filename containing `synthetic.fast5` does not mean that input is distributed with this repository.

These runners preserve their research assumptions. In particular, upstream filtering can use eventual held-out traces, making those evaluations test-informed; within-run results are not independent biological validation. No experimental accuracy claims are reproduced by the synthetic example.

## Dependencies and checks

The example uses NumPy, SciPy, Numba and pandas through the shared package. Experimental runners additionally use h5py and Matplotlib. Install the research and development extras to run the included tests:

```bash
python -m pip install -e '.[research,dev]'
python -m pytest
python -m pyright
```

The tests cover DTW constraints, consensus behaviour, interpolation, boundary-evaluation bookkeeping and medoid baseline gates using generated inputs. They do not reproduce the private-data experiments. The type-check command covers the new example and the modified calibration-loading modules; it is not a full audit of the inherited research code.

[Source manifest](SOURCE_MANIFEST.json) records hashes of the original selected source files. No open-source licence is granted by this portfolio; sharing approval and permission to reuse or redistribute code are distinct.
