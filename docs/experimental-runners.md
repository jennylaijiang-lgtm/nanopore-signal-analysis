# Experimental runners: inputs and execution order

These scripts are research-code examples. They do not run the full experiments from this repository alone. No FAST5 data, annotation tables, measured calibration values, stored split labels, baseline results or construct illustrations are distributed here.

## Environment and calibration

From the repository root, with the virtual environment activated:

```bash
python -m pip install -e '.[research]'
export COWLER_LUT_PATH=/absolute/path/to/DNA_6mer_prediction_model_cv.csv
```

The DNA lookup table must provide `kmer_pull_3_5`, `pre_mean`, `pre_std`, `post_mean` and `post_std` columns; a complete six-base model has 4,096 rows. The values must be the appropriate calibrated model for the experiment. The code does not generate a substitute table. Python callers may instead pass their table directly to `predict_DNA_6mer_5_3(..., lut=table)`.

The default calibration table is loaded on first use and cached for that process. Restart Python after changing the environment variable. Imports, `--help`, the synthetic example and the selected tests do not require this table.

Use `python -m scripts.<runner>` from the repository root so that the shared `cowler` and `scripts` imports resolve. Historical relative paths in runner defaults are preserved. Supply explicit paths for a different data layout.

## Consensus and preprocessing

The FAST5 raw/annotation pairs contain experiment-specific schemas and metadata, including event boundaries and stored `isConsensus` labels. Expected filenames include `JS445_synthetic.fast5` and `JS445_synthetic.annot.fast5`; these are original experiment filenames, not files produced by the standalone example.

```bash
python -m scripts.evaluate_psk_dba_consensus --help
python -m scripts.evaluate_psk_preprocessing_ablation --help
python -m scripts.evaluate_psk_length_normalized_consensus --help
```

For an authorised local dataset, the primary runner accepts:

```bash
python -m scripts.evaluate_psk_dba_consensus \
  --data-dir /absolute/path/to/otherfast5 \
  --js445-dir /absolute/path/to/js445 \
  --out-dir results/dba
```

The preprocessing-ablation and length-normalised comparison runners also accept `--data-dir`, `--js445-dir` and `--out-dir`. The ablation compares eight conditions while preserving stored training/test labels. Data-derived length targets must use training traces only. Upstream distance-outlier screening nevertheless uses eventual held-out traces in the original workflow: preserve that qualification when interpreting scores.

## Cross-construct boundary evaluation

```bash
python -m scripts.evaluate_boundary_constructs --help
python -m scripts.evaluate_boundary_constructs \
  --data-dir /absolute/path/to/otherfast5 \
  --js445-dir /absolute/path/to/js445 \
  --out-dir results/boundaries
```

This uses JS445–JS453 raw/annotation pairs and the calibration table. It compares event-held-out fits within each construct with region models fitted on JS445 and transferred without target refitting. The imported JS445 evaluator provides shared data-loading and prediction helpers. Accuracy denominators exclude rejected or errored events, whose counts must be reported separately.

## Medoid robustness

The complete included analysis comprises **Stage 1 → Stage 2 → Stage 2 follow-up → Stage 3 → Stage 4**. These are gated stages, not independent commands. Each validates prior manifests, hashes and baseline reproduction before proceeding.

Stage 1 requires all of the following, with matching event identities and preprocessing:

- Frozen baseline directory, including `run_summary.json`, `split_manifest.csv`, `upstream_outlier_metrics.csv` and baseline profiles/classification artifacts consumed by later stages.
- Within-dataset DTW distance archive and companion outlier-metrics rows.
- Retained trace archive and its event/trace manifest.

Use the explicit Stage 1 options to locate those existing research artifacts:

```bash
python -m scripts.evaluate_medoid_robustness \
  --baseline-dir /absolute/path/to/baseline \
  --distance-cache /absolute/path/to/within_dataset_dtw_distances.npz \
  --distance-rows /absolute/path/to/upstream_outlier_metrics.csv \
  --trace-cache /absolute/path/to/native_filtered_traces.npz \
  --trace-rows /absolute/path/to/trace_manifest.csv \
  --out-dir results/medoid
python -m scripts.evaluate_medoid_robustness_stage2 --out-dir results/medoid
python -m scripts.evaluate_medoid_robustness_stage2_followup --out-dir results/medoid
python -m scripts.evaluate_medoid_robustness_stage3 --out-dir results/medoid
python -m scripts.evaluate_medoid_robustness_stage4 --out-dir results/medoid
```

Stage 4 retains the original required icon location: `../Inkscape/psk_no_border/` relative to the repository root, containing `JS445.png` through `JS453.png`. Those illustrations are not included. The run also expects consistent Git provenance across stages, so use a committed checkout without editing it between stages.

This portfolio includes all staged analysis code, but it does not recreate every upstream cache-building workflow. Arbitrary newly generated files or a run using different preprocessing defaults are not interchangeable with a frozen historical baseline. Stage validation failures should be investigated, not bypassed to force an output.

## Scope of verification

The standalone example demonstrates the public APIs with generated signals. The included tests exercise algorithms and bookkeeping using synthetic fixtures. Experimental runners can be imported and their help inspected without private inputs; the original full experiments cannot be validated here without those inputs.

Original source docstrings may refer to design documents, notebooks or historical outputs in the full research project. Those internal materials are not bundled; this guide documents the setup needed for the selected portfolio subset.

## Tests and type checks

```bash
python -m pip install -e '.[demo,research,dev]'
python -m pytest
python -m pyright
```

The tests exercise algorithms and bookkeeping with generated fixtures. Pyright covers the example and modified calibration-loading modules, not a full audit of inherited research code. GitHub Actions runs these checks and the synthetic workflow on Python 3.12 without laboratory data.
