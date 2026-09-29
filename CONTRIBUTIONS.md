# Contributions and attribution

This repository presents selected work by **Jenny Jiang** from a research project in the Cees Dekker Lab at TU Delft, supervised by **Dr Xiuqi Chen**. It combines Jenny's implementations and research analyses with supporting code from the shared `cowler` project.

## Jenny's contributions

- `cowler/consensus/dba.py`: implementation of uncertainty-weighted DTW barycentre averaging, developed from an existing documented scaffold. The contribution is the implementation and its integration into this research workflow; it is not a claim to have invented DBA or DTW.
- `cowler/consensus/length_normalize.py`: numerical preprocessing, mean/variance interpolation, target-length selection and input validation.
- The selected consensus/classification, preprocessing-ablation, cross-construct boundary and medoid-robustness runners document the research-analysis work showcased in this portfolio. They build on shared project algorithms and utilities.

## Supporting code

- `cowler/align/dtw.py` was written **primarily by Dr Xiuqi Chen**, Jenny's supervisor. It is included as an attributed dependency, not as an example of Jenny's independent algorithm implementation.
- `cowler/align/segment.py` identifies its CPIC step finder as a port from the existing poreFlow workflow. That provenance is retained in the source.
- Other support modules and the imported tests come from the shared `cowler` research project. Their inclusion does not imply sole authorship by Jenny. Existing module documentation is retained, including references to components outside this selected subset.

## Portfolio preparation

The selected algorithms and experimental runners retain their source implementations. Codex assisted with repository packaging, documentation, the new synthetic demonstration and portability checks. The newly generated example is demonstration scaffolding, not an original internship deliverable.

The only changes to imported production code are in `cowler/io/lut.py`, `cowler/io/normalize.py` and `cowler/io/__init__.py`: calibration data is loaded lazily, `COWLER_LUT_PATH` can select an external CSV, a clear missing-data error is provided, and the eager package-level calibration-table export is removed. The calibration resource import also uses the Python 3.11+ standard-library location. This lets the synthetic example and command-line help work without distributing laboratory data. The two featured consensus implementations are unchanged.

`SOURCE_MANIFEST.json` records original source hashes before these packaging changes. No experimental datasets, calibration tables, internal reports or original Git history are included. Approval to share the selected code does not itself grant an open-source licence; no licence has been added.

The boundary bookkeeping test now substitutes the calibration-profile loader alongside its existing fake decoder, so it can test unchanged model reuse without measured data. New portability tests exercise missing-table handling and loading an explicitly supplied calibration CSV.

Six original DBA preprocessing-ablation tests and their event fixture are extracted from the shared project test module; unrelated HMM and plotting tests/imports are omitted.
