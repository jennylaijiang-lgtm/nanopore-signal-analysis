"""Component 1: semi-global step-level signal alignment (HMM).

segment.py    raw signal -> observed steps (mean, std, dwell)
transition.py physical half-step transition matrix (pstep/phold/pmiss), cached
cost.py       local cost matrices C[Na, Nb], shared by dp + dtw
dp.py         emission + Viterbi (s,e + labels) + forward-backward (logL + posteriors)
dtw.py        dynamic time warping: pairwise + fragment-local, no profile needed
denovo.py     reference-free basecall against the whole 6-mer LUT
hybrid_segment.py trusted-anchor DNA -> non-DNA boundary segmentation
qc.py         per-read QC + LUT std fitting from path residuals
record.py     per-step + read-level alignment record (Parquet)

Three aligners, three questions -- pick by what is KNOWN:
    dp      molecule known         -> (s,e) + per-step labels + logL   [Component 1]
    denovo  molecule unknown, LUT known -> called sequence, no (s,e)
    dtw     no profile at all      -> warping path + distance, no (s,e), no logL
`dp` and `denovo` share `dp.emission`; `dp` and `dtw` share `cost.py`. DTW is NOT a
Component 1 aligner -- see `dtw.py` for why (and where it does belong: fragment-local
analysis, reference-free grouping, and DBA consensus).
"""
