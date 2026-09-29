"""Chemistry-agnostic multi-read signal consensus for peptide nanopore data.

Implemented signal workflow
---------------------------
``cluster`` computes pairwise uncertainty-weighted DTW distances, cuts a
hierarchical clustering, and selects one medoid per cluster.  The same
``SignalCluster`` then initializes either:

- ``dba``: hard DTW alignments plus iterative barycenter averaging.
- ``hmm``: soft profile-HMM alignments plus forward-backward refinement.

Both return a consensus *signal profile* with per-position depth/uncertainty.
They intentionally do not emit a residue sequence: peptide steps per residue and
the residue-k-mer level model are chemistry measurements, not DNA defaults.

Still scaffolded
----------------
``posterior``, ``poa``, ``vote``, and the Parquet/orchestration layer in
``consensus`` remain future strategies.  Experimental labels or Component 2
assignments in ``group`` remain alternatives when same-peptide identity is known
without reference-free clustering.
"""

from .cluster import (
    ClusteringResult,
    MedoidCandidate,
    SignalCluster,
    cluster_from_distances,
    cluster_reads_dtw,
    medoid,
    rank_medoid_candidates,
)
from .dba import Barycenter, dba, dba_cluster
from .hmm import (
    ProfileHMMConsensus,
    ProfileHMMScore,
    ProfileHMMSettings,
    hmm_cluster,
    hmm_consensus,
    hmm_transition_matrix,
    score_hmm_profile,
    snapshot_hmm_profile,
)

__all__ = [
    "Barycenter",
    "ClusteringResult",
    "MedoidCandidate",
    "ProfileHMMConsensus",
    "ProfileHMMScore",
    "ProfileHMMSettings",
    "SignalCluster",
    "cluster_from_distances",
    "cluster_reads_dtw",
    "dba",
    "dba_cluster",
    "hmm_cluster",
    "hmm_consensus",
    "hmm_transition_matrix",
    "medoid",
    "rank_medoid_candidates",
    "score_hmm_profile",
    "snapshot_hmm_profile",
]
