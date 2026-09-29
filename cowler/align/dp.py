"""Component 1: semi-global step-level signal alignment (HMM).

Align one read to the known molecule's half-step profile, discovering the
contiguous sub-span it covers (variable start/end). Produces the (s,e) start/end
sites, per-step labels for Components 4/5, and a read likelihood for Component 2.

Inspired by a MATLAB forward-backward decoder (`tmp/fbeMAP_original.m`) tuned on
this chemistry; re-derived, NOT line-ported (its numerics + Symbolic-Toolbox
deps do not port cleanly). The one deliberate fix vs that script: read (s,e) from
a monotone Viterbi path, NOT from per-step argmax of the posteriors. In the
original workflow `(s,e) = min/max(argmax_t W)`, so a single outlier step set the
ends and free ends were unreliable -- the path structure that pins the boundary
was thrown away.

INPUT = steps, not raw samples
    A step finder (`segment.py`) turns each read's normalized signal into N
    observed steps, each a (mean, std, dwell) triple. Alignment runs over those N
    steps. This matches the physical transition model (pstep/phold/pmiss count
    enzyme steps, not samples) and the proven workflow. N is small (tens-hundreds)
    so full O(N*L) is cheap; no banding.

States = half-step profile (length L = 2M, M = #kmers): interleaved pre/post
levels (mean, std) from the io profile builder (`predict_DNA_6mer_5_3`).

Emission -- 1D Gaussian convolution of step vs reference level:
    s(t,h) = -0.5 * [ log(2*pi) + log(v) + (xm[t] - Em[h])**2 / v ]
    with v = xs[t]**2 + Es[h]**2,  (xm,xs)=step mean/std, (Em,Es)=profile mean/std.
    Folding BOTH uncertainties down-weights noisy / short steps -- this is the
    scalar reduction of the original `smatrixAC`. (Viterbi minimizes cost = -s.)

Transition -- physical, in half-step units; build once per molecule and cache
    (see `transition.py`). Net half-step jump k between consecutive observed steps:
      pstep  P(enzyme step forward) / (fwd + back)
      phold  P(step oversegmented -> stay on same state, k=0)
      pmiss  P(an enzyme step goes unobserved)
    pk(k) = net displacement of a +-1 walk over (1 + Geom(pmiss)) enzyme steps,
    computed by a direct truncated series (NOT the MATLAB hypergeom closed form).
    Semi-global free ends via a start/end pseudo-state: free start = uniform entry
    to any state, free end = any state may terminate. NEVER force start h=0 / end
    h=L-1.

Two decoders sharing the same emission + transition:
    viterbi(...)           -> monotone MAP path. (s,e) = path-endpoint half-steps
                              mapped to bases (h//2 + BASE_OFFSET); per-step
                              half-step assignment; total_score.   [owns s,e + labels]
    forward_backward(...)  -> logL + posteriors W = P(state | read); optional Pbad.
                              Feeds Component 2 (demux, variant LLR) + soft QC.  [owns logL]

Optional, deferred (off by default): per-step contaminant rejection (`Pbad` via
lookback>1, the generalized BCJR in fbeMAP). lookback=1 -- as actually used -- is
plain forward-backward, so this machinery starts inert. Enable only if step
outliers measurably hurt.

Tuned defaults for this chemistry: pstep=0.9, phold=0.05, pmiss=0.1, lookback=1.
BASE_OFFSET is configurable (predict_DNA uses constriction offset 4, vs k//2=3) --
verify against a control molecule.

Public API:
    emission(steps, profile)                 -- -> log-emission S[N, L]
    viterbi(S, logT, params)                 -- -> path, (s, e), total_score
    forward_backward(S, logT, params)        -- -> logL, W, (Pbad)
    align(read_steps, profile, params)       -- emission + viterbi (+ fb) -> record
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NEG_INF = -np.inf
_LOG_2PI = float(np.log(2.0 * np.pi))


def emission(
    xm: npt.NDArray[np.float64],
    xs: npt.NDArray[np.float64],
    Em: npt.NDArray[np.float64],
    Es: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Log-emission S[N, L]: step (xm,xs) vs profile (Em,Es), Gaussian convolution.

        v = xs**2 + Es**2
        s(t,h) = -0.5 * [ log(2*pi) + log(v) + (xm[t] - Em[h])**2 / v ]
    """
    xm = np.asarray(xm, float)[:, None]
    xs = np.asarray(xs, float)[:, None]
    Em = np.asarray(Em, float)[None, :]
    Es = np.asarray(Es, float)[None, :]
    v = xs ** 2 + Es ** 2
    return -0.5 * (_LOG_2PI + np.log(v) + (xm - Em) ** 2 / v)


def _logsumexp(a: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
    """logsumexp along one axis, stable and tolerant of all-`-inf` slices."""
    m = np.max(a, axis=axis, keepdims=True)
    m_fin = np.where(np.isfinite(m), m, 0.0)          # all -inf slice -> shift by 0
    out = m_fin + np.log(np.sum(np.exp(a - m_fin), axis=axis, keepdims=True))
    out = np.where(np.isfinite(m), out, NEG_INF)      # restore -inf for empty slices
    return np.squeeze(out, axis=axis)


@dataclass
class AlignResult:
    path: npt.NDArray[np.int_]  # half-step state per observed step, length N
    s: int                      # start half-step (path[0])
    e: int                      # end half-step (path[-1])
    total_score: float


@dataclass
class FBResult:
    logL: float                          # read log-likelihood (free ends)
    W: npt.NDArray[np.float64]           # posteriors P(state h | read), shape [N, L]


def viterbi(
    S: npt.NDArray[np.float64],
    logT: npt.NDArray[np.float64],
) -> AlignResult:
    """Semi-global monotone MAP path. Free start (uniform init) + free end (max over
    final-row states). Maximizes sum of emission + transition log-scores.
    """
    N, L = S.shape
    V = np.full((N, L), NEG_INF)
    bp = np.zeros((N, L), dtype=np.int64)

    V[0] = S[0]  # free start: uniform entry, constant dropped
    for t in range(1, N):
        # M[i, j] = V[t-1, i] + logT[i, j]
        M = V[t - 1][:, None] + logT
        bp[t] = np.argmax(M, axis=0)
        V[t] = S[t] + M[bp[t], np.arange(L)]

    e = int(np.argmax(V[N - 1]))  # free end
    path = np.empty(N, dtype=np.int64)
    path[N - 1] = e
    for t in range(N - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]

    return AlignResult(
        path=path,
        s=int(path[0]),
        e=int(path[-1]),
        total_score=float(V[N - 1, e]),
    )


def forward_backward(
    S: npt.NDArray[np.float64],
    logT: npt.NDArray[np.float64],
) -> FBResult:
    """Semi-global forward-backward over the same emission + transition grid as
    `viterbi` (logsumexp in place of max). Free start (uniform init) + free end
    (any state may terminate).

        alpha[t, j] = S[t, j] + logsumexp_i( alpha[t-1, i] + logT[i, j] )
        beta[t, i]  = logsumexp_j( logT[i, j] + S[t+1, j] + beta[t+1, j] )
        logL        = logsumexp_j alpha[N-1, j]
        W[t]        = exp( alpha[t] + beta[t] - logL )

    Returns logL (Component 2's read likelihood) and posteriors W (soft QC + the
    ~15% degenerate half-steps that Viterbi resolves only by the transition prior).
    """
    N, L = S.shape

    alpha = np.full((N, L), NEG_INF)
    alpha[0] = S[0]  # free start: uniform entry (matches viterbi init)
    for t in range(1, N):
        M = alpha[t - 1][:, None] + logT          # [i, j]
        alpha[t] = S[t] + _logsumexp(M, axis=0)

    beta = np.full((N, L), NEG_INF)
    beta[N - 1] = 0.0  # free end: any state may terminate
    for t in range(N - 2, -1, -1):
        M = logT + (S[t + 1] + beta[t + 1])[None, :]  # [i, j]
        beta[t] = _logsumexp(M, axis=1)

    logL = float(_logsumexp(alpha[N - 1], axis=0))
    W = np.exp(alpha + beta - logL)
    return FBResult(logL=logL, W=W)
