"""Physical half-step transition matrix for the step-level aligner.

Net half-step jump distribution pk(k) between consecutive observed steps, derived
from enzyme stepping parameters. Built once per molecule length and cached
(depends only on L + params, not on the read).

Jump pmf (half-step units), computed by direct truncated series -- NOT the
hypergeom closed form in `pstepToRefT` (that needs the Symbolic Toolbox and does
not port cleanly; the series is the same quantity, evaluated numerically):

    pf = pstep, pb = 1 - pstep
    pk(0)  = phold + (1-phold) * <walk returns to 0>
    pk(k)  = (1-phold) * sum_{n >= |k|, n == k (mod 2)}
                 pmiss**(n-1) * (1-pmiss)                    # 1 observed + (n-1) missed steps
               * C(n, (n+k)/2) * pf**((n+k)/2) * pb**((n-k)/2)
    truncate n when the geometric tail pmiss**(n-1) is negligible. Row-normalize.

Semi-global free ends: prepend a start/end pseudo-state (the "0" state in the
original). Free start = uniform entry into any state; free end = any state may
terminate. With fixstart, entry must instead pay forward-jump penalties to start
past state 0. Never force start at state 0 / end at state L-1.

Public API:
    jump_pmf(pstep, phold, pmiss, kmax)                 -- pk over [-kmax .. kmax]
    build_transition(L, pstep, phold, pmiss, *, fixstart=False)
                                                        -- -> log T[(L+1), (L+1)] (cached)
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np
import numpy.typing as npt

NEG_INF = -np.inf


def jump_pmf(
    pstep: float = 0.9,
    phold: float = 0.05,
    pmiss: float = 0.1,
    kmax: int = 10,
    tol: float = 1e-12,
) -> npt.NDArray[np.float64]:
    """Net half-step jump pmf over k in [-kmax .. kmax] (index k -> k + kmax).

    pk(0)  = phold + (1-phold) * series(0)
    pk(k)  = (1-phold) * series(k),   k != 0
    series(k) = sum_{n>=|k|, n==k (mod 2)} pmiss**(n-1) * (1-pmiss)
                  * C(n, (n+k)/2) * pf**((n+k)/2) * pb**((n-k)/2)
    Truncated when the geometric tail pmiss**(n-1) drops below tol. Row-normalized.
    """
    pf, pb = pstep, 1.0 - pstep
    ks = np.arange(-kmax, kmax + 1)
    pk = np.zeros(2 * kmax + 1)

    for ik, k in enumerate(ks):
        ak = abs(k)
        s = 0.0
        # n = total enzyme steps in the gap (1 observed + (n-1) missed), n >= 1,
        # sharing parity with k. A real step never nets 0, so k=0 starts at n=2
        # (a missed step there-and-back). The genuine stay/hold mass -- including
        # oversegmentation, where noise splits one dwell into two observed steps at
        # the same level -- is supplied solely by phold below (the tunable lever);
        # merges of two unresolvable steps are skips (k>=+2), carried by pmiss.
        n = ak if ak > 0 else 2
        while True:
            tail = pmiss ** (n - 1)
            if tail < tol:
                break
            half = (n + k) // 2
            if 0 <= half <= n:
                s += tail * (1.0 - pmiss) * comb(n, half) * pf ** half * pb ** (n - half)
            n += 2
            if n > 4 * kmax + 50:  # hard cap
                break
        pk[ik] = s

    pk *= (1.0 - phold)
    pk[kmax] += phold  # k = 0 hold mass
    pk /= pk.sum()
    return pk


@lru_cache(maxsize=32)
def build_transition(
    L: int,
    pstep: float = 0.9,
    phold: float = 0.05,
    pmiss: float = 0.1,
    kmax: int = 10,
) -> npt.NDArray[np.float64]:
    """Log transition matrix logT[L, L] over real half-step states (cached on args).

    logT[i, j] = log pk(j - i), jumps beyond +-kmax are -inf. Rows NOT renormalized
    after truncation (boundary states lose a little forward mass; negligible for L>>kmax).

    Free ends are handled by the decoder, NOT here: free start = uniform Viterbi init,
    free end = max over all states at the final step. So this is the L x L real-state
    block only -- no (L+1) start/end pseudo-state row/col (deliberate simplification of
    the documented API; never forces start h=0 / end h=L-1).
    """
    pk = jump_pmf(pstep, phold, pmiss, kmax)
    with np.errstate(divide="ignore"):
        logpk = np.log(pk)  # -inf where pk == 0

    logT = np.full((L, L), NEG_INF)
    i = np.arange(L)
    for k in range(-kmax, kmax + 1):
        j = i + k
        valid = (j >= 0) & (j < L)
        logT[i[valid], j[valid]] = logpk[k + kmax]
    return logT
