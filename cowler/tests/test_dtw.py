"""DTW recursion + cost builders (`align/dtw.py`, `align/cost.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from cowler.align.cost import cost_abs, cost_gaussian, cost_l2
from cowler.align.dtw import dtw, dtw_pairwise, warp_to


def test_identical_sequences_zero_cost_diagonal_path():
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    r = dtw(cost_abs(a, a))
    assert r.dist == pytest.approx(0.0)
    assert np.array_equal(r.path[:, 0], r.path[:, 1])
    assert (r.j_start, r.j_end) == (0, a.size - 1)


def test_duplicated_element_absorbed_by_singleton_move():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 1.0, 2.0, 3.0])  # one element held
    r = dtw(cost_abs(a, b))
    assert r.dist == pytest.approx(0.0)
    assert len(r.path) == b.size          # one extra move vs the diagonal
    assert r.j_end == b.size - 1


def test_open_ends_recover_a_sub_span():
    q = np.array([3.0, 4.0, 5.0])
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    r = dtw(cost_abs(q, t), open_begin=True, open_end=True)
    assert (r.j_start, r.j_end) == (3, 5)
    assert r.dist == pytest.approx(0.0)


def test_closed_ends_on_a_sub_span_are_garbage_not_degraded():
    """Closed DTW must consume the whole target -> invented warping at the ends."""
    q = np.array([3.0, 4.0, 5.0])
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    closed = dtw(cost_abs(q, t), max_run=None)
    open_ = dtw(cost_abs(q, t), open_begin=True, open_end=True, max_run=None)
    assert closed.j_start == 0 and closed.j_end == t.size - 1
    assert closed.dist > open_.dist


def test_max_run_caps_consecutive_singleton_moves():
    a = np.array([0.0, 5.0])
    b = np.concatenate([np.zeros(6), np.full(6, 5.0)])
    r = dtw(cost_abs(a, b), max_run=None)
    assert _longest_run(r.path) > 3
    with pytest.raises(ValueError):        # 12 target steps, 2 observations
        dtw(cost_abs(a, b), max_run=3)


def test_symmetric2_normalizer_is_exact():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20)
    b = rng.normal(size=25)
    r = dtw(cost_abs(a, b), step="symmetric2", max_run=None)
    assert r.norm_dist == pytest.approx(r.dist / (a.size + b.size))


def test_gaussian_cost_downweights_noisy_steps():
    """A step with large std must cost less than the same mismatch when sharp."""
    am, bm = np.array([0.0]), np.array([2.0])
    sharp = cost_gaussian(am, np.array([0.1]), bm, np.array([0.1]))
    noisy = cost_gaussian(am, np.array([2.0]), bm, np.array([2.0]))
    assert noisy[0, 0] < sharp[0, 0]


def test_cost_builders_shapes_and_signs():
    a, b = np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])
    assert cost_l2(a, b).shape == (2, 3)
    assert cost_abs(a, b).shape == (2, 3)
    assert cost_l2(a, b)[0, 2] == pytest.approx(4.0)
    assert cost_abs(a, b)[0, 2] == pytest.approx(2.0)


def test_pairwise_matrix_symmetric_zero_diagonal():
    rng = np.random.default_rng(1)
    reads = [(rng.normal(size=n), np.full(n, 0.2)) for n in (12, 15, 13)]
    D = dtw_pairwise(reads, max_run=3)
    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)


def test_pairwise_is_offset_and_scale_invariant_when_normalized():
    rng = np.random.default_rng(2)
    m = rng.normal(size=30)
    scaled = (3.0 * m + 10.0, np.full(30, 0.6))
    D = dtw_pairwise([(m, np.full(30, 0.2)), scaled], max_run=3)
    same = dtw_pairwise([(m, np.full(30, 0.2)), (m, np.full(30, 0.2))], max_run=3)
    assert D[0, 1] == pytest.approx(same[0, 1], abs=1e-6)


def test_warp_to_averages_onto_target_axis_and_flags_gaps():
    path = np.array([[0, 0], [1, 0], [2, 1]])
    out = warp_to(path, np.array([1.0, 3.0, 5.0]), n_target=3)
    assert out[0] == pytest.approx(2.0)   # two observations mapped to j=0
    assert out[1] == pytest.approx(5.0)
    assert np.isnan(out[2])               # never visited -> NaN, not 0


def _longest_run(path: np.ndarray) -> int:
    best = cur = 0
    prev = None
    for step in np.diff(path, axis=0):
        kind = tuple(step)
        if kind == (1, 1):
            cur, prev = 0, None
            continue
        cur = cur + 1 if kind == prev else 1
        prev = kind
        best = max(best, cur)
    return best
