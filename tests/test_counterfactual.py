import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import counterfactual as cf


@pytest.fixture
def group():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "balls":         rng.integers(0, 4, n),
        "strikes":       rng.integers(0, 3, n),
        "base_state":    rng.choice(["empty", "on1", "risp"], n),
        "outs_when_up":  rng.integers(0, 3, n),
        "pitcher_throws": rng.choice(["L", "R"], n),
        "pitch_group":   rng.choice(["FB", "brk", "off"], n),
    })


def test_build_design_orders_controls_before_axes(group):
    loc = np.ones((len(group), 6))
    X, slices = cf.build_design(group, loc)
    assert set(slices) == {"count", "gamestate", "platoon"}
    first_axis_col = min(s.start for s in slices.values())
    # location (6) + pitch_group dummies sit strictly before every axis block
    assert first_axis_col >= 6
    assert X.shape[1] == max(s.stop for s in slices.values())


def test_desituate_replaces_only_named_axis(group):
    loc = np.ones((len(group), 6))
    X, slices = cf.build_design(group, loc)
    out = cf.desituate(X, slices, ["count"])
    count_block = out[:, slices["count"]]
    # every row of the count block is now the column mean
    assert np.allclose(count_block, count_block[0])
    assert np.allclose(count_block[0], X[:, slices["count"]].mean(axis=0))
    # other blocks untouched
    assert np.array_equal(out[:, slices["platoon"]], X[:, slices["platoon"]])
    assert np.array_equal(out[:, :6], X[:, :6])


def test_desituate_preserves_column_means(group):
    """Setting dummies to their mean must not move the average swing."""
    loc = np.ones((len(group), 6))
    X, slices = cf.build_design(group, loc)
    out = cf.desituate(X, slices, ["count", "gamestate", "platoon"])
    assert np.allclose(out.mean(axis=0), X.mean(axis=0))


def test_crossfit_covers_every_row_exactly_once():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    Y = rng.normal(size=(100, 2))
    fits = cf.crossfit_shapes(X, Y, n_folds=5, seed=7)
    covered = np.concatenate([test for test, _ in fits])
    assert sorted(covered) == list(range(100))
    assert all(coefs.shape == (4, 2) for _, coefs in fits)


def test_predict_oof_matches_manual_assembly():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    Y = rng.normal(size=(60, 2))
    fits = cf.crossfit_shapes(X, Y, n_folds=3, seed=7)
    got = cf.predict_oof(X, fits, 2)
    want = np.empty((60, 2))
    for test, coefs in fits:
        want[test] = X[test] @ coefs
    assert np.allclose(got, want)


def test_blend_endpoints_and_linearity():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose(cf.blend(a, b, 0.0), a)
    assert np.allclose(cf.blend(a, b, 1.0), b)
    assert np.allclose(cf.blend(a, b, 0.5), (a + b) / 2)
    assert np.allclose(cf.blend(a, b, 2.0), 2 * b - a)


def test_envelope_and_fraction_outside():
    obs = np.tile(np.arange(100.0), (2, 1)).T  # two identical features, 0..99
    env = cf.envelope(obs)
    assert np.allclose(cf.fraction_outside(obs, env), 0.02, atol=0.01)
    way_out = np.full_like(obs, 1000.0)
    assert cf.fraction_outside(way_out, env) == 1.0


def test_admissible_alphas_excludes_extrapolation():
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(300, 3))
    env = cf.envelope(obs)
    shape_cf = np.zeros((300, 3))
    shape_actual = np.full((300, 3), 2.0)  # already near the edge
    ok = cf.admissible_alphas(shape_cf, shape_actual, env)
    assert 0.0 in ok
    assert 2.0 not in ok
    assert ok == sorted(ok)


def test_alpha_grid_extends_past_the_turn():
    """The value curve peaks near alpha=4; a grid stopping at 2.0 reports a truncation."""
    assert max(cf.ALPHA_GRID) >= 6.0
    assert cf.ALPHA_GRID == sorted(cf.ALPHA_GRID)


def test_axis_contributions_sum_to_the_total(group):
    """axis_blend relies on exact additivity of the per-axis contributions."""
    rng = np.random.default_rng(4)
    loc = rng.normal(size=(len(group), 6))
    X, slices = cf.build_design(group, loc)
    fits = cf.crossfit_shapes(X, rng.normal(size=(len(group), 5)))
    actual = cf.predict_oof(X, fits, 5)
    base = cf.predict_oof(cf.desituate(X, slices, list(cf.AXES)), fits, 5)

    total = np.zeros_like(actual)
    for axis in cf.AXES:
        others = [a for a in cf.AXES if a != axis]
        total += cf.predict_oof(cf.desituate(X, slices, others), fits, 5) - base
    assert np.allclose(total, actual - base)


def test_axis_blend_moves_only_its_own_axis():
    actual = np.array([[10.0, 20.0]])
    base = np.array([[4.0, 6.0]])
    axis = np.array([[6.0, 10.0]])  # contributes +2, +4
    assert np.allclose(cf.axis_blend(actual, axis, base, 1.0), actual)
    assert np.allclose(cf.axis_blend(actual, axis, base, 0.0), [[8.0, 16.0]])
    assert np.allclose(cf.axis_blend(actual, axis, base, 2.0), [[12.0, 24.0]])


def test_policy_alphas_caps_scaled_displacement():
    ok = cf.policy_alphas(0.4, 0.6, grid=[0.0, 0.5, 1.0, 1.5, 2.0])
    assert ok == [0.0, 0.5, 1.0, 1.5]  # 1.5 * 0.4 = 0.6 exactly, admitted
    # an already-high modulator is told he is close to the ceiling
    assert max(cf.policy_alphas(0.58, 0.6, grid=[0.0, 0.5, 1.0, 1.5, 2.0])) == 1.0
    # a degenerate unit is not silently capped to nothing
    assert cf.policy_alphas(0.0, 0.6, grid=[0.0, 1.0]) == [0.0, 1.0]


def test_cell_labels_cover_all_three_axes():
    frame = pd.DataFrame({
        "strikes":        [0, 1, 2, 2],
        "base_state":     ["empty", "on1", "risp", "risp"],
        "pitcher_throws": ["R", "L", "R", "L"],
        "batter_stand":   ["R", "R", "L", "L"],
    })
    cells = cf.cell_labels(frame)
    assert set(cells) == {"count", "gamestate", "platoon"}
    assert list(cells["count"]) == ["0 strikes", "1 strike", "2 strikes", "2 strikes"]
    assert list(cells["gamestate"]) == ["empty", "on1", "risp", "risp"]
    assert list(cells["platoon"]) == ["same-hand", "opp-hand", "opp-hand", "same-hand"]
