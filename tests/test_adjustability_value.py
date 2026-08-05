"""Guards the swing-shape maths that every published adjustment number rests on.

These test BEHAVIOUR of the public steps in src/adjustability_value.py — the properties the
design argument depends on — rather than the internals, so the file can be restructured
without the tests having to be rewritten alongside it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "uncommited"))
import adjustability_value as av
import pitch_controls as pc

LEAGUE_SD = np.ones(len(av.SHAPE))


@pytest.fixture
def swings():
    """One synthetic hitter with a REAL situational habit: he flattens with two strikes.

    A planted signal means a test that says "the situational move shrinks" is measuring
    something, rather than passing on noise.
    """
    rng = np.random.default_rng(0)
    n = 400
    strikes = rng.integers(0, 3, n)
    frame = pd.DataFrame({
        "batter_id":      1,
        "batter_stand":   "R",
        "balls":          rng.integers(0, 4, n),
        "strikes":        strikes,
        "base_state":     rng.choice(["empty", "on1", "risp"], n),
        "outs_when_up":   rng.integers(0, 3, n),
        "pitcher_throws": rng.choice(["L", "R"], n),
        "pitch_group":    rng.choice(["FB", "brk", "off"], n),
        "px":             rng.normal(size=n),
        "pz":             rng.normal(size=n),
    })
    for characteristic in pc.PITCH_CHARS:
        frame[characteristic] = rng.normal(size=n)
    for dial in av.SHAPE:
        frame[dial] = rng.normal(size=n)
    # The planted habit: swing path tilt drops by 2 units per strike.
    frame["swing_path_tilt"] = frame["swing_path_tilt"] - 2.0 * strikes
    return frame


@pytest.fixture
def hitter(swings):
    return av.learn_how_he_swings(swings, LEAGUE_SD)


def test_switching_the_situation_off_leaves_his_average_swing_alone(hitter):
    """The comparison swing must be mean-preserving, or the headline measures the wrong thing.

    Flattening the situation dummies to their column MEAN rather than to a reference category
    is what buys this: only situational variation moves, his average swing does not. It holds
    to within cross-fitting slop, since each fold applies its own coefficients to a design
    centred on the full sample.
    """
    assert np.allclose(hitter.with_situation_off.mean(axis=0),
                       hitter.as_he_swung_it.mean(axis=0), atol=0.05)


def test_each_axis_contribution_adds_up_to_the_whole(hitter):
    """Count + runners + hand must sum exactly to the total, or the decomposition is a fiction.

    Exact, not approximate: switching an axis off is linear in the design and the prediction
    is linear in that, and every reading here uses the same fold's coefficients.
    """
    total = np.zeros_like(hitter.as_he_swung_it)
    for axis in av.AXIS_NAMES:
        total += hitter.with_one_axis_live[axis] - hitter.with_situation_off
    assert np.allclose(total, hitter.as_he_swung_it - hitter.with_situation_off)


def test_swings_are_predicted_out_of_fold(hitter, swings):
    """Readings must not come from coefficients fit on the swing being read.

    The headline is a difference of two fitted values, which is exactly where in-sample fits
    would let ~20 situation dummies manufacture situational signal out of noise. If this ever
    starts matching the in-sample fit, cross-fitting has been lost.
    """
    design = np.column_stack([
        np.ones(len(swings)), swings.px, swings.pz,
        swings.px ** 2, swings.pz ** 2, swings.px * swings.pz,
        pd.get_dummies(swings[["pitch_group"]].astype(str), drop_first=True).to_numpy(float),
        pc.control_matrix(swings),
        pd.get_dummies(swings[av.SITUATION_COLS].astype(str), drop_first=True).to_numpy(float),
    ])
    measured = swings[av.SHAPE].to_numpy(float)
    in_sample = design @ np.linalg.lstsq(design, measured, rcond=None)[0]
    assert not np.allclose(hitter.as_he_swung_it, in_sample, atol=1e-6)


def test_the_portable_block_reads_deviations_not_levels(hitter):
    """The situation row must be centred, or lending a block would import the lender's swing.

    A centred row makes the block a map from "how unusual is this situation" to a
    displacement, which is the only reason it can be applied to a different hitter.
    """
    assert np.allclose(hitter.situation_row.mean(axis=0), 0.0, atol=1e-12)


def test_the_planted_habit_is_recovered(hitter):
    """The fit should find the two-strike flattening that the fixture put there."""
    tilt = av.SHAPE.index("swing_path_tilt")
    moved = hitter.as_he_swung_it[:, tilt] - hitter.with_situation_off[:, tilt]
    two_strike = hitter.swings["strikes"].to_numpy() == 2
    assert moved[two_strike].mean() < moved[~two_strike].mean() - 1.0


def test_shuffling_the_situation_destroys_the_habit(swings):
    """The placebo must actually break the situation-shape link it claims to break."""
    real = av.learn_how_he_swings(swings, LEAGUE_SD)
    fake = av.learn_how_he_swings(swings, LEAGUE_SD, shuffle_seed=11)
    assert np.abs(fake.situational_move).mean() < np.abs(real.situational_move).mean() / 2


def test_fraction_outside_flags_swings_he_has_never_made(hitter):
    """The envelope rail keeps the run-value model from pricing unsupported shapes."""
    # The envelope is the 1st-99th percentile, so ~2% of his own swings sit outside it.
    assert hitter.fraction_outside(hitter.measured) == pytest.approx(0.02, abs=0.01)
    assert hitter.fraction_outside(np.full_like(hitter.measured, 1000.0)) == 1.0


def test_alpha_grid_extends_past_the_turn():
    """The value curve peaks near alpha=4; a grid stopping at 2.0 reports a truncation."""
    assert max(av.ALPHA_GRID) >= 6.0
    assert av.ALPHA_GRID == sorted(av.ALPHA_GRID)


def test_affordable_alphas_caps_scaled_displacement():
    grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    assert av.affordable_alphas(0.4, 0.6, grid=grid) == [0.0, 0.5, 1.0, 1.5]  # 1.5*0.4 = 0.6
    # an already-high modulator is told he is close to the ceiling
    assert max(av.affordable_alphas(0.58, 0.6, grid=grid)) == 1.0
    # a degenerate unit is not silently capped to nothing
    assert av.affordable_alphas(0.0, 0.6, grid=[0.0, 1.0]) == [0.0, 1.0]


def test_situation_cells_cover_all_three_axes():
    frame = pd.DataFrame({
        "strikes":        [0, 1, 2, 2],
        "base_state":     ["empty", "on1", "risp", "risp"],
        "pitcher_throws": ["R", "L", "R", "L"],
        "batter_stand":   ["R", "R", "L", "L"],
    })
    cells = av.FittedSwings(
        swings=frame, measured=None, as_he_swung_it=None, with_situation_off=None,
        with_one_axis_live=None, situation_row=None, axis_rows=None,
        situational_move=None, how_he_adjusts=None,
    ).situation_cells()
    assert set(cells) == {"count", "gamestate", "platoon"}
    assert list(cells["count"]) == ["0 strikes", "1 strike", "2 strikes", "2 strikes"]
    assert list(cells["gamestate"]) == ["empty", "on1", "risp", "risp"]
    assert list(cells["platoon"]) == ["same-hand", "opp-hand", "opp-hand", "same-hand"]
