"""Scores whether a hitter's situational adjustments are actually worth runs. On every pitch
he saw, we swap in the way a randomly picked other hitter would have adjusted, re-grade both
swings, and add up the difference over a season.

    runs_total = sum over swings of [ xRV(his adjustment) - xRV(a replacement hitter's) ]

HOW TO READ THIS FILE. The functions below are in the order they run, and each is one step
of the procedure. Start at main() if you want the loop; start here if you want the argument:

  1. load_data            the three xRV models, the swings, the frozen league scale.
  2. learn_how_he_swings  ONE regression per hitter: his swing shape explained by where the
                          pitch was, what it was doing, and the situation. Reading that fit
                          back gives his swing four ways — as he swung it, with the situation
                          switched off, with one situation axis left live, and as a portable
                          "how he adjusts" block that can be handed to another hitter.
  3. runs_for             prices any swing shape in runs, on the pitches he actually saw.
  4. value_against_        the headline. His swing vs. a random peer's adjusting habits
     replacement          applied to his pitches, then split into count / runners / hand.
  5. how_much_should_      prescription part 1: would dialling his existing adjustment up or
     he_adjust            down be worth runs? (Ships cohort-level only — split-half r = 0.30.)
  6. which_dial_should_    prescription part 2: which of the five shape dials to move, and
     he_move              where. (Vertical attack angle is the reliable one, r = 0.68.)
  7. write_and_report     joins the context tables, writes both parquets, prints the summary.

Terms used throughout. A "unit" is one batter in one stance, since a switch hitter's two
sides are different movements. "Situation" means count, runners/outs, and pitcher handedness
— the three things a hitter may respond to on purpose. Where the pitch was and what it was
doing are CONTROLS: they are held at their observed values everywhere and never switched
off, because a hitter cannot choose his swing before he identifies the pitch.
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "uncommited"))

import db                                                            # noqa: E402
import pitch_controls as pc                                          # noqa: E402
import xRV_model as xrv                                              # noqa: E402
from adjustability import KEY, MIN_SWINGS, SEASONS                   # noqa: E402
from adjustability import add_context, location_design               # noqa: E402

# The three things a hitter may respond to on purpose, and the columns that encode each.
# These, and only these, get switched off to build the comparison swing.
SITUATION_AXES = {
    "count":     ["balls", "strikes"],
    "gamestate": ["base_state", "outs_when_up"],
    "platoon":   ["pitcher_throws"],
}
AXIS_NAMES = list(SITUATION_AXES)
SITUATION_COLS = [col for cols in SITUATION_AXES.values() for col in cols]

# Fixed league-wide category sets for the portable "how he adjusts" block. Per-unit dummies
# would give a different column set to any hitter missing a category, and those blocks could
# then not be swapped between hitters.
SITUATION_LEVELS = {
    "balls":          ["0", "1", "2", "3"],
    "strikes":        ["0", "1", "2"],
    "base_state":     ["empty", "on1", "risp"],
    "outs_when_up":   ["0", "1", "2"],
    "pitcher_throws": ["L", "R"],
}

SHAPE = xrv.SHAPE_FEATURES
N_SEASONS = len(SEASONS)
POLICY_REF = ROOT / "src" / "uncommited" / "adjustability_policy_reference.json"
LOAD_COLS = KEY + [
    "play_id", "game_pk", "game_year", "batter_full_name", "balls", "strikes", "outs_when_up",
    "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type", "pitcher_throws",
    "on_1b_id", "on_2b_id", "on_3b_id", "delta_run_exp", "woba",
] + SHAPE

N_FOLDS = 5
SEED = 7
N_REPLACEMENTS = 10
# Fine near alpha=1, where any realistic prescription lives; coarse past 3.0, where the only
# job is to locate the turn. The old grid stopped at 2.0, below the peak, which is why 86%
# of units pegged at its ceiling and the recommendation was a grid artifact.
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
AXIS_ALPHA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
ENVELOPE_LO, ENVELOPE_HI = 1.0, 99.0
MAX_OUTSIDE = 0.05
POLICY_QUANTILE = 0.90
GRAD_STEP_SD = 0.1
POLICY_STEP_SD = 0.25
MIN_CELL_SWINGS = 25


@dataclass
class FittedSwings:
    """One hitter-stance's swings, read four ways out of a single fitted regression.

    Everything here is in raw dial units (degrees, feet, mph) except `situational_move` and
    `how_he_adjusts`, which are in league-SD units so they are comparable between hitters
    whose dials have different ranges.
    """
    swings: pd.DataFrame                    # his pitches, one row per competitive swing
    measured: np.ndarray                    # the shapes actually recorded by the cameras
    as_he_swung_it: np.ndarray              # the fit's read of his swing, situation included
    with_situation_off: np.ndarray          # same fit, all three situation axes flattened
    with_one_axis_live: dict                # axis -> only that axis left live
    situation_row: np.ndarray               # mean-centered situation dummies, league levels
    axis_rows: dict                         # axis -> its row slice of situation_row
    situational_move: np.ndarray            # how far the situation moves him, in SD units
    how_he_adjusts: np.ndarray              # the portable block: situation -> displacement
    envelope: tuple = field(default=None)   # per-dial (1st, 99th) percentile of `measured`

    def fraction_outside(self, shape):
        """Share of (swing, dial) pairs in `shape` outside anything he has ever actually done.

        A rail against asking the run-value model to price a swing it has no support for.
        """
        low, high = self.envelope
        return float(((shape < low) | (shape > high)).mean())

    def situation_cells(self):
        """Readable situation buckets for the prescription, one label per swing per axis.

        Deliberately coarser than the axes used to switch the situation off: prescriptions
        are read one cell at a time, and the full balls x strikes x bases x outs cross is
        far too thin to estimate a per-cell gradient.
        """
        strikes = self.swings["strikes"].to_numpy()
        same_hand = (self.swings["pitcher_throws"].to_numpy()
                     == self.swings["batter_stand"].to_numpy())
        return {
            "count": np.where(strikes >= 2, "2 strikes",
                              np.where(strikes == 1, "1 strike", "0 strikes")),
            "gamestate": self.swings["base_state"].to_numpy().astype(str),
            "platoon": np.where(same_hand, "same-hand", "opp-hand"),
        }


# ----------------------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------------------

def load_data(with_pitch_chars=True):
    """The three persisted xRV boosters, the run-value tables, the swings, and the scale.

    pitch_type is cast to a category ONCE here, on the full frame. xrv.build_features calls
    astype("category"), which infers its categories from whatever it is handed — scoring
    per-unit frames later would give each unit its own integer codes and silently misalign
    every prediction. The --verify flag in main() is the guard on exactly this.
    """
    models = {}
    for name, model_class in [("p_bip", XGBClassifier), ("p_foul", XGBClassifier),
                              ("v_bip", XGBRegressor)]:
        model = model_class(enable_categorical=True)
        model.load_model(DATA / "xrv_models" / f"{name}.json")
        models[name] = model

    swings = pd.read_parquet(DATA / "swings_model.parquet", columns=LOAD_COLS)
    swings = swings[swings["game_year"].isin(SEASONS)]
    swings["pitch_type"] = swings["pitch_type"].astype("category")
    swings = add_context(swings).dropna(subset=SHAPE)
    if with_pitch_chars:
        swings = pc.join_pitch_chars(swings)
    swings_per_unit = swings.groupby(KEY, observed=True)[SHAPE[0]].transform("size")
    swings = swings[swings_per_unit >= MIN_SWINGS].reset_index(drop=True)

    # Frozen league scale and displacement cap, so prescriptions stay comparable as seasons
    # are added. League aggregates only, no PII. Delete the JSON to re-peg.
    if POLICY_REF.exists():
        reference = json.loads(POLICY_REF.read_text())
    else:
        reference = {
            "seasons": SEASONS,
            "shape_features": SHAPE,
            "shape_sd": swings[SHAPE].std(ddof=0).astype(float).to_dict(),
            "policy_quantile": POLICY_QUANTILE,
            "displacement_cap": None,
        }
    return models, xrv.load_run_value_tables(), swings, reference


# ----------------------------------------------------------------------------------------
# 2. Learn one hitter's swing rule
# ----------------------------------------------------------------------------------------

def learn_how_he_swings(swings, league_sd, shuffle_seed=None):
    """Fit one hitter's swing shape, then read it back with the situation on and off.

    The regression is `shape ~ where the pitch was + what it was doing + count + runners
    + pitcher hand`, fit per hitter. Pitch location and characteristics are controls and
    stay at their observed values in every reading; without them the situation dummies
    absorb count-correlated velocity and break, and the hitter's mechanical reaction to
    harder stuff gets scored as a deliberate adjustment.

    Every reading is CROSS-FITTED — predicted with coefficients estimated on other folds.
    The headline is a difference between two of these readings, so in-sample fits would let
    ~20 situation dummies manufacture situational signal out of noise, and that noise would
    not cancel in the difference unless both readings use the same fold.

    `shuffle_seed` permutes the situation columns within the hitter, destroying any real
    situation-shape relationship while preserving the marginals. That is the placebo.
    """
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        swings = swings.copy()
        swings[SITUATION_COLS] = swings[SITUATION_COLS].to_numpy()[rng.permutation(len(swings))]

    # Design matrix, controls first so the situation blocks sit in known column slices.
    blocks = [location_design(swings)]
    width = blocks[0].shape[1]
    for control in (pd.get_dummies(swings[["pitch_group"]].astype(str),
                                   drop_first=True).to_numpy(float),
                    pc.control_matrix(swings)):
        blocks.append(control)
        width += control.shape[1]
    axis_slices = {}
    for axis, cols in SITUATION_AXES.items():
        dummies = pd.get_dummies(swings[cols].astype(str), drop_first=True).to_numpy(float)
        axis_slices[axis] = slice(width, width + dummies.shape[1])
        blocks.append(dummies)
        width += dummies.shape[1]
    design = np.column_stack(blocks)

    measured = swings[SHAPE].to_numpy(float)
    splitter = KFold(n_splits=min(N_FOLDS, len(design)), shuffle=True, random_state=SEED)
    folds = [(test, np.linalg.lstsq(design[train], measured[train], rcond=None)[0])
             for train, test in splitter.split(design)]

    def read_back(matrix):
        """Predict with each fold's own coefficients, so every row stays out-of-fold."""
        out = np.empty((len(matrix), len(SHAPE)))
        for test, coefs in folds:
            out[test] = matrix[test] @ coefs
        return out

    def switch_off(axes_to_flatten):
        """Copy of the design with those axes' dummies flattened to their column means.

        Setting them to the MEAN rather than to a reference category is what makes this
        mean-preserving: his average swing is unchanged and only situational VARIATION moves.
        """
        out = design.copy()
        for axis in axes_to_flatten:
            out[:, axis_slices[axis]] = design[:, axis_slices[axis]].mean(axis=0)
        return out

    as_he_swung_it = read_back(design)
    with_situation_off = read_back(switch_off(AXIS_NAMES))
    with_one_axis_live = {}
    for axis in AXIS_NAMES:
        others = [a for a in AXIS_NAMES if a != axis]
        with_one_axis_live[axis] = read_back(switch_off(others))

    # The portable block. Centering the dummies is what makes it transferable: it then maps
    # "how unusual is this situation" to a displacement, describing how a hitter DEVIATES by
    # situation rather than what his average swing looks like. Handing it to someone else
    # therefore lends his adjusting habits without importing his mechanics.
    situation_dummies, axis_rows, width = [], {}, 0
    for axis, cols in SITUATION_AXES.items():
        start = width
        for col in cols:
            values = swings[col].astype(str).to_numpy()
            levels = SITUATION_LEVELS[col]
            situation_dummies.append(
                np.stack([(values == level).astype(float) for level in levels], axis=1))
            width += len(levels)
        axis_rows[axis] = slice(start, width)
    situation_row = np.column_stack(situation_dummies)
    situation_row = situation_row - situation_row.mean(axis=0, keepdims=True)

    situational_move = (as_he_swung_it - with_situation_off) / league_sd
    # Rank-deficient by construction — each column group's dummies sum to a constant — so
    # lstsq's minimum-norm solution is the intended one.
    how_he_adjusts, *_ = np.linalg.lstsq(situation_row, situational_move, rcond=None)

    return FittedSwings(
        swings=swings,
        measured=measured,
        as_he_swung_it=as_he_swung_it,
        with_situation_off=with_situation_off,
        with_one_axis_live=with_one_axis_live,
        situation_row=situation_row,
        axis_rows=axis_rows,
        situational_move=situational_move,
        how_he_adjusts=how_he_adjusts,
        envelope=(np.percentile(measured, ENVELOPE_LO, axis=0),
                  np.percentile(measured, ENVELOPE_HI, axis=0)),
    )


# ----------------------------------------------------------------------------------------
# 3. Price a swing shape in runs
# ----------------------------------------------------------------------------------------

def runs_for(models, run_value_tables, swings, shape):
    """Expected run value PER SWING if he had made `shape` on each of these same pitches.

    Uses assemble_xrv, not the count-neutral variant, so two-strike stakes stay live: the
    mechanism under test is whether compressing the swing buys enough contact to pay for the
    strikeout risk, and that risk only exists if the count-aware run values are in play.
    """
    frame = swings.copy()
    for dial_index, feature in enumerate(SHAPE):
        frame[feature] = shape[:, dial_index]
    enriched = xrv.build_features(frame)
    prob_bip  = models["p_bip"].predict_proba(enriched[xrv.FEATURES])[:, 1]
    prob_foul = models["p_foul"].predict_proba(enriched[xrv.FEATURES])[:, 1]
    value_bip = models["v_bip"].predict(enriched[xrv.FEATURES])
    return xrv.assemble_xrv(enriched, prob_bip, prob_foul, value_bip, run_value_tables)


# ----------------------------------------------------------------------------------------
# 4. The headline: his adjusting against a replacement hitter's
# ----------------------------------------------------------------------------------------

def value_against_replacement(models, run_value_tables, hitter, league_sd,
                              everyones_blocks=None, index=None,
                              n_replacements=N_REPLACEMENTS):
    """Season runs his adjusting is worth against a randomly drawn peer's.

    "Replacement" here means a randomly drawn hitter from this same cohort, NOT WAR's
    replacement level — the bar is an average big-league adjuster, not a fringe one. That
    choice is the whole design. Against a frozen swing instead, 96% of hitters "gain" and
    the number is mostly playing time; against a POOLED league policy only 25% clear the
    bar, because a smoothed average wins by being smooth rather than by being right. A
    random individual is precision-matched — his fitted block carries the same estimation
    noise the hitter's own does — so the result is two-sided by construction.

    Averaging over draws is what makes the benchmark "a typical peer" rather than one
    arbitrary hitter. Passing `everyones_blocks=None` returns only the vs-frozen-swing
    quantity, which is all the placebo and split-half checks need.
    """
    swings = hitter.swings
    xrv_situation_off = runs_for(models, run_value_tables, swings, hitter.with_situation_off)
    xrv_actual = runs_for(models, run_value_tables, swings, hitter.as_he_swung_it)

    record = {
        "batter_id":    swings["batter_id"].iloc[0],
        "batter_stand": swings["batter_stand"].iloc[0],
        "n_swings":     len(swings),
        "runs_vs_desituated": float((xrv_actual - xrv_situation_off).sum() / N_SEASONS),
        "displacement_sd":    float(np.linalg.norm(hitter.situational_move, axis=1).mean()),
        "xrv_actual_mean":    float(xrv_actual.mean()),
        "xrv_cf_mean":        float(xrv_situation_off.mean()),
    }
    if everyones_blocks is None or not n_replacements:
        return record

    rng = np.random.default_rng(SEED + index)
    other_hitters = [j for j in range(len(everyones_blocks)) if j != index]
    n_draws = min(n_replacements, len(everyones_blocks) - 1)
    drawn = rng.choice(other_hitters, size=n_draws, replace=False)

    replacement_xrv = np.zeros(len(swings))
    replacement_xrv_by_axis = {axis: np.zeros(len(swings)) for axis in AXIS_NAMES}
    for j in drawn:
        # A block returns a displacement in league-SD units, so scale it back to raw dial
        # units before adding it onto his de-situated swing.
        replacement_xrv += runs_for(
            models, run_value_tables, swings,
            hitter.with_situation_off
            + (hitter.situation_row @ everyones_blocks[j]) * league_sd)
        for axis in AXIS_NAMES:
            # His own block with only this one axis's rows taken from the replacement.
            swapped = hitter.how_he_adjusts.copy()
            swapped[hitter.axis_rows[axis]] = everyones_blocks[j][hitter.axis_rows[axis]]
            replacement_xrv_by_axis[axis] += runs_for(
                models, run_value_tables, swings,
                hitter.with_situation_off + (hitter.situation_row @ swapped) * league_sd)

    is_two_strike = swings["strikes"].to_numpy() == 2
    per_swing_delta = xrv_actual - replacement_xrv / len(drawn)
    record["n_replacements"] = len(drawn)
    record["runs_total"] = float(per_swing_delta.sum() / N_SEASONS)
    record["runs_total_2k"] = float(per_swing_delta[is_two_strike].sum() / N_SEASONS)
    record["runs_per_swing"] = float(per_swing_delta.mean())

    # Each axis on its own, then the cross-term by subtraction so the identity closes exactly.
    axis_sum = 0.0
    for axis in AXIS_NAMES:
        axis_delta = xrv_actual - replacement_xrv_by_axis[axis] / len(drawn)
        record[f"runs_{axis}"] = float(axis_delta.sum() / N_SEASONS)
        axis_sum += record[f"runs_{axis}"]
    record["runs_interaction"] = record["runs_total"] - axis_sum
    return record


# ----------------------------------------------------------------------------------------
# 5. Prescription, part 1: how much should he adjust?
# ----------------------------------------------------------------------------------------

def how_much_should_he_adjust(models, run_value_tables, hitter):
    """Rescale the adjustment he already makes and see what it is worth.

    alpha = 1 is what he does now; alpha = 0 is not adjusting at all; alpha = 2 is adjusting
    twice as hard in the same direction. The shape model is linear in the situation dummies,
    so this interpolation is exact and no refit is needed per alpha.

    Returns the record plus the full alpha curve, which main() needs later — the cap that
    turns this curve into a recommendation is a COHORT quantile and is not known until every
    hitter has been fit.

    This layer ships cohort-level only: its split-half reliability is 0.30, below the
    pre-committed 0.5 floor, so the per-hitter column is not a trustworthy recommendation.
    """
    swings = hitter.swings
    frozen = hitter.with_situation_off
    xrv_situation_off = runs_for(models, run_value_tables, swings, frozen)

    def season_runs_at(shape):
        scored = runs_for(models, run_value_tables, swings, shape)
        return float((scored - xrv_situation_off).sum() / N_SEASONS)

    def dial_to(alpha):
        return frozen + alpha * (hitter.as_he_swung_it - frozen)

    # The observed-percentile envelope is a hard rail, not the binding constraint: it is
    # built from measured shapes, whose execution noise (~1.5 SD) dwarfs the situational
    # displacement (~0.37 SD) being blended, so it almost never rejects.
    feasible = [a for a in ALPHA_GRID
                if hitter.fraction_outside(dial_to(a)) < MAX_OUTSIDE]
    alpha_curve = {alpha: season_runs_at(dial_to(alpha)) for alpha in feasible}

    peak_alpha = max(alpha_curve, key=alpha_curve.get) if alpha_curve else float("nan")
    record = {
        "alpha_peak_unconstrained": peak_alpha,
        "alpha_peak_at_boundary": bool(
            feasible and peak_alpha in (min(feasible), max(feasible))),
    }

    # Same scan per axis, with the other two axes left at their observed intensity. Exact,
    # because the per-axis contributions sum to the total by construction.
    for axis in AXIS_NAMES:
        axis_curve = {}
        for alpha in AXIS_ALPHA_GRID:
            shape = hitter.as_he_swung_it + (alpha - 1.0) * (hitter.with_one_axis_live[axis]
                                                             - frozen)
            if hitter.fraction_outside(shape) < MAX_OUTSIDE:
                axis_curve[alpha] = season_runs_at(shape)
        if axis_curve:
            record[f"alpha_star_{axis}"] = max(axis_curve, key=axis_curve.get)
            # Flatness detector: a near-zero range means the argmax is noise, not a policy.
            record[f"runs_range_{axis}"] = max(axis_curve.values()) - min(axis_curve.values())
        else:
            record[f"alpha_star_{axis}"] = float("nan")
            record[f"runs_range_{axis}"] = float("nan")

    # Slope of the value curve right where he currently sits.
    record["marginal_runs_per_alpha"] = (
        season_runs_at(dial_to(1.1)) - season_runs_at(dial_to(0.9))) / 0.2
    return record, alpha_curve


# ----------------------------------------------------------------------------------------
# 6. Prescription, part 2: which dial should he move, and where?
# ----------------------------------------------------------------------------------------

def which_dial_should_he_move(models, run_value_tables, hitter, league_sd):
    """One long row per (situation cell, dial): what the dial pays there, and what he does.

    Part 1 can only say "more or less of the same" — it rescales the one direction he
    already moves in, projecting a five-dimensional problem onto a line. This says which of
    the five dials to move.

    The per-dial slope is a central difference of xRV at +/- 0.1 SD, probed at his MEASURED
    shapes: the advice is about the swings he actually takes, and that is where the run-value
    model is best supported, so this reads a local slope instead of extrapolating.

    `grad_delta_runs_per_sd` subtracts his own all-swing slope, and IT is the prescription,
    not the raw cell slope. The raw slope is dominated by his baseline, so ranking on it just
    surfaces the biggest cell (opp-hand, ~70% of swings) for every hitter; the delta says
    where THIS situation differs from how he normally hits. Read it beside `shift_sd`, how
    far his swing already moves that dial there — the prescription is the gap between them.
    """
    swings = hitter.swings
    slopes = np.empty_like(hitter.measured)
    for dial_index in range(len(SHAPE)):
        step = GRAD_STEP_SD * league_sd[dial_index]
        shape_up, shape_down = hitter.measured.copy(), hitter.measured.copy()
        shape_up[:, dial_index]   += step
        shape_down[:, dial_index] -= step
        xrv_up   = runs_for(models, run_value_tables, swings, shape_up)
        xrv_down = runs_for(models, run_value_tables, swings, shape_down)
        slopes[:, dial_index] = (xrv_up - xrv_down) / (2 * GRAD_STEP_SD)

    already_shifts = (hitter.as_he_swung_it - hitter.with_situation_off) / league_sd
    his_baseline_slope = slopes.mean(axis=0)

    rows = []
    for axis, labels in hitter.situation_cells().items():
        for cell in pd.unique(labels):
            in_cell = labels == cell
            n_cell_swings = int(in_cell.sum())
            if n_cell_swings < MIN_CELL_SWINGS:
                continue
            # Deltas are n-weighted contrasts and sum to exactly zero across an axis's cells,
            # so this scales a realistic move over THIS CELL's season swings, never his whole
            # season, and the per-cell figures must never be summed to a season total.
            season_swings = n_cell_swings / N_SEASONS
            for dial_index, dial in enumerate(SHAPE):
                cell_slope = float(slopes[in_cell, dial_index].mean())
                situational_slope = cell_slope - float(his_baseline_slope[dial_index])
                shift = float(already_shifts[in_cell, dial_index].mean())
                rows.append({
                    "batter_id": swings["batter_id"].iloc[0],
                    "batter_stand": swings["batter_stand"].iloc[0],
                    "axis": axis,
                    "cell": cell,
                    "dial": dial,
                    "n_swings": n_cell_swings,
                    "grad_runs_per_sd": cell_slope,
                    "grad_delta_runs_per_sd": situational_slope,
                    "shift_sd": shift,
                    "runs_from_shift": cell_slope * shift * season_swings,
                    "runs_per_quarter_sd": cell_slope * POLICY_STEP_SD * season_swings,
                    "situational_runs_per_quarter_sd":
                        situational_slope * POLICY_STEP_SD * season_swings,
                })
    return rows


# ----------------------------------------------------------------------------------------
# 7. Assemble, write, report
# ----------------------------------------------------------------------------------------

def write_and_report(unit_table, gradient_rows, alpha_curves, reference, suffix, repeg):
    """Turn the alpha curves into recommendations, join context, write both parquets, print."""
    # Admit only levels of modulation major-league hitters actually sustain. The cap is a
    # quantile of FITTED displacement, so it is tied to the shape design and must be re-pegged
    # whenever that changes — adding the pitch-characteristic controls shrank displacements,
    # which would have left an older cap non-binding. `shape_sd` is a raw league scale and
    # stays frozen across re-pegs.
    if repeg:
        reference["displacement_cap"] = None
    if reference["displacement_cap"] is None:
        reference["displacement_cap"] = float(
            unit_table["displacement_sd"].quantile(POLICY_QUANTILE))
    displacement_cap = reference["displacement_cap"]

    alpha_star, runs_at_alpha_star = [], []
    for displacement_sd, curve in zip(unit_table["displacement_sd"], alpha_curves):
        capped = affordable_alphas(displacement_sd, displacement_cap)
        allowed = {alpha: runs for alpha, runs in curve.items() if alpha in capped}
        best_alpha = max(allowed, key=allowed.get) if allowed else float("nan")
        alpha_star.append(best_alpha)
        # Gain is measured against alpha = 1.0, what he already does.
        runs_at_alpha_star.append(allowed[best_alpha] - curve[1.0]
                                  if allowed and 1.0 in curve else float("nan"))
    unit_table["alpha_star_policy"] = alpha_star
    unit_table["runs_at_alpha_star"] = runs_at_alpha_star
    unit_table["runs_at_alpha_star_per_swing"] = (
        unit_table["runs_at_alpha_star"] * N_SEASONS / unit_table["n_swings"])

    # Per-unit mean xrv_grade_neutral. The count-NEUTRAL grade is used so this control for
    # hitter quality is not itself confounded by count distribution.
    connection = db.connect("xrv_swings")
    swing_plus = connection.sql(f"""
        SELECT batter_id, batter_stand, AVG(xrv_grade_neutral) AS swing_plus
        FROM xrv_swings
        WHERE game_year IN ({", ".join(str(season) for season in SEASONS)})
        GROUP BY batter_id, batter_stand
    """).df()
    connection.close()

    adjustability = pd.read_parquet(DATA / "adjustability.parquet", columns=KEY + [
        "label", "adjustability", "adj_count", "adjustability_plus",
        "adjustability_pctile",
        "twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty",
    ]).merge(swing_plus, on=KEY, how="left")
    cards = pd.read_parquet(DATA / "shape_cards.parquet",
                            columns=KEY + ["role", "archetype_name", "grade"])
    primary_shape = (cards[cards["role"] == "primary"][KEY + ["archetype_name", "grade"]]
                     .rename(columns={"grade": "primary_grade"}))
    repertoire = pd.read_parquet(DATA / "repertoire_scores.parquet",
                                 columns=KEY + ["repertoire_pctile", "effective_shapes"])
    unit_table = (unit_table.merge(adjustability, on=KEY, how="left")
                            .merge(primary_shape, on=KEY, how="left")
                            .merge(repertoire, on=KEY, how="left"))

    gradients = (pd.DataFrame(gradient_rows)
                 .merge(unit_table[KEY + ["label"]], on=KEY, how="left"))

    # Roll up per AXIS, never to one overall winner: the count axis carries an order of
    # magnitude more situational slope than the other two, so a single argmax would read
    # `count` for essentially every hitter and bury the runners and hand prescriptions.
    # Values stay signed — negative means move the dial down.
    ranked = gradients.assign(gain=gradients["situational_runs_per_quarter_sd"].abs())
    top_levers = None
    for axis, axis_block in ranked.groupby("axis"):
        best = axis_block.loc[axis_block.groupby(KEY)["gain"].idxmax()]
        axis_levers = best[KEY].copy()
        axis_levers[f"top_lever_{axis}"] = (best["cell"] + "/" + best["dial"]).to_numpy()
        axis_levers[f"top_lever_{axis}_runs"] = best["situational_runs_per_quarter_sd"].to_numpy()
        top_levers = (axis_levers if top_levers is None
                      else top_levers.merge(axis_levers, on=KEY, how="outer"))
    unit_table = unit_table.merge(top_levers, on=KEY, how="left")

    value_path = DATA / f"adjustability_value{suffix}.parquet"
    gradient_path = DATA / f"adjustability_gradients{suffix}.parquet"
    unit_table.to_parquet(value_path, index=False)
    gradients.to_parquet(gradient_path, index=False)
    print(f"\nWrote {len(unit_table)} rows -> {value_path}")
    print(f"Wrote {len(gradients)} rows -> {gradient_path}")

    if not suffix and (repeg or not POLICY_REF.exists()):
        POLICY_REF.write_text(json.dumps(reference, indent=2) + "\n")
        print(f"Pegged policy reference -> {POLICY_REF}")

    print("\n=== Season runs vs a replacement hitter's policy ===")
    for col in ["runs_total", "runs_count", "runs_gamestate", "runs_platoon",
                "runs_interaction", "runs_total_2k", "runs_vs_desituated"]:
        values = unit_table[col]
        print(f"  {col:<20} mean={values.mean():+6.2f}  median={values.median():+6.2f}  "
              f"sd={values.std():5.2f}  %pos={100 * (values > 0).mean():5.1f}  "
              f"range=[{values.min():+6.1f}, {values.max():+6.1f}]")
    corr_n_swings = unit_table["runs_total"].corr(unit_table["n_swings"])
    corr_swing_plus = unit_table["runs_total"].corr(unit_table["swing_plus"])
    print(f"\n  corr(runs_total, n_swings)  = {corr_n_swings:+.3f}")
    print(f"  corr(runs_total, swing_plus) = {corr_swing_plus:+.3f}")
    interaction = unit_table["runs_interaction"].abs()
    total = unit_table["runs_total"].abs().clip(lower=0.1)
    print(f"\n  |interaction| mean={interaction.mean():.2f}  max={interaction.max():.2f}  "
          f"median % of |total|={100 * (interaction / total).median():.0f}%")

    print("\n=== Policy ===")
    print(f"  displacement cap Q{100 * POLICY_QUANTILE:.0f} = {displacement_cap:.3f} SD")
    for col in ["displacement_sd", "alpha_peak_unconstrained", "alpha_star_policy",
                "runs_at_alpha_star", "runs_at_alpha_star_per_swing"]:
        values = unit_table[col]
        print(f"  {col:<28} mean={values.mean():+7.3f}  median={values.median():+7.3f}")
    at_boundary = 100 * unit_table["alpha_peak_at_boundary"].mean()
    print(f"  peak still at a grid edge: {at_boundary:.0f}% of units")
    for axis in AXIS_NAMES:
        print(f"  alpha_star_{axis:<10} median={unit_table[f'alpha_star_{axis}'].median():.2f}  "
              f"runs range median={unit_table[f'runs_range_{axis}'].median():.3f}")

    print("\n=== Situational gradient (cohort mean |cell - own baseline|, runs/swing per SD) ===")
    with_abs = gradients.assign(g=gradients["grad_delta_runs_per_sd"].abs())
    gradient_pivot = with_abs.pivot_table(index="dial", columns="cell", values="g", aggfunc="mean")
    print(gradient_pivot.round(4).to_string())

    print("\n=== Spot check ===")
    spot_check = unit_table[unit_table["label"]
                            .str.contains("Judge|Arraez|Schwarber|Teoscar", na=False)]
    print(spot_check[["label", "runs_total", "alpha_star_policy", "runs_at_alpha_star"]
                     + [f"top_lever_{a}" for a in AXIS_NAMES]].to_string(index=False))
    return unit_table


def affordable_alphas(displacement_sd, cap, grid=ALPHA_GRID):
    """Alphas whose scaled displacement stays inside the league-referenced cap.

    Asks whether the proposed level of modulation is one big-league hitters actually sustain.
    Non-circular by construction: the reference is the cohort, not this hitter's own fitted
    range, which would degenerate to alpha <= 1.
    """
    if not np.isfinite(displacement_sd) or displacement_sd <= 0:
        return list(grid)
    # The cap is a quantile OF these values, so the unit defining it lands exactly on the
    # boundary at alpha = 1 and must not be rejected by float error.
    return [a for a in grid if a * displacement_sd <= cap * (1 + 1e-9)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeg", action="store_true",
                        help="recompute the displacement cap and rewrite the reference")
    args = parser.parse_args()

    models, run_value_tables, swings, reference = load_data()
    print(f"{len(swings):,} swings, "
          f"{swings.groupby(KEY, observed=True).ngroups} units")

    if args.verify:
        # Score the MEASURED shapes and compare to the published xrv column. This is the
        # pitch_type category-alignment guard: if the codes were misaligned the deviation
        # would be large and obvious.
        published = pd.read_parquet(DATA / "xrv_swings.parquet", columns=["play_id", "xrv"])
        all_units = dict.fromkeys(zip(swings["batter_id"], swings["batter_stand"]))
        worst_deviation = 0.0
        for batter_id, stand in list(all_units)[:5]:
            group = swings[(swings["batter_id"] == batter_id)
                           & (swings["batter_stand"] == stand)]
            scored = runs_for(models, run_value_tables, group, group[SHAPE].to_numpy(float))
            merged = pd.DataFrame({"play_id": group["play_id"].to_numpy(), "got": scored}
                                  ).merge(published, on="play_id", how="inner")
            if merged.empty:
                raise AssertionError(f"no play_id overlap for {batter_id} {stand}")
            worst_deviation = max(worst_deviation,
                                  float((merged["got"] - merged["xrv"]).abs().max()))
        print(f"max |scored - published xrv| = {worst_deviation:.3e}")
        print("PASS" if worst_deviation < 1e-5 else "FAIL — pitch_type categories misaligned")
        return

    league_sd = np.asarray([reference["shape_sd"][feature] for feature in SHAPE], float)
    groups = list(swings.groupby(KEY, observed=True, sort=False))
    if args.limit:
        groups = groups[:args.limit]

    print(f"pass 1: fitting {len(groups)} unit policies")
    hitters = [learn_how_he_swings(group.reset_index(drop=True), league_sd)
               for _, group in groups]
    everyones_blocks = [hitter.how_he_adjusts for hitter in hitters]

    print(f"pass 2: scoring arms ({N_REPLACEMENTS} replacements per unit)")
    records, alpha_curves, gradient_rows = [], [], []
    for i, hitter in enumerate(hitters):
        record = value_against_replacement(models, run_value_tables, hitter, league_sd,
                                           everyones_blocks, i)
        alpha_record, alpha_curve = how_much_should_he_adjust(
            models, run_value_tables, hitter)
        record.update(alpha_record)
        alpha_curves.append(alpha_curve)
        gradient_rows.extend(which_dial_should_he_move(
            models, run_value_tables, hitter, league_sd))
        records.append(record)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(hitters)} units")

    write_and_report(pd.DataFrame(records), gradient_rows, alpha_curves, reference,
                     "_subset" if args.limit else "", args.repeg)


if __name__ == "__main__":
    main()
