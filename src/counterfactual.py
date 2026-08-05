"""Counterfactual de-situated swing shapes — pure numeric core.

No I/O and no xRV. This module turns a unit's swing table into fitted actual and
counterfactual shape matrices, so the pieces where a silent bug would corrupt every
published number can be unit-tested without loading models or data.

The counterfactual sets each situation axis's dummies to their unit MEAN rather than to
a reference category. That is mean-preserving: the hitter's average swing is unchanged
and only situational variation is removed.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

AXES = {
    "count":     ["balls", "strikes"],
    "gamestate": ["base_state", "outs_when_up"],
    "platoon":   ["pitcher_throws"],
}
# Held at observed values in every arm. pitch type is reactive — a hitter cannot choose
# his swing before identifying the pitch, and the measured shape on a breaking ball partly
# reflects being fooled, so de-situating it removes consequence rather than policy.
CONTROL_COLS = ["pitch_group"]

N_FOLDS = 5
SEED = 7
# Fine near alpha=1, where any realistic prescription lives; coarse past 3.0, where the
# only job is to locate the turn. The old grid stopped at 2.0, below the peak, which is
# why 86% of units pegged at its ceiling and alpha* was a grid artifact.
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
AXIS_ALPHA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
ENVELOPE_LO, ENVELOPE_HI = 1.0, 99.0
MAX_OUTSIDE = 0.05
POLICY_QUANTILE = 0.90
GRAD_STEP_SD = 0.1
POLICY_STEP_SD = 0.25
MIN_CELL_SWINGS = 25


def build_design(group, location_matrix):
    """[location | pitch_group | count | gamestate | platoon] and the axis column slices."""
    blocks = [location_matrix]
    width = location_matrix.shape[1]

    control = pd.get_dummies(group[CONTROL_COLS].astype(str), drop_first=True).to_numpy(float)
    blocks.append(control)
    width += control.shape[1]

    axis_slices = {}
    for axis, cols in AXES.items():
        dummies = pd.get_dummies(group[cols].astype(str), drop_first=True).to_numpy(float)
        axis_slices[axis] = slice(width, width + dummies.shape[1])
        blocks.append(dummies)
        width += dummies.shape[1]

    return np.column_stack(blocks), axis_slices


def desituate(X, axis_slices, axes_off):
    """Copy of X with the named axes' dummy columns flattened to their column means."""
    out = X.copy()
    for axis in axes_off:
        block = axis_slices[axis]
        out[:, block] = X[:, block].mean(axis=0)
    return out


def crossfit_shapes(X, Y, n_folds=N_FOLDS, seed=SEED):
    """Out-of-fold coefficient sets as [(test_index, coefs), ...].

    In-sample fits would let ~20 situation dummies manufacture situational signal from
    noise, and the headline is a difference of fitted values — exactly where that leaks.
    """
    splitter = KFold(n_splits=min(n_folds, len(X)), shuffle=True, random_state=seed)
    return [(test, np.linalg.lstsq(X[train], Y[train], rcond=None)[0])
            for train, test in splitter.split(X)]


def predict_oof(design, fits, n_features):
    """Assemble an out-of-fold prediction matrix using each fold's own coefficients."""
    out = np.empty((len(design), n_features))
    for test, coefs in fits:
        out[test] = design[test] @ coefs
    return out


def blend(shape_cf, shape_actual, alpha):
    """Shape at situational intensity alpha.

    The shape model is linear in the situation dummies, so this interpolation is exact —
    no refit is needed per alpha.
    """
    return shape_cf + alpha * (shape_actual - shape_cf)


def envelope(observed):
    """Per-feature (lo, hi) from the unit's OBSERVED percentiles."""
    return (np.percentile(observed, ENVELOPE_LO, axis=0),
            np.percentile(observed, ENVELOPE_HI, axis=0))


def fraction_outside(shape, env):
    lo, hi = env
    return float(((shape < lo) | (shape > hi)).mean())


def admissible_alphas(shape_cf, shape_actual, env, grid=ALPHA_GRID, max_outside=MAX_OUTSIDE):
    """Alphas for which fewer than max_outside of (swing, feature) pairs fall outside
    the unit's own observed 1st–99th percentile envelope.

    The filter rejects blended shapes that venture into regions the hitter has never
    actually occupied, preventing xRV from extrapolating into out-of-sample territory.
    In practice this constraint rarely binds: across the 2024-25 cohort, 406 of 471
    units (86%) still land at the grid maximum alpha=2.0, and 88.5% are at a boundary.
    The reason is that the envelope is built from OBSERVED shape percentiles, which span
    the hitter's full empirical spread, while the situational component of the fitted
    shapes is far narrower — blending out to alpha=2 almost never leaves that wide
    envelope."""
    return [a for a in grid
            if fraction_outside(blend(shape_cf, shape_actual, a), env) < max_outside]


def axis_blend(shape_actual, shape_axis, shape_cf, alpha):
    """Shape with one axis at intensity alpha and the other axes left at observed intensity.

    Exact, with no extra regression: desituating an axis is a linear operation on the
    design and the prediction is linear in it, so the per-axis contributions
    (shape_axis - shape_cf) sum exactly to (shape_actual - shape_cf).
    """
    return shape_actual + (alpha - 1.0) * (shape_axis - shape_cf)


def mean_displacement(shape_actual, shape_cf, scale):
    """Unit-level D_u: mean Euclidean norm of per-swing situational displacement, in SD units."""
    return float(np.linalg.norm((shape_actual - shape_cf) / scale, axis=1).mean())


def policy_alphas(d_u, cap, grid=ALPHA_GRID):
    """Alphas whose scaled displacement stays inside the league-referenced cap.

    Asks whether the proposed level of situational modulation is one major-league hitters
    actually sustain. Non-circular: the reference is the cohort, not this hitter's own
    fitted range, which would degenerate to alpha <= 1.
    """
    if not np.isfinite(d_u) or d_u <= 0:
        return list(grid)
    # The cap is a quantile OF the d_u values, so the unit defining it lands exactly on the
    # boundary at alpha=1 and must not be rejected by float error.
    return [a for a in grid if a * d_u <= cap * (1 + 1e-9)]


def cell_labels(group):
    """Interpretable situation cells for the prescriptive layer, one label array per axis.

    Coarser than the axes the de-situation uses: prescriptions are read per cell, and the
    full balls x strikes x base x outs cross is too thin to estimate a per-cell gradient.
    """
    strikes = group["strikes"].to_numpy()
    same_hand = group["pitcher_throws"].to_numpy() == group["batter_stand"].to_numpy()
    return {
        "count": np.where(strikes >= 2, "2 strikes",
                          np.where(strikes == 1, "1 strike", "0 strikes")),
        "gamestate": group["base_state"].to_numpy().astype(str),
        "platoon": np.where(same_hand, "same-hand", "opp-hand"),
    }
