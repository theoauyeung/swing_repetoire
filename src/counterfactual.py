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
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
ENVELOPE_LO, ENVELOPE_HI = 1.0, 99.0
MAX_OUTSIDE = 0.05


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
    """Alphas leaving under max_outside of (swing, feature) pairs outside the hitter's own
    observed envelope. Without this the scan pegs at the grid edge for most hitters —
    xRV extrapolates cheerfully into shapes nobody has ever made."""
    return [a for a in grid
            if fraction_outside(blend(shape_cf, shape_actual, a), env) < max_outside]
