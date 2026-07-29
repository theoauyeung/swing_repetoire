"""Swing-level DML: per-swing situational shift magnitude as treatment.

Per (batter, stand), 5-fold within-batter cross-fitting of
    dial ~ location_surface + situation_dummies
produces a per-swing treatment T = mean |standardised situational shift| across
the 3 volitional dials (bat_speed, swing_length, swing_path_tilt). The DML
uses GroupKFold on batter_id, within-batter demeaning for batter FE, and a
clustered sandwich SE (clustered by batter_id).

Two analyses:
  Season-wide : all swings, outcomes delta_run_exp + is_whiff
  Two-strike  : strikes==2 subset, same outcomes

Output: results/adjustability_value.md
Run   : python src/adjustability_value.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBRegressor

# Reuse regression helpers and shared constants from adjustability.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adjustability import location_design, situation_dummies, add_context  # noqa: E402
from adjustability import DIALS, KEY, SEASONS, MIN_SWINGS               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TREATMENTS = ["adjustability", "adj_count", "adj_gamestate", "adj_pitch", "adj_platoon"]
# Maps each treatment name to the axes dict key (or "composite" for the headline)
TREATMENT_AXIS = {
    "adjustability": "composite",
    "adj_count":     "count",
    "adj_gamestate": "gamestate",
    "adj_pitch":     "pitch",
    "adj_platoon":   "platoon",
}

AXES = {
    "count":     ["balls", "strikes"],
    "gamestate": ["base_state", "outs_when_up"],
    "pitch":     ["pitch_group"],
    "platoon":   ["pitcher_throws"],
}

N_INNER_FOLDS = 5   # within-batter cross-fitting folds
N_OUTER_FOLDS = 5   # DML nuisance cross-fitting folds
SEED          = 42
CORR_WARN     = 0.6  # soft threshold for treatment validation

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
    random_state=SEED, verbosity=0,
)

# Categorical confounder columns that need dummy-encoding before XGBoost
CONFOUNDER_CAT = ["base_state", "pitch_group", "pitcher_throws"]
# Continuous confounder columns (location surface + count + outs + pitcher quality)
CONFOUNDER_NUM = ["px", "pz", "px2", "pz2", "pxpz", "balls", "strikes",
                  "outs_when_up", "pitcher_quality"]


def build_axis_dummies(group: pd.DataFrame, axes: dict) -> tuple:
    """
    Build situation dummy columns for all axes, tracking which columns belong to each axis.

    Dummies are built on the FULL batter group so all categorical levels are captured
    (avoids unseen-level mismatches when slicing training/validation folds).

    Returns:
        sit_matrix  — ndarray shape (n, total_sit_cols)
        axis_slices — dict mapping axis_name -> slice into sit_matrix columns
    """
    pieces = []
    axis_slices = {}
    col_offset = 0
    for axis_name, cols in axes.items():
        dum = pd.get_dummies(group[cols].astype(str), drop_first=True).to_numpy(float)
        axis_slices[axis_name] = slice(col_offset, col_offset + dum.shape[1])
        pieces.append(dum)
        col_offset += dum.shape[1]
    sit_matrix = np.column_stack(pieces) if pieces else np.zeros((len(group), 0))
    return sit_matrix, axis_slices


def build_swing_treatments(df: pd.DataFrame, axes: dict) -> pd.DataFrame:
    """
    5-fold within-batter cross-fitting: fit dial ~ location + situation on 4/5 of
    each batter's swings, score the held-out 1/5 as |standardised situational shift|
    averaged across the 3 dials.

    For per-axis treatments, only that axis's coefficient sub-block is used —
    the joint fit already partials out overlap between axes.

    Returns DataFrame with columns:
        batter_id, batter_stand, swing_idx,
        T_composite, T_count, T_gamestate, T_pitch, T_platoon
    """
    axis_names = list(axes.keys())
    records = []

    for (batter_id, stand), group in df.groupby(KEY, sort=False):
        if len(group) < MIN_SWINGS:
            continue

        group = group.reset_index(drop=True)
        n = len(group)

        loc_full                = location_design(group)               # (n, 6)
        sit_full, axis_slices   = build_axis_dummies(group, axes)      # (n, n_sit)
        n_loc                   = loc_full.shape[1]
        X_full                  = np.column_stack([loc_full, sit_full]) # (n, n_loc+n_sit)

        # Accumulators — NaN so missing folds are visible
        T = {name: np.full(n, np.nan) for name in ["composite"] + axis_names}

        kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)
        for train_idx, val_idx in kf.split(X_full):
            X_train, X_val = X_full[train_idx], X_full[val_idx]

            # dial_shifts[name] shape: (len(val_idx), len(DIALS))
            dial_shifts = {name: np.zeros((len(val_idx), len(DIALS)))
                           for name in ["composite"] + axis_names}

            for d_idx, dial in enumerate(DIALS):
                y_train  = group.loc[train_idx, dial].to_numpy(float)
                dial_sd  = float(y_train.std())
                if dial_sd == 0:
                    continue

                coefs, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
                sit_coefs = coefs[n_loc:]  # situation sub-vector

                # Composite: all situation columns
                raw = X_val[:, n_loc:] @ sit_coefs
                dial_shifts["composite"][:, d_idx] = np.abs(raw / dial_sd)

                # Per-axis: only that axis's column slice
                for axis_name, slc in axis_slices.items():
                    axis_coefs = sit_coefs[slc]
                    raw_axis   = X_val[:, n_loc + slc.start: n_loc + slc.stop] @ axis_coefs
                    dial_shifts[axis_name][:, d_idx] = np.abs(raw_axis / dial_sd)

            # Average across dials → one scalar per swing per treatment
            for name in ["composite"] + axis_names:
                T[name][val_idx] = dial_shifts[name].mean(axis=1)

        for i in range(n):
            records.append({
                "batter_id":    batter_id,
                "batter_stand": stand,
                "swing_idx":    i,
                "T_composite":  T["composite"][i],
                **{f"T_{ax}": T[ax][i] for ax in axis_names},
            })

    return pd.DataFrame(records)
