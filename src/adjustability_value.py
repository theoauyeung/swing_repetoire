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
MAX_STD_SHIFT = 1.5  # clip per-dial standardised shift before abs to guard against degenerate folds

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
                dial_shifts["composite"][:, d_idx] = np.abs(
                    np.clip(raw / dial_sd, -MAX_STD_SHIFT, MAX_STD_SHIFT)
                )

                # Per-axis: only that axis's column slice
                for axis_name, slc in axis_slices.items():
                    axis_coefs = sit_coefs[slc]
                    raw_axis   = X_val[:, n_loc + slc.start: n_loc + slc.stop] @ axis_coefs
                    dial_shifts[axis_name][:, d_idx] = np.abs(
                        np.clip(raw_axis / dial_sd, -MAX_STD_SHIFT, MAX_STD_SHIFT)
                    )

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


def load_swings() -> pd.DataFrame:
    """Load and context-enrich the full competitive swing table for 2024-25."""
    return add_context(pd.read_parquet(
        DATA / "swings_model.parquet",
        columns=KEY + [
            "game_year", "batter_full_name",
            "balls", "strikes", "outs_when_up",
            "plate_x", "plate_z", "sz_top", "sz_bot",
            "pitch_type", "pitcher_throws",
            "on_1b_id", "on_2b_id", "on_3b_id",
            "pitcher_id", "delta_run_exp", "is_whiff",
        ] + DIALS,
    ).query("game_year in @SEASONS"))


def build_pitcher_quality(df: pd.DataFrame) -> pd.Series:
    """Per-pitcher mean delta_run_exp across all their swings — proxy for pitcher difficulty."""
    return df.groupby("pitcher_id")["delta_run_exp"].mean().rename("pitcher_quality")


def validate_treatments(treats: pd.DataFrame) -> float:
    """
    Aggregate swing-level composite treatment to (batter, stand) mean and compare
    to the batter-level adjustability score from adjustability.parquet.
    Pearson r should be >= CORR_WARN; prints a warning if not.
    """
    agg = (treats.groupby(KEY)["T_composite"].mean().rename("T_composite_agg").reset_index())
    ref = pd.read_parquet(DATA / "adjustability.parquet", columns=KEY + ["adjustability"])
    merged = agg.merge(ref, on=KEY, how="inner")
    r = float(merged["T_composite_agg"].corr(merged["adjustability"]))
    print(f"  Validation: swing-level composite vs batter-level adjustability r={r:.3f}  "
          f"(n={len(merged)})")
    if r < CORR_WARN:
        print(f"  WARNING: r={r:.3f} < {CORR_WARN} — treatment construction may have diverged from intent")
    return r


def assemble_analysis_df(df: pd.DataFrame, treats: pd.DataFrame) -> tuple:
    """
    Merge swing table with treatment scores, add derived confounder columns,
    dummy-encode categoricals. Returns one wide DataFrame ready for dml_swing calls.
    """
    df = df.reset_index(drop=True)
    df["swing_idx"] = df.groupby(KEY).cumcount()

    merged = df.merge(treats, on=KEY + ["swing_idx"], how="inner")

    # Quadratic / interaction location terms
    merged["px2"]  = merged["px"] ** 2
    merged["pz2"]  = merged["pz"] ** 2
    merged["pxpz"] = merged["px"] * merged["pz"]

    # Dummy-encode categoricals (drop_first to avoid collinearity)
    cat_dummies = pd.get_dummies(
        merged[CONFOUNDER_CAT].astype(str), drop_first=True
    )

    # CONFOUNDER_NUM already includes "strikes" and "balls"; list them once
    result = pd.concat(
        [merged[["batter_id", "batter_stand",
                  "delta_run_exp", "is_whiff"]
                 + [f"T_{ax}" for ax in list(AXES.keys())]
                 + ["T_composite"]
                 + CONFOUNDER_NUM].reset_index(drop=True),
         cat_dummies.reset_index(drop=True)],
        axis=1,
    )
    return result, CONFOUNDER_NUM + cat_dummies.columns.tolist()


def dml_swing(df: pd.DataFrame, treatment_col: str, outcome_col: str,
              confounder_cols: list) -> tuple:
    """
    Swing-level DML. Robinson (1988) partial linear model:
      - Within-batter demeaning for batter FE (absorbs swing quality, playing time, etc.)
      - XGBoost nuisance models with GroupKFold on batter_id
      - Clustered sandwich SE by batter (accounts for ~200 correlated swings per hitter)

    Treatment and outcome standardised to mean=0, SD=1 before nuisance fitting so θ is
    a standardised partial effect comparable across outcomes.

    Returns: (theta, se, t, p, n, r2_T, r2_Y)
    """
    df = (df.dropna(subset=[treatment_col, outcome_col] + confounder_cols)
            .copy()
            .reset_index(drop=True))

    # Within-batter demeaning — subtracts each batter's mean from every column
    # This is the within-estimator (strict FE): absorbs all batter-level confounders
    for col in [treatment_col, outcome_col] + confounder_cols:
        df[col] = df[col] - df.groupby("batter_id")[col].transform("mean")

    T_raw = df[treatment_col].to_numpy(float)
    Y_raw = df[outcome_col].to_numpy(float)
    T_std, Y_std = T_raw.std(), Y_raw.std()
    T = T_raw / T_std if T_std > 0 else T_raw
    Y = Y_raw / Y_std if Y_std > 0 else Y_raw

    X           = df[confounder_cols].to_numpy(float)
    batter_ids  = df["batter_id"].to_numpy()
    n           = len(df)

    T_resid = np.zeros(n)
    Y_resid = np.zeros(n)

    gkf = GroupKFold(n_splits=N_OUTER_FOLDS)
    for train_idx, val_idx in gkf.split(X, groups=batter_ids):
        model_T = XGBRegressor(**XGB_PARAMS)
        model_T.fit(X[train_idx], T[train_idx])
        T_resid[val_idx] = T[val_idx] - model_T.predict(X[val_idx])

        model_Y = XGBRegressor(**XGB_PARAMS)
        model_Y.fit(X[train_idx], Y[train_idx])
        Y_resid[val_idx] = Y[val_idx] - model_Y.predict(X[val_idx])

    theta = float(np.dot(T_resid, Y_resid) / np.dot(T_resid, T_resid))

    # Clustered sandwich SE: sum influence scores within each batter before squaring
    psi             = T_resid * (Y_resid - theta * T_resid)
    cluster_sums    = pd.Series(psi, index=batter_ids).groupby(level=0).sum()
    B               = len(cluster_sums)
    mean_sq_T       = float(np.mean(T_resid ** 2))
    # Small-sample correction: B/(B-1); negligible at B≈471 but correct practice
    var_theta       = float((cluster_sums ** 2).sum()) / (mean_sq_T ** 2) / n ** 2 * (B / (B - 1))
    se              = float(np.sqrt(max(var_theta, 0.0)))

    t = theta / se if se > 0 else np.nan
    # df = B-1: SE is estimated from B cluster sums, not n rows
    p = float(2 * stats.t.sf(abs(t), df=B - 1)) if se > 0 else np.nan

    r2_T = float(1 - np.sum(T_resid**2) / np.sum((T - T.mean())**2))
    r2_Y = float(1 - np.sum(Y_resid**2) / np.sum((Y - Y.mean())**2))

    return theta, se, t, p, n, r2_T, r2_Y


def main():
    print("Loading swings...")
    df = load_swings()
    print(f"  {len(df):,} swings")

    print("Building pitcher quality...")
    pq = build_pitcher_quality(df)
    df = df.merge(pq.reset_index(), on="pitcher_id", how="left")

    print("Building swing treatments (within-batter cross-fitting)...")
    treats = build_swing_treatments(df, AXES)
    print(f"  {len(treats):,} treatment rows")

    print("Validating treatments...")
    validate_treatments(treats)

    print("Assembling analysis DataFrame...")
    analysis_df, confounder_cols = assemble_analysis_df(df, treats)
    df_2k = analysis_df[analysis_df["strikes"] == 2].copy()
    print(f"  All swings n={len(analysis_df):,}  |  Two-strike n={len(df_2k):,}")

    # Map treatment name -> column name in analysis_df
    treat_col = {
        "adjustability": "T_composite",
        "adj_count":     "T_count",
        "adj_gamestate": "T_gamestate",
        "adj_pitch":     "T_pitch",
        "adj_platoon":   "T_platoon",
    }

    specs = [
        ("Season-wide",  analysis_df, "delta_run_exp", "Run value per swing"),
        ("Season-wide",  analysis_df, "is_whiff",      "Whiff rate"),
        ("Two-strike",   df_2k,       "delta_run_exp", "Two-strike run value"),
        ("Two-strike",   df_2k,       "is_whiff",      "Two-strike whiff rate"),
    ]

    rows = []
    for treatment in TREATMENTS:
        for scope, data, outcome, label in specs:
            print(f"  DML: {treatment} / {scope} / {outcome}...")
            theta, se, t, p, n, r2_T, r2_Y = dml_swing(
                data, treat_col[treatment], outcome, confounder_cols
            )
            rows.append(dict(
                treatment=treatment, scope=scope, outcome=label, n=n,
                theta=round(theta, 4), se=round(se, 4),
                ci_lo=round(theta - 1.96 * se, 4),
                ci_hi=round(theta + 1.96 * se, 4),
                t=round(t, 2), p=round(p, 4),
                r2_T=round(r2_T, 3), r2_Y=round(r2_Y, 3),
            ))
            print(f"    theta={theta:+.4f}  SE={se:.4f}  t={t:.2f}  p={p:.4f}"
                  f"  r2_T={r2_T:.3f}  r2_Y={r2_Y:.3f}")

    tab = pd.DataFrame(rows)
    lines = [
        "# Adjustability value — swing-level DML causal estimates\n",
        f"Swing-level DML (n~100k swings). Treatment = per-swing fitted situational shift magnitude "
        f"from within-batter 5-fold cross-fitting, averaged across 3 dials "
        f"(bat_speed, swing_length, swing_path_tilt) in within-batter SD units. "
        f"Unsigned (absolute value). 2024-25, >=400 swings per (batter, stand).\n",
        "**Method:** Robinson (1988) partial linear model. XGBoost nuisance models "
        f"(GroupKFold on batter_id, {N_OUTER_FOLDS} folds). Within-batter demeaning for batter FE. "
        "Clustered sandwich SE by batter.\n",
        "**Confounders:** location surface (px, pz, px^2, pz^2, px*pz), count (balls, strikes), "
        "game state (base_state, outs_when_up), pitch group, platoon, pitcher quality. "
        "Batter-level traits (swing quality, repertoire, playing time, handedness) absorbed by demeaning.\n",
        "**Two-strike subset:** strikes==2 filter applied before demeaning.\n",
        "## Results\n",
        tab.to_markdown(index=False),
        "",
    ]
    out = ROOT / "results" / "adjustability_value.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
