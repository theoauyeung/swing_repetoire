"""Adjustability value — multi-axis matched penalties + between-batter OLS.

Three situational penalties (each vs empty-base reference within matched cells):
  twostrike_rv_penalty : 2-strike vs 0-1 strike            within (pitch_type × zone)
  gamestate_rv_penalty : any runner vs empty bases          within (pitch_type × zone × strikes)
  platoon_rv_penalty   : same-hand vs opp-hand matchup     within (pitch_type × zone × strikes)

Each penalty is then regressed on its corresponding adjustment axis (+ swing_plus and
repertoire_pctile as controls) in a between-batter OLS with clustered SE.

Output: data/adjustability.parquet (updated), results/adjustability_value.md
Run   : python src/adjustability_value.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
from adjustability import location_design, add_context  # noqa: E402
from adjustability import DIALS, KEY, SEASONS, MIN_SWINGS               # noqa: E402

SEASONS_SQL = "(" + ", ".join(str(s) for s in SEASONS) + ")"

# ─── Section 1 helpers ────────────────────────────────────────────────────────

def compute_matched_penalties() -> pd.DataFrame:
    """
    Per-batter matched two-strike penalty via DuckDB (empirical delta_run_exp).

    Within each (batter, pitch_type × plate_zone) cell that has ≥3 two-strike
    AND ≥3 early-count (0-1 strike) swings, compute:
        Δrv    = mean delta_run_exp at 2 strikes − mean delta_run_exp at 0-1 strikes
        Δwhiff = mean is_whiff      at 2 strikes − mean is_whiff      at 0-1 strikes

    Uses realized delta_run_exp (not xrv_grade) because xrv_grade conditions on count
    — the rv layer mechanically penalises 2-strike swings (rv_whiff = lw_K − ERV(b,2))
    regardless of the batter, so Δxrv would capture count mechanics, not adjustment skill.

    Per-batter penalty = weighted mean of Δ across cells (weights = n_2k in cell).

    Returns columns: batter_id, batter_stand, twostrike_rv_penalty,
                     twostrike_whiff_penalty, matched_n_2k, matched_cells.
    """
    con = db.connect("swings")
    result = con.sql(f"""
    WITH base AS (
        SELECT batter_id, batter_stand,
               (strikes = 2)::INT     AS twoK,
               delta_run_exp,
               is_whiff::DOUBLE       AS is_whiff,
               pitch_type || '|' || plate_zone::VARCHAR AS zone_pitch
        FROM swings
        WHERE game_year IN {SEASONS_SQL}
          AND delta_run_exp IS NOT NULL
    ),
    unit_n AS (
        SELECT batter_id, batter_stand
        FROM base
        GROUP BY batter_id, batter_stand
        HAVING COUNT(*) >= {MIN_SWINGS}
    ),
    cell AS (
        SELECT b.batter_id, b.batter_stand, b.zone_pitch,
               AVG(delta_run_exp) FILTER (WHERE twoK = 1) AS mu_rv_2k,
               AVG(delta_run_exp) FILTER (WHERE twoK = 0) AS mu_rv_early,
               AVG(is_whiff)      FILTER (WHERE twoK = 1) AS mu_wh_2k,
               AVG(is_whiff)      FILTER (WHERE twoK = 0) AS mu_wh_early,
               COUNT(*)           FILTER (WHERE twoK = 1) AS n_2k,
               COUNT(*)           FILTER (WHERE twoK = 0) AS n_early
        FROM base b
        INNER JOIN unit_n u USING (batter_id, batter_stand)
        GROUP BY b.batter_id, b.batter_stand, b.zone_pitch
        HAVING COUNT(*) FILTER (WHERE twoK = 1) >= 3
           AND COUNT(*) FILTER (WHERE twoK = 0) >= 3
    )
    SELECT batter_id, batter_stand,
           SUM(n_2k::DOUBLE * (mu_rv_2k  - mu_rv_early)) / SUM(n_2k) AS twostrike_rv_penalty,
           SUM(n_2k::DOUBLE * (mu_wh_2k  - mu_wh_early)) / SUM(n_2k) AS twostrike_whiff_penalty,
           SUM(n_2k)::INTEGER  AS matched_n_2k,
           COUNT(*)::INTEGER   AS matched_cells
    FROM cell
    GROUP BY batter_id, batter_stand
    """).df()
    con.close()
    return result


def compute_gamestate_penalty() -> pd.DataFrame:
    """
    Broad game-state penalty: any runner on base vs empty, within (pitch_type × zone × strikes).

    Collapses all runner configurations (1B, 2B, 3B, combinations) into "any runner" vs.
    the "empty" reference. Using the full range of base-occupied states aligns with how
    adj_gamestate is constructed (base_state = risp | on1 | empty, plus outs_when_up) and
    yields near-complete unit coverage (471/471 units). Earlier RISP+0-out / DP-avoid
    specifications recovered only 267 and 428 units respectively, cutting power unnecessarily.

    Strikes in the matching key holds count fixed.
    """
    con = db.connect("swings")
    result = con.sql(f"""
    WITH base AS (
        SELECT batter_id, batter_stand,
               (on_1b_id IS NOT NULL OR on_2b_id IS NOT NULL OR on_3b_id IS NOT NULL) AS any_runner,
               (on_1b_id IS NULL AND on_2b_id IS NULL AND on_3b_id IS NULL)            AS empty,
               delta_run_exp,
               is_whiff::DOUBLE AS is_whiff,
               pitch_type || '|' || plate_zone::VARCHAR || '|' || strikes::VARCHAR AS match_key
        FROM swings
        WHERE game_year IN {SEASONS_SQL}
          AND delta_run_exp IS NOT NULL
    ),
    unit_n AS (
        SELECT batter_id, batter_stand FROM base
        GROUP BY batter_id, batter_stand HAVING COUNT(*) >= {MIN_SWINGS}
    ),
    cells AS (
        SELECT b.batter_id, b.batter_stand, b.match_key,
               AVG(delta_run_exp) FILTER (WHERE any_runner) AS mu_rv_runner,
               AVG(delta_run_exp) FILTER (WHERE empty)      AS mu_rv_empty,
               AVG(is_whiff)      FILTER (WHERE any_runner) AS mu_wh_runner,
               AVG(is_whiff)      FILTER (WHERE empty)      AS mu_wh_empty,
               COUNT(*)           FILTER (WHERE any_runner) AS n_runner,
               COUNT(*)           FILTER (WHERE empty)      AS n_empty
        FROM base b INNER JOIN unit_n u USING (batter_id, batter_stand)
        GROUP BY b.batter_id, b.batter_stand, b.match_key
        HAVING COUNT(*) FILTER (WHERE any_runner) >= 3
           AND COUNT(*) FILTER (WHERE empty)      >= 3
    )
    SELECT batter_id, batter_stand,
           SUM(n_runner::DOUBLE * (mu_rv_runner  - mu_rv_empty)) / SUM(n_runner) AS gamestate_rv_penalty,
           SUM(n_runner::DOUBLE * (mu_wh_runner  - mu_wh_empty)) / SUM(n_runner) AS gamestate_whiff_penalty,
           SUM(n_runner)::INTEGER AS gamestate_n,
           COUNT(*)::INTEGER      AS gamestate_cells
    FROM cells GROUP BY batter_id, batter_stand
    """).df()
    con.close()
    return result


def compute_platoon_penalty() -> pd.DataFrame:
    """
    Arm-side (same-hand) penalty: same-hand matchups vs opposite-hand matchups,
    within (pitch_type × zone × strikes). Positive = hitter loses more run value
    against same-hand pitchers (normal platoon disadvantage).

    Strikes in the matching key holds count fixed so arm-side effects aren't
    confounded by the pitch mix pitchers throw in two-strike counts.
    """
    con = db.connect("swings")
    result = con.sql(f"""
    WITH base AS (
        SELECT batter_id, batter_stand,
               (batter_stand = pitcher_throws)::INT AS same_hand,
               delta_run_exp,
               is_whiff::DOUBLE AS is_whiff,
               pitch_type || '|' || plate_zone::VARCHAR || '|' || strikes::VARCHAR AS match_key
        FROM swings
        WHERE game_year IN {SEASONS_SQL}
          AND delta_run_exp IS NOT NULL
          AND pitcher_throws IS NOT NULL
    ),
    unit_n AS (
        SELECT batter_id, batter_stand FROM base
        GROUP BY batter_id, batter_stand HAVING COUNT(*) >= {MIN_SWINGS}
    ),
    cell AS (
        SELECT b.batter_id, b.batter_stand, b.match_key,
               AVG(delta_run_exp) FILTER (WHERE same_hand = 1) AS mu_rv_same,
               AVG(delta_run_exp) FILTER (WHERE same_hand = 0) AS mu_rv_opp,
               AVG(is_whiff)      FILTER (WHERE same_hand = 1) AS mu_wh_same,
               AVG(is_whiff)      FILTER (WHERE same_hand = 0) AS mu_wh_opp,
               COUNT(*)           FILTER (WHERE same_hand = 1) AS n_same,
               COUNT(*)           FILTER (WHERE same_hand = 0) AS n_opp
        FROM base b INNER JOIN unit_n u USING (batter_id, batter_stand)
        GROUP BY b.batter_id, b.batter_stand, b.match_key
        HAVING COUNT(*) FILTER (WHERE same_hand = 1) >= 3
           AND COUNT(*) FILTER (WHERE same_hand = 0) >= 3
    )
    SELECT batter_id, batter_stand,
           SUM(n_same::DOUBLE * (mu_rv_same - mu_rv_opp)) / SUM(n_same) AS platoon_rv_penalty,
           SUM(n_same::DOUBLE * (mu_wh_same - mu_wh_opp)) / SUM(n_same) AS platoon_whiff_penalty,
           SUM(n_same)::INTEGER AS platoon_n,
           COUNT(*)::INTEGER    AS platoon_cells
    FROM cell GROUP BY batter_id, batter_stand
    """).df()
    con.close()
    return result


def compute_swing_plus() -> pd.DataFrame:
    """Per-batter mean xrv_grade_neutral over all 2024-25 swings.

    Uses the count-neutral grade (league-average run values in the assembly layer) so
    swing_plus is not confounded by count distribution when used as a control alongside
    count-stratified outcomes such as twostrike_rv_penalty.
    """
    con = db.connect("xrv_swings")
    result = con.sql(f"""
        SELECT batter_id, batter_stand, AVG(xrv_grade_neutral) AS swing_plus
        FROM xrv_swings
        WHERE game_year IN {SEASONS_SQL}
        GROUP BY batter_id, batter_stand
    """).df()
    con.close()
    return result


# ─── Section 2 helpers ────────────────────────────────────────────────────────

def _ols_clustered(y: np.ndarray, X: np.ndarray,
                   groups: np.ndarray) -> tuple:
    """
    OLS with clustered sandwich SE (clustered by groups).
    Returns (coefs, se, t_stats, p_vals, n, B).
    B = number of clusters; df = B - 1.
    """
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coefs
    n, k  = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    meat    = np.zeros((k, k))
    unique_groups = np.unique(groups)
    B = len(unique_groups)
    for g in unique_groups:
        mask  = groups == g
        score = X[mask].T @ resid[mask]
        meat += np.outer(score, score)
    meat *= B / (B - 1)  # small-sample correction

    vcov    = XtX_inv @ meat @ XtX_inv
    se      = np.sqrt(np.clip(np.diag(vcov), 0, None))
    t_stats = np.where(se > 0, coefs / se, np.nan)
    p_vals  = np.where(
        se > 0,
        [2 * float(stats.t.sf(abs(t), df=B - 1)) for t in t_stats],
        np.nan,
    )
    return coefs, se, t_stats, p_vals, n, B


_AXIS_PENALTIES = [
    # (penalty_col, primary_treat_col, scope_label, primary_treat_label)
    ("twostrike_rv_penalty",  "adj_count",     "Two-strike",          "adj_count"),
    ("gamestate_rv_penalty",  "adj_gamestate", "Any runner vs empty", "adj_gamestate"),
    ("platoon_rv_penalty",    "adj_platoon",   "Arm-side (same-hand)","adj_platoon"),
]


def run_ols(adj: pd.DataFrame) -> list:
    """
    Between-batter OLS for each situational penalty against its axis + controls.

    For each penalty in _AXIS_PENALTIES, fits:
        penalty_z ~ treat_z + swing_plus_z + repertoire_pctile_z

    Primary treatment is the axis-specific column; composite adjustability is also
    tested as a secondary row. adj_pitch is excluded from the headline (reactive axis —
    hitters cannot choose pitch type before the pitch is thrown).

    All variables z-scored. Clustered SE by batter_id. Returns markdown lines.
    """
    rep = pd.read_parquet(DATA / "repertoire_scores.parquet",
                          columns=KEY + ["repertoire_pctile"])
    base = adj.merge(rep, on=KEY, how="inner")

    def _z(s): return (s - s.mean()) / s.std()

    rows = []
    for penalty_col, primary_treat, scope_label, _ in _AXIS_PENALTIES:
        df = base.dropna(subset=[penalty_col, primary_treat, "swing_plus", "repertoire_pctile"])
        if len(df) < 50:
            print(f"  OLS skipped ({scope_label} / {penalty_col}): only {len(df)} complete rows")
            continue
        print(f"  OLS {scope_label} n={len(df)}")

        y      = _z(df[penalty_col]).to_numpy(float)
        sp_z   = _z(df["swing_plus"]).to_numpy(float)
        rp_z   = _z(df["repertoire_pctile"]).to_numpy(float)
        groups = df["batter_id"].to_numpy()

        for treat_col in [primary_treat, "adjustability"]:
            label = treat_col if treat_col == primary_treat else "composite adjustability"
            t_z = _z(df[treat_col]).to_numpy(float)
            X   = np.column_stack([np.ones(len(df)), sp_z, rp_z, t_z])
            coefs, se, t_stats, p_vals, n, B = _ols_clustered(y, X, groups)
            θ, θ_se, θ_t, θ_p = coefs[-1], se[-1], t_stats[-1], p_vals[-1]
            rows.append(dict(
                scope=scope_label,
                treatment=label,
                penalty=penalty_col,
                theta=round(float(θ), 4),
                se=round(float(θ_se), 4),
                ci_lo=round(float(θ - 1.96 * θ_se), 4),
                ci_hi=round(float(θ + 1.96 * θ_se), 4),
                t=round(float(θ_t), 2),
                p=round(float(θ_p), 4),
                n=n, B=B,
            ))
            print(f"    {scope_label} / {label}: θ={θ:+.4f}  SE={θ_se:.4f}  t={θ_t:.2f}  p={θ_p:.4f}")

    tab = pd.DataFrame(rows).drop(columns=["penalty"])
    lines = [
        "## Between-batter OLS: multi-axis situational penalties\n",
        (f"2024-25, ≥{MIN_SWINGS} swings per (batter, stand). N varies by axis (see table). "
         "Each outcome is a per-batter weighted mean of (situation Δ) within matched cells "
         "(pitch_type × zone for two-strike; + strikes bucket for game-state and platoon). "
         "All variables z-scored. Clustered SE by batter_id.\n"),
        "**Controls:** swing_plus (mean xrv_grade_neutral), repertoire_pctile.\n",
        "Positive θ = more adjustable batters have a *less negative* (better) penalty.\n",
        "**Notes:** (1) adj_pitch excluded — pitch-type identification is reactive "
        "(hitters cannot choose arm slot before the pitch is thrown), not a volitional lever. "
        "(2) Platoon N excludes switch-hitter units: within a (batter_id, batter_stand) unit, "
        "switch hitters face almost exclusively opposite-hand pitchers, leaving near-zero "
        "same-hand swings; those units do not survive the ≥3/≥3 cell filter.\n",
        tab.to_markdown(index=False),
        "",
    ]
    return lines


# ─── Section 3: swing-level DML (fallback) ────────────────────────────────────

TREATMENTS = ["adjustability", "adj_count", "adj_gamestate", "adj_pitch", "adj_platoon"]
AXES = {
    "count":     ["balls", "strikes"],
    "gamestate": ["base_state", "outs_when_up"],
    "pitch":     ["pitch_group"],
    "platoon":   ["pitcher_throws"],
}

N_INNER_FOLDS = 5
N_OUTER_FOLDS = 5
SEED          = 42
CORR_WARN     = 0.6
MAX_STD_SHIFT = 1.5

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
    random_state=SEED, verbosity=0,
)

CONFOUNDER_CAT = ["base_state", "pitch_group", "pitcher_throws"]
CONFOUNDER_NUM = ["px", "pz", "px2", "pz2", "pxpz", "balls", "strikes",
                  "outs_when_up", "pitcher_quality"]


def build_axis_dummies(group: pd.DataFrame, axes: dict) -> tuple:
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
    axis_names = list(axes.keys())
    records = []

    for (batter_id, stand), group in df.groupby(KEY, sort=False):
        if len(group) < MIN_SWINGS:
            continue

        group = group.reset_index(drop=True)
        n = len(group)

        loc_full              = location_design(group)
        sit_full, axis_slices = build_axis_dummies(group, axes)
        n_loc                 = loc_full.shape[1]
        X_full                = np.column_stack([loc_full, sit_full])

        T = {name: np.full(n, np.nan) for name in ["composite"] + axis_names}

        kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)
        for train_idx, val_idx in kf.split(X_full):
            X_train, X_val = X_full[train_idx], X_full[val_idx]
            dial_shifts = {name: np.zeros((len(val_idx), len(DIALS)))
                           for name in ["composite"] + axis_names}

            for d_idx, dial in enumerate(DIALS):
                y_train = group.loc[train_idx, dial].to_numpy(float)
                dial_sd = float(y_train.std())
                if dial_sd == 0:
                    continue
                coefs, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
                sit_coefs = coefs[n_loc:]

                raw = X_val[:, n_loc:] @ sit_coefs
                dial_shifts["composite"][:, d_idx] = np.abs(
                    np.clip(raw / dial_sd, -MAX_STD_SHIFT, MAX_STD_SHIFT)
                )
                for axis_name, slc in axis_slices.items():
                    axis_coefs = sit_coefs[slc]
                    raw_axis   = X_val[:, n_loc + slc.start: n_loc + slc.stop] @ axis_coefs
                    dial_shifts[axis_name][:, d_idx] = np.abs(
                        np.clip(raw_axis / dial_sd, -MAX_STD_SHIFT, MAX_STD_SHIFT)
                    )

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
    return df.groupby("pitcher_id")["delta_run_exp"].mean().rename("pitcher_quality")


def validate_treatments(treats: pd.DataFrame) -> float:
    agg = treats.groupby(KEY)["T_composite"].mean().rename("T_composite_agg").reset_index()
    ref = pd.read_parquet(DATA / "adjustability.parquet", columns=KEY + ["adjustability"])
    merged = agg.merge(ref, on=KEY, how="inner")
    r = float(merged["T_composite_agg"].corr(merged["adjustability"]))
    print(f"  Validation: swing-level composite vs batter-level adjustability r={r:.3f}  "
          f"(n={len(merged)})")
    if r < CORR_WARN:
        print(f"  WARNING: r={r:.3f} < {CORR_WARN} — treatment may have diverged from intent")
    return r


def assemble_analysis_df(df: pd.DataFrame, treats: pd.DataFrame) -> tuple:
    df = df.reset_index(drop=True)
    df["swing_idx"] = df.groupby(KEY).cumcount()

    merged = df.merge(treats, on=KEY + ["swing_idx"], how="inner")
    merged["px2"]  = merged["px"] ** 2
    merged["pz2"]  = merged["pz"] ** 2
    merged["pxpz"] = merged["px"] * merged["pz"]

    cat_dummies = pd.get_dummies(merged[CONFOUNDER_CAT].astype(str), drop_first=True)
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
    Swing-level DML. Robinson (1988) partial linear model.
    Known limitation: per-axis treatments are partially constructed from the same
    variables used as confounders, which can inflate r2_T and compress θ.
    Retained as a robustness check; between-batter OLS (Section 2) is the headline.
    """
    df = (df.dropna(subset=[treatment_col, outcome_col] + confounder_cols)
            .copy()
            .reset_index(drop=True))

    for col in [treatment_col, outcome_col] + confounder_cols:
        df[col] = df[col] - df.groupby("batter_id")[col].transform("mean")

    T_raw = df[treatment_col].to_numpy(float)
    Y_raw = df[outcome_col].to_numpy(float)
    T_std, Y_std = T_raw.std(), Y_raw.std()
    T = T_raw / T_std if T_std > 0 else T_raw
    Y = Y_raw / Y_std if Y_std > 0 else Y_raw

    X          = df[confounder_cols].to_numpy(float)
    batter_ids = df["batter_id"].to_numpy()
    n          = len(df)

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

    psi          = T_resid * (Y_resid - theta * T_resid)
    cluster_sums = pd.Series(psi, index=batter_ids).groupby(level=0).sum()
    B            = len(cluster_sums)
    mean_sq_T    = float(np.mean(T_resid ** 2))
    var_theta    = float((cluster_sums ** 2).sum()) / (mean_sq_T ** 2) / n ** 2 * (B / (B - 1))
    se           = float(np.sqrt(max(var_theta, 0.0)))

    t = theta / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=B - 1)) if se > 0 else np.nan

    r2_T = float(1 - np.sum(T_resid**2) / np.sum((T - T.mean())**2))
    r2_Y = float(1 - np.sum(Y_resid**2) / np.sum((Y - Y.mean())**2))

    return theta, se, t, p, n, r2_T, r2_Y


def run_dml(analysis_df: pd.DataFrame, confounder_cols: list) -> list:
    """Run all DML specs and return markdown lines."""
    treat_col = {
        "adjustability": "T_composite",
        "adj_count":     "T_count",
        "adj_gamestate": "T_gamestate",
        "adj_pitch":     "T_pitch",
        "adj_platoon":   "T_platoon",
    }
    df_2k = analysis_df[analysis_df["strikes"] == 2].copy()

    specs = [
        ("Season-wide", analysis_df, "delta_run_exp", "Run value per swing"),
        ("Season-wide", analysis_df, "is_whiff",      "Whiff rate"),
        ("Two-strike",  df_2k,       "delta_run_exp", "Two-strike run value"),
        ("Two-strike",  df_2k,       "is_whiff",      "Two-strike whiff rate"),
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
            print(f"    θ={theta:+.4f}  SE={se:.4f}  t={t:.2f}  p={p:.4f}"
                  f"  r2_T={r2_T:.3f}  r2_Y={r2_Y:.3f}")

    tab = pd.DataFrame(rows)
    lines = [
        "\n---\n",
        "## Swing-level DML (robustness / fallback)\n",
        (f"N ≈ {len(analysis_df):,} swings (2024-25, ≥{MIN_SWINGS} swings per unit). "
         "Treatment = per-swing fitted situational shift magnitude from within-batter "
         f"5-fold cross-fitting, averaged across 3 dials in within-batter SD units. "
         "Robinson (1988) partial linear model. Within-batter demeaning for batter FE. "
         "GroupKFold (batter_id) XGBoost nuisance models. Clustered SE by batter.\n"),
        "**Note:** per-axis treatments (T_count, T_pitch, etc.) share variables with confounders, "
        "which may inflate r2_T and compress θ toward zero. "
        "The between-batter OLS above is the primary specification.\n",
        tab.to_markdown(index=False),
        "",
    ]
    return lines


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Section 1: matched penalties ──
    print("=== Section 1: matched situational penalties ===")
    two_strike  = compute_matched_penalties()
    game_state  = compute_gamestate_penalty()
    platoon     = compute_platoon_penalty()
    swing_plus  = compute_swing_plus()
    print(f"  two-strike:  {len(two_strike)} units")
    print(f"  game-state:  {len(game_state)} units (any runner vs empty)")
    print(f"  platoon:     {len(platoon)} units (switch hitters excluded — near-zero same_hand within unit)")

    adj = pd.read_parquet(DATA / "adjustability.parquet")
    stale = [
        "twostrike_xrv_penalty", "twostrike_rv_penalty", "twostrike_whiff_penalty",
        "matched_n_2k", "matched_cells",
        "risp_0out_rv_penalty", "risp_0out_whiff_penalty", "risp_0out_n", "risp_0out_cells",
        "dp_rv_penalty", "dp_whiff_penalty", "dp_n", "dp_cells",
        "gamestate_rv_penalty", "gamestate_whiff_penalty", "gamestate_n", "gamestate_cells",
        "platoon_rv_penalty", "platoon_whiff_penalty", "platoon_n", "platoon_cells",
        "swing_plus",
    ]
    adj = adj.drop(columns=[c for c in stale if c in adj.columns])
    adj = (adj
           .merge(two_strike, on=KEY, how="left")
           .merge(game_state,  on=KEY, how="left")
           .merge(platoon,     on=KEY, how="left")
           .merge(swing_plus,  on=KEY, how="left"))
    adj.to_parquet(DATA / "adjustability.parquet", index=False)

    for col in ["twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty"]:
        sub = adj.dropna(subset=[col])
        print(f"  {col}: n={len(sub)}  mean={sub[col].mean():.4f}  median={sub[col].median():.4f}")

    # ── Section 2: between-batter OLS ──
    print("\n=== Section 2: between-batter OLS ===")
    ols_lines = run_ols(adj)

    out = ROOT / "results" / "adjustability_value.md"
    header = [
        "# Adjustability value\n",
        (f"2024-25, ≥{MIN_SWINGS} swings per (batter, stand). "
         "**Section 1:** matched situational penalties (empirical delta_run_exp). "
         "Two-strike: within pitch_type × zone. "
         "Game-state (any runner vs empty) and platoon: within pitch_type × zone × strikes. "
         "**Section 2:** between-batter OLS with count-neutral swing_plus (xrv_grade_neutral) "
         "as Swing+ control. "
         "**Section 3 (DML):** fallback robustness check.\n"),
    ]
    out.write_text("\n".join(header + ols_lines), encoding="utf-8")
    print(f"  Wrote {out}")

    # ── Section 3: DML (fallback, ~30 min) ──
    # Commented out by default — run manually when needed.
    # Known limitation: per-axis treatments (T_count, T_pitch) share variables with the
    # confounder set, which may inflate r2_T and compress θ. Between-batter OLS (Section 2)
    # is the headline; DML is a robustness check only.
    #
    # To run: uncomment the block below and execute `python src/adjustability_value.py`
    #
    # print("\n=== Section 3: swing-level DML (fallback, slow) ===")
    # print("Loading swings...")
    # df = load_swings()
    # print(f"  {len(df):,} swings")
    # pq = build_pitcher_quality(df)
    # df = df.merge(pq.reset_index(), on="pitcher_id", how="left")
    # print("Building swing treatments...")
    # treats = build_swing_treatments(df, AXES)
    # validate_treatments(treats)
    # print("Assembling analysis DataFrame...")
    # analysis_df, confounder_cols = assemble_analysis_df(df, treats)
    # print(f"  All swings n={len(analysis_df):,}  |  Two-strike n={analysis_df[analysis_df['strikes']==2].shape[0]:,}")
    # dml_lines = run_dml(analysis_df, confounder_cols)
    # with open(out, "a", encoding="utf-8") as f:
    #     f.write("\n".join(dml_lines))

    print(f"\nFinal output: {out}")


if __name__ == "__main__":
    main()
