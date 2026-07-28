"""DML (Double Machine Learning) causal estimate: does swing adjustability improve outcomes?

Treatments: all four component axes + composite:
  adjustability  — composite (all situation over location-only)
  adj_count      — count axis unique contribution
  adj_gamestate  — game-state axis unique contribution
  adj_pitch      — pitch-type axis unique contribution
  adj_platoon    — pitcher-handedness axis unique contribution
Each axis is run as a separate DML — marginal effect conditional on non-adjustment confounders.

Confounders: swing_plus, log(n_swings), repertoire_plus, batter handedness,
             pitcher quality faced (swing-weighted mean pitcher Δrun-exp from this dataset),
             whiff rate (contact-skill proxy).
Outcomes  : season-wide (rv_per_swing, woba_lw) and two-strike (two_strike_rv_delta, two_strike_whiff_delta).

Method: partial linear model. XGBoost nuisance models (5-fold cross-fitting)
residualize both treatment and outcome on confounders independently. θ = OLS(Y_resid ~ T_resid).
Treatment and outcome both standardized so θ is a standardized partial effect. SE from the
DML influence-function sandwich estimator — no bootstrap required.

Caveat: conditional unconfoundedness assumed. Unmeasured confounders (true plate discipline,
pitcher sequencing, role/contract incentives) remain a threat.

Output: results/adjustability_value.md
Run   : python src/adjustability_value.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KEY  = ["batter_id", "batter_stand"]
SEASONS    = [2024, 2025]
MIN_SWINGS = 400
MATCH_MIN  = 3
TREATMENTS = ["adjustability", "adj_count", "adj_gamestate", "adj_pitch", "adj_platoon"]
N_FOLDS    = 5
SEED       = 42

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
    random_state=SEED, verbosity=0,
)

PA_TO_LW = {
    "home_run": "home_run", "triple": "triple", "double": "double", "single": "single",
    "strikeout": "strikeout", "strikeout_double_play": "strikeout",
    "field_out": "out_in_play", "force_out": "out_in_play", "double_play": "out_in_play",
    "grounded_into_double_play": "out_in_play", "sac_fly": "out_in_play",
    "sac_fly_double_play": "out_in_play", "fielders_choice": "out_in_play",
    "fielders_choice_out": "out_in_play", "triple_play": "out_in_play",
    "field_error": "other", "catcher_interf": "other",
}

# Confounders without logn (recomputed post-aggregation) and without stand_L at unit level
_CONF_WTAVG = ["swing_plus", "repertoire_plus", "stand_L", "pitcher_quality", "whiff_rate"]
CONFOUNDER_COLS = _CONF_WTAVG + ["logn"]


def _two_strike_penalty(group, feature):
    """Matched two-strike minus early-count difference for one batter's swings."""
    weighted_sum = 0.0
    total_two_strike_swings = 0.0

    for _, zone_group in group.groupby("zone_pitch", sort=False):
        two_strike  = zone_group.loc[zone_group["twoK"] == 1, feature]
        early_count = zone_group.loc[zone_group["twoK"] == 0, feature]

        if len(two_strike) >= MATCH_MIN and len(early_count) >= MATCH_MIN:
            diff = two_strike.mean() - early_count.mean()
            weighted_sum += len(two_strike) * diff
            total_two_strike_swings += len(two_strike)

    if total_two_strike_swings == 0:
        return np.nan
    return weighted_sum / total_two_strike_swings


def dml(d, treatment, outcome):
    """
    5-fold cross-fitted DML. Treatment and outcome standardized → θ is a standardized
    partial effect. SE from influence-function sandwich estimator (valid under cross-fitting).
    Returns (theta, se, t, p, n, r2_T, r2_Y).
    """
    d = d.dropna(subset=[treatment, outcome] + CONFOUNDER_COLS).copy().reset_index(drop=True)
    n = len(d)

    # Standardize to mean=0, SD=1 so θ is comparable across outcomes
    T_raw = d[treatment].to_numpy(float)
    Y_raw = d[outcome].to_numpy(float)
    T = (T_raw - T_raw.mean()) / T_raw.std()
    Y = (Y_raw - Y_raw.mean()) / Y_raw.std()
    X = d[CONFOUNDER_COLS].to_numpy(float)

    # Cross-fitting: fit separate models for T and Y on confounders, save residuals
    T_resid = np.zeros(n)
    Y_resid = np.zeros(n)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train_idx, val_idx in kf.split(X):
        model_T = XGBRegressor(**XGB_PARAMS)
        model_T.fit(X[train_idx], T[train_idx])
        T_resid[val_idx] = T[val_idx] - model_T.predict(X[val_idx])

        model_Y = XGBRegressor(**XGB_PARAMS)
        model_Y.fit(X[train_idx], Y[train_idx])
        Y_resid[val_idx] = Y[val_idx] - model_Y.predict(X[val_idx])

    # OLS: regress Y residuals on T residuals
    theta = float(np.dot(T_resid, Y_resid) / np.dot(T_resid, T_resid))

    # Sandwich SE via influence function
    influence_scores  = T_resid * (Y_resid - theta * T_resid)
    mean_sq_influence = float(np.mean(influence_scores ** 2))
    mean_sq_T_resid   = float(np.mean(T_resid ** 2))
    var_theta = mean_sq_influence / (mean_sq_T_resid ** 2) / n
    se = float(np.sqrt(max(var_theta, 0.0)))

    t = theta / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - len(CONFOUNDER_COLS) - 1)) if se > 0 else np.nan

    r2_T = float(1 - np.sum(T_resid**2) / np.sum((T - T.mean())**2))
    r2_Y = float(1 - np.sum(Y_resid**2) / np.sum((Y - Y.mean())**2))

    return theta, se, t, p, n, r2_T, r2_Y


def load_swings():
    return pd.read_parquet(
        DATA / "swings_model.parquet",
        columns=KEY + ["game_year", "pitcher_id", "delta_run_exp", "is_whiff",
                       "pa_outcome", "pitch_type", "plate_zone", "strikes"],
    ).query("game_year in @SEASONS")


def build_unit_table(s):
    """Per-(batter, stand) treatment + confounder table."""
    xrv = pd.read_parquet(DATA / "xrv_swings.parquet",
                          columns=KEY + ["game_year", "xrv_grade"])
    swing_plus = (xrv[xrv.game_year.isin(SEASONS)]
                  .groupby(KEY)["xrv_grade"].mean()
                  .rename("swing_plus").reset_index())

    rep = pd.read_parquet(DATA / "repertoire_scores.parquet",
                          columns=KEY + ["repertoire_plus"])

    # Pitcher quality: per-pitcher mean delta_run_exp across all swings they threw.
    # Self-influence is negligible — a batter's ~200-300 swings vs a pitcher's 2k+.
    pitcher_rv = (s.groupby("pitcher_id")["delta_run_exp"]
                   .mean().rename("pitcher_rv").reset_index())
    pq = (s.merge(pitcher_rv, on="pitcher_id")
           .groupby(KEY)["pitcher_rv"].mean()
           .rename("pitcher_quality").reset_index())

    wr = s.groupby(KEY)["is_whiff"].mean().rename("whiff_rate").reset_index()
    ns = s.groupby(KEY).size().rename("n_swings").reset_index()

    adj = pd.read_parquet(DATA / "adjustability.parquet",
                          columns=KEY + TREATMENTS)

    unit = (adj.merge(ns, on=KEY)
               .merge(swing_plus, on=KEY)
               .merge(rep, on=KEY)
               .merge(pq, on=KEY)
               .merge(wr, on=KEY))
    unit["logn"]    = np.log(unit["n_swings"])
    unit["stand_L"] = (unit["batter_stand"] == "L").astype(float)
    return unit[unit["n_swings"] >= MIN_SWINGS].copy()


def aggregate_to_batter(unit):
    """Usage-weight (batter, stand) → batter level for season-wide DML."""
    # Weighted average: multiply each column by swing count, sum across stances, then divide.
    # This collapses switch hitters (who have two rows) into one batter-level row.
    agg_cols = TREATMENTS + _CONF_WTAVG

    weighted = unit.copy()
    for col in agg_cols:
        weighted[col] = weighted[col] * weighted["n_swings"]

    grouped = weighted.groupby("batter_id")[["n_swings"] + agg_cols].sum().reset_index()

    for col in agg_cols:
        grouped[col] = grouped[col] / grouped["n_swings"]

    grouped["logn"] = np.log(grouped["n_swings"])
    return grouped


def season_outcomes(s):
    lw = pd.read_csv(DATA / "linear_weights.csv").set_index("outcome_type")["lw"].to_dict()
    s = s.copy()
    s["woba_lw"] = s["pa_outcome"].map(PA_TO_LW).map(lw)
    rv   = s.groupby("batter_id")["delta_run_exp"].mean().rename("rv_per_swing")
    woba = s.dropna(subset=["woba_lw"]).groupby("batter_id")["woba_lw"].mean().rename("woba_lw")
    return pd.concat([rv, woba], axis=1).reset_index()


def twostrike_outcomes(s):
    cache = DATA / "twostrike_penalties.parquet"
    if cache.exists():
        print("  two-strike: loaded from cache")
        return pd.read_parquet(cache, columns=KEY + ["two_strike_rv_delta", "two_strike_whiff_delta"])

    print("  two-strike: computing...")
    s2 = s.copy()
    s2["twoK"]       = (s2.strikes == 2).astype(float)
    s2["is_whiff"]   = s2.is_whiff.astype(float)
    s2["zone_pitch"] = s2.pitch_type.astype(str) + "|" + s2.plate_zone.astype(str)

    rows = []
    for (bid, stand), g in s2.groupby(KEY, sort=False):
        if len(g) < MIN_SWINGS:
            continue
        rows.append({
            "batter_id":              bid,
            "batter_stand":           stand,
            "two_strike_rv_delta":    _two_strike_penalty(g, "delta_run_exp"),
            "two_strike_whiff_delta": _two_strike_penalty(g, "is_whiff"),
        })
    return pd.DataFrame(rows)


def main():
    print("Loading swings...")
    s = load_swings()

    print("Building unit table...")
    unit   = build_unit_table(s)
    batter = aggregate_to_batter(unit)

    print("Building outcomes...")
    s_out  = season_outcomes(s)
    ts_out = twostrike_outcomes(s)

    df_season = batter.merge(s_out, on="batter_id")
    df_2k     = unit.merge(ts_out, on=KEY)
    print(f"Season n={len(df_season)}  |  Two-strike n={len(df_2k)}")

    specs = [
        ("Season-wide",  df_season, "rv_per_swing",           "Run value per swing"),
        ("Season-wide",  df_season, "woba_lw",                "wOBA (swing-ending PAs)"),
        ("Two-strike",   df_2k,     "two_strike_rv_delta",    "Matched 2K run value"),
        ("Two-strike",   df_2k,     "two_strike_whiff_delta", "Matched 2K whiff rate"),
    ]

    rows = []
    for treatment in TREATMENTS:
        for scope, df, outcome, label in specs:
            print(f"DML: {treatment} / {scope} / {outcome}...")
            theta, se, t, p, n, r2_T, r2_Y = dml(df, treatment, outcome)
            rows.append(dict(
                treatment=treatment, outcome=label, scope=scope, n=n,
                theta=round(theta, 4), se=round(se, 4),
                ci_lo=round(theta - 1.96 * se, 4),
                ci_hi=round(theta + 1.96 * se, 4),
                t=round(t, 2), p=round(p, 4),
                r2_T=round(r2_T, 3), r2_Y=round(r2_Y, 3),
            ))
            print(f"  θ={theta:+.4f}  SE={se:.4f}  t={t:.2f}  p={p:.4f}"
                  f"  r2_T={r2_T:.3f}  r2_Y={r2_Y:.3f}")

    tab = pd.DataFrame(rows)

    lines = [
        "# Adjustability value — DML causal estimates\n",
        f"Each of {len(TREATMENTS)} treatments run as a separate DML "
        f"(2024-25, ≥{MIN_SWINGS} swings). Standardized; θ is a standardized partial effect.\n",
        "**Method:** Robinson (1988) partial linear model. XGBoost nuisance models "
        f"({N_FOLDS}-fold cross-fitting). SE from influence-function sandwich estimator.\n",
        "**Confounders (same for all treatments):** swing quality (swing_plus), log playing time, "
        "repertoire width, batter handedness, pitcher quality faced, whiff rate.\n",
        "**Caveat:** conditional unconfoundedness assumed. Unmeasured confounders remain a threat.\n",
        "## Results\n",
        tab.to_markdown(index=False),
        "",
    ]

    out = ROOT / "results" / "adjustability_value.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
