"""Conditional two-strike payoff test (Facet-2, the correctly-targeted payoff).

The season-wide payoff regression (payoff_regression.py) found no adjustability payoff — but a
two-strike compression skill should show up IN two-strike situations, not diluted across a whole
season (~85% non-2-strike swings). This tests the payoff where the skill operates.

Everyone gets worse with two strikes. The question: do hitters who ADJUST their swing at two strikes
(high count_adj) suffer a SMALLER two-strike penalty? Both the skill and the penalty are measured the
same conditional way — within pitch location x type — so we compare like with like.

Per (batter, stand) unit (2024-25, >= MIN_SWINGS):
  penalty_rv    = within-location FE slope of delta_run_exp on two_strike  (run value lost at 2K;
                  negative = production drops; LESS negative = more resilient)
  penalty_whiff = within-location FE slope of is_whiff on two_strike       (extra whiffs at 2K)
Then across units, OLS: penalty ~ count_adj + swing_plus + repertoire_plus + logn (standardized).
Hypothesis: count_adj > 0 on penalty_rv (adjusters lose less run value at two strikes).

CAVEATS: observational; two-strike pitches differ from early-count pitches (pitchers expand the zone),
and the within-location cells only partly absorb that pitch-mix shift; count_adj feature set still
under review (docs/adjustability-decontamination.md). Draft, directional.

Input:  data/swings_model.parquet, data/context_response.parquet, data/repertoire_scores.parquet,
        data/xrv_swings.parquet
Output: results/payoff_twostrike.md (committed); prints tables.

Run:  python src/payoff_twostrike.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KEY = ["batter_id", "batter_stand"]
SEASONS = [2024, 2025]
MIN_SWINGS = 400


def fe_slope(g, feat, treat, cell):
    """Within-cell (demeaned) OLS slope of feat on treat."""
    grp = g.groupby(cell)
    rt = g[treat].to_numpy() - grp[treat].transform("mean").to_numpy()
    rf = g[feat].to_numpy() - grp[feat].transform("mean").to_numpy()
    v = float((rt * rt).sum())
    return float((rf * rt).sum() / v) if v > 0 else np.nan


def ols(y, X, names):
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    covb = (sse / (n - p)) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(covb))
    t = beta / se
    r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
    tab = pd.DataFrame({"term": names, "std_beta": beta.round(3), "se": se.round(3),
                        "t": t.round(2), "p": (2 * stats.t.sf(np.abs(t), n - p)).round(4)})
    return tab, r2, n


def penalties():
    s = pd.read_parquet(DATA / "swings_model.parquet",
                        columns=["game_year", "batter_stand", "batter_id", "strikes",
                                 "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type",
                                 "delta_run_exp", "is_whiff"])
    s = s[s.game_year.isin(SEASONS)].copy()
    pull = np.where(s.batter_stand.to_numpy() == "L", 1.0, -1.0)
    s["px"] = s.plate_x * pull
    s["pz"] = (s.plate_z - s.sz_bot) / (s.sz_top - s.sz_bot)
    s = s.dropna(subset=["px", "pz", "delta_run_exp"])
    lh = np.select([s.px > 0.3, s.px < -0.3], ["in", "out"], default="mid")
    lv = np.select([s.pz > 0.66, s.pz < 0.33], ["hi", "lo"], default="mid")
    grp = s.pitch_type.map({"FF": "FB", "SI": "FB", "FC": "FB", "SL": "brk", "CU": "brk", "KC": "brk",
                            "ST": "brk", "SV": "brk", "SC": "brk", "KN": "brk", "CH": "off",
                            "FS": "off", "FO": "off"}).fillna("other")
    s["loc_pitch"] = pd.Series(lh, index=s.index) + "_" + pd.Series(lv, index=s.index) + "|" + grp
    s["twoK"] = (s.strikes == 2).astype(float)
    s["is_whiff"] = s["is_whiff"].astype(float)
    rows = []
    for (bid, stand), g in s.groupby(KEY, sort=False):
        if len(g) < MIN_SWINGS:
            continue
        rows.append({"batter_id": bid, "batter_stand": stand,
                     "penalty_rv": fe_slope(g, "delta_run_exp", "twoK", "loc_pitch"),
                     "penalty_whiff": fe_slope(g, "is_whiff", "twoK", "loc_pitch")})
    return pd.DataFrame(rows)


def predictors():
    cr = pd.read_parquet(DATA / "context_response.parquet", columns=KEY + ["n_swings", "count_adj"])
    rep = pd.read_parquet(DATA / "repertoire_scores.parquet", columns=KEY + ["repertoire_plus"])
    swp = pd.read_parquet(DATA / "xrv_swings.parquet", columns=KEY + ["game_year", "xrv_grade"])
    swp = (swp[swp.game_year.isin(SEASONS)].groupby(KEY)["xrv_grade"].mean()
           .rename("swing_plus").reset_index())
    return cr.merge(rep, on=KEY).merge(swp, on=KEY)


def main():
    df = penalties().merge(predictors(), on=KEY, how="inner")
    df["logn"] = np.log(df["n_swings"])
    z = lambda c: (df[c] - df[c].mean()) / df[c].std()
    preds = ["count_adj", "swing_plus", "repertoire_plus", "logn"]

    L = ["# Conditional two-strike payoff test (DRAFT)\n",
         f"Do hitters who adjust their swing at two strikes suffer a smaller two-strike penalty? "
         f"Per (batter, stand), 2024-25, ≥{MIN_SWINGS} swings (**n={len(df)}**). `penalty_rv` = "
         "within-location FE slope of run value (delta_run_exp) on two-strike (negative = production "
         "drops at 2K; less negative = resilient). `penalty_whiff` = same for whiff rate. OLS across "
         "hitters, predictors standardized. **Positive `count_adj` on penalty_rv = adjusting pays "
         "off.** Observational draft — see caveats in the script.\n",
         f"League average: penalty_rv = {df.penalty_rv.mean():+.4f} run value/swing at 2K (everyone "
         f"drops), penalty_whiff = {df.penalty_whiff.mean():+.4f}.\n"]

    for outcome, arrow in [("penalty_rv", "higher (less negative) = more resilient"),
                           ("penalty_whiff", "lower = fewer extra whiffs at 2K")]:
        d = df.dropna(subset=[outcome])
        zz = lambda c: (d[c] - d[c].mean()) / d[c].std()
        X = np.column_stack([np.ones(len(d))] + [zz(c).to_numpy() for c in preds])
        y = ((d[outcome] - d[outcome].mean()) / d[outcome].std()).to_numpy()
        tab, r2, n = ols(y, X, ["intercept"] + preds)
        L += [f"## Outcome: {outcome}  ({arrow})  — n={n}, R²={r2:.3f}", tab.to_markdown(index=False), ""]
        print(f"\nOutcome: {outcome}  (n={n}, R^2={r2:.3f})")
        print(tab.to_string(index=False))

    cors = {o: round(float(df["count_adj"].corr(df[o])), 3) for o in ["penalty_rv", "penalty_whiff"]}
    L += ["## count_adj zero-order correlations with the two-strike penalties",
          "\n".join(f"- {k}: {v:+.3f}" for k, v in cors.items()), ""]
    (ROOT / "results" / "payoff_twostrike.md").write_text("\n".join(L), encoding="utf-8")
    print("\nzero-order:", cors, "\nwrote results/payoff_twostrike.md")


if __name__ == "__main__":
    main()
