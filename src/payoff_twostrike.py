"""Conditional two-strike payoff test (Facet-2, the correctly-targeted payoff).

The season-wide payoff regression (payoff_regression.py) found no adjustability payoff — but a
two-strike compression skill should show up IN two-strike situations, not diluted across a whole
season (~85% non-2-strike swings). This tests the payoff where the skill operates.

Everyone gets worse with two strikes. The question: do hitters who ADJUST their swing at two strikes
(high count_adj) suffer a SMALLER two-strike penalty? Both the skill and the penalty are measured the
same conditional way — within pitch location x type — so we compare like with like.

Two penalty estimators per (batter, stand) unit (2024-25, >= MIN_SWINGS):
  (A) COARSE FE — within-location(3x3)xpitch-group FE slope of the outcome on two_strike.
  (B) MATCHED (tighter) — match each hitter's two-strike swings to his OWN early-count swings in the
      SAME exact pitch_type x Statcast plate_zone, take the mean outcome gap per matched cell, and
      average (weighted by the hitter's 2-strike swing count = ATT). Only cells with >= MATCH_MIN
      swings in BOTH groups count; `coverage` = fraction of the hitter's 2-strike swings that matched.
      This nets out the two-strike pitch-mix shift (pitchers expand the zone / throw more breaking
      balls) far more finely than the coarse bins, so the residual penalty is "same pitch, same zone,
      worse outcome because it's two strikes."
For each: penalty_rv (run value lost; less-negative = resilient), penalty_whiff (extra whiffs).
Then across units, OLS: penalty ~ count_adj + swing_plus + repertoire_plus + logn (standardized).
Hypothesis: count_adj > 0 on penalty_rv, < 0 on penalty_whiff (adjusters protect better at 2K).

CAVEATS: observational; the matched version controls pitch type + zone but not release velocity
(not in swings_model) or sequencing; count_adj feature set still under review
(docs/adjustability-decontamination.md). Draft, directional.

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
MATCH_MIN = 3     # min swings in BOTH groups for a matched pitch_type x zone cell to count


def fe_slope(g, feat, treat, cell):
    """Within-cell (demeaned) OLS slope of feat on treat."""
    grp = g.groupby(cell)
    rt = g[treat].to_numpy() - grp[treat].transform("mean").to_numpy()
    rf = g[feat].to_numpy() - grp[feat].transform("mean").to_numpy()
    v = float((rt * rt).sum())
    return float((rf * rt).sum() / v) if v > 0 else np.nan


def matched_penalty(g, feat, stratum):
    """ATT-matched two-strike penalty: within each pitch_type x zone cell that has >= MATCH_MIN swings
    in BOTH the 2-strike and early groups, take mean(2K) - mean(early); average weighted by the cell's
    2-strike swing count. Returns (penalty, coverage = frac of 2K swings that matched)."""
    tot2 = float((g["twoK"] == 1).sum())
    num = den = 0.0
    for _, s in g.groupby(stratum, sort=False):
        a = s.loc[s.twoK == 1, feat]
        b = s.loc[s.twoK == 0, feat]
        if len(a) >= MATCH_MIN and len(b) >= MATCH_MIN:
            num += len(a) * (a.mean() - b.mean())
            den += len(a)
    if den == 0:
        return np.nan, 0.0
    return num / den, den / tot2 if tot2 else 0.0


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
                                 "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type", "plate_zone",
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
    # finer matched stratum: exact pitch type x Statcast zone
    s["zone_pitch"] = s["pitch_type"].astype(str) + "|" + s["plate_zone"].astype(str)
    s["twoK"] = (s.strikes == 2).astype(float)
    s["is_whiff"] = s["is_whiff"].astype(float)
    rows = []
    for (bid, stand), g in s.groupby(KEY, sort=False):
        if len(g) < MIN_SWINGS:
            continue
        m_rv, cov = matched_penalty(g, "delta_run_exp", "zone_pitch")
        m_wh, _ = matched_penalty(g, "is_whiff", "zone_pitch")
        rows.append({"batter_id": bid, "batter_stand": stand,
                     "penalty_rv": fe_slope(g, "delta_run_exp", "twoK", "loc_pitch"),
                     "penalty_whiff": fe_slope(g, "is_whiff", "twoK", "loc_pitch"),
                     "matched_rv": m_rv, "matched_whiff": m_wh, "coverage": cov})
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
         f"Per (batter, stand), 2024-25, ≥{MIN_SWINGS} swings (**n={len(df)}**). Two estimators of the "
         "penalty: **COARSE FE** (`penalty_*`, within 3x3-location x pitch-group) and **MATCHED** "
         "(`matched_*`, each hitter's 2-strike swings vs his own early swings in the same exact "
         "pitch_type x Statcast zone — tighter control for the 2-strike pitch-mix shift). OLS across "
         "hitters, predictors standardized. **Positive `count_adj` on run-value penalty = adjusting "
         "pays off.** Observational draft — see caveats in the script.\n",
         f"Matched coverage: mean {df.coverage.mean()*100:.0f}% of 2-strike swings matched (median "
         f"{df.coverage.median()*100:.0f}%). League avg penalties — FE rv {df.penalty_rv.mean():+.4f}, "
         f"matched rv {df.matched_rv.mean():+.4f} run value/swing at 2K (everyone drops).\n"]

    outs = [("penalty_rv", "COARSE FE, run value — higher = more resilient"),
            ("matched_rv", "MATCHED, run value — higher = more resilient"),
            ("penalty_whiff", "COARSE FE, whiff — lower = fewer extra whiffs"),
            ("matched_whiff", "MATCHED, whiff — lower = fewer extra whiffs")]
    for outcome, arrow in outs:
        d = df.dropna(subset=[outcome])
        zz = lambda c: (d[c] - d[c].mean()) / d[c].std()
        X = np.column_stack([np.ones(len(d))] + [zz(c).to_numpy() for c in preds])
        y = ((d[outcome] - d[outcome].mean()) / d[outcome].std()).to_numpy()
        tab, r2, n = ols(y, X, ["intercept"] + preds)
        L += [f"## Outcome: {outcome}  ({arrow})  — n={n}, R²={r2:.3f}", tab.to_markdown(index=False), ""]
        print(f"\nOutcome: {outcome}  (n={n}, R^2={r2:.3f})")
        print(tab.to_string(index=False))

    cors = {o: round(float(df["count_adj"].corr(df[o])), 3)
            for o in ["penalty_rv", "matched_rv", "penalty_whiff", "matched_whiff"]}
    L += ["## count_adj zero-order correlations with the two-strike penalties",
          "\n".join(f"- {k}: {v:+.3f}" for k, v in cors.items()), ""]
    (ROOT / "results" / "payoff_twostrike.md").write_text("\n".join(L), encoding="utf-8")
    print("\nzero-order:", cors, "\nwrote results/payoff_twostrike.md")


if __name__ == "__main__":
    main()
