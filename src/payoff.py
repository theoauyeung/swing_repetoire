"""Adjustment-payoff test (Facet-2): does swing adjustability improve outcomes?

Two tests, and the CONTRAST between them is the finding:

  (1) SEASON-WIDE (batter level, usage-weighted stances) — production ~ adjustability + adj_count +
      repertoire_plus + k + swing_plus + logn.  A two-strike compression skill is exercised on ~15%
      of swings, so it should WASH OUT here. This is the foil that rules out "adjusters are just good
      hitters": if adjustability doesn't help season-wide but does at two strikes, it's a targeted
      skill, not general quality leaking in.
  (2) CONDITIONAL TWO-STRIKE ((batter, stand) level) — the two-strike PENALTY (how much worse the
      outcome gets at two strikes, measured within pitch location x type) regressed on the adjustment
      axes + controls, INCLUDING an `adj_count x swing_plus` interaction: is the two-strike protection
      larger for high- or low-quality swings?

Outcome is REALIZED run value (`delta_run_exp`) — the honest payoff. We deliberately do NOT use the
model's expected `xrv` as the outcome: `xrv` is fit on the same trait dials `adj_count` is built from,
so an xrv-based penalty regressed on adj_count would test "does our model reward the adjusted shape,"
not "do real outcomes improve." xRV enters only as `swing_plus` (mean xrv_grade), the swing-QUALITY
control — its correct role. `is_whiff` is the mechanism outcome (the contact channel).

Two penalty estimators per unit (2024-25, >= MIN_SWINGS):
  COARSE FE — within-location(3x3)xpitch-group FE slope of the outcome on two_strike.
  MATCHED   — each hitter's 2-strike swings vs his OWN early swings in the same exact pitch_type x
              Statcast plate_zone (ATT-weighted; cells need >= MATCH_MIN in both groups). Nets out the
              two-strike pitch-mix shift far more finely; `coverage` = frac of 2K swings matched.

Adjustability metric is v3 (adjustability.py): unsigned adjusted-R^2 magnitudes. adj_count is the
count axis (matched to the two-strike mechanism); adj_pitch enters alongside so a count payoff isn't
pitch adjustment relabeled. adj_gamestate is dropped (YoY-unreliable, r=0.19).

CAVEATS: observational; matched version controls pitch type + zone but not release velocity (not in
swings_model) or sequencing; adj_count is unsigned (payoff => net adaptive on average); take-decision
quality uncontrolled (research-design Limitation #3); a seasonal wOBA/wRC+ variant needs the DB (VPN).

Input:  data/swings_model.parquet, data/adjustability.parquet, data/repertoire_scores.parquet,
        data/xrv_swings.parquet
Output: results/payoff.md (committed); prints tables.

Run:  python src/payoff.py
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


def ols(y, X, names):
    """Plain OLS with classical SEs on standardized inputs. Returns (tidy table, R^2, n)."""
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


def fe_slope(g, feat, treat, cell):
    """Within-cell (demeaned) OLS slope of feat on treat."""
    grp = g.groupby(cell)
    rt = g[treat].to_numpy() - grp[treat].transform("mean").to_numpy()
    rf = g[feat].to_numpy() - grp[feat].transform("mean").to_numpy()
    v = float((rt * rt).sum())
    return float((rf * rt).sum() / v) if v > 0 else np.nan


def matched_penalty(g, feat, stratum):
    """ATT-matched two-strike penalty: within each pitch_type x zone cell with >= MATCH_MIN swings in
    BOTH groups, mean(2K) - mean(early); averaged weighted by the cell's 2-strike swing count.
    Returns (penalty, coverage = frac of 2K swings that matched)."""
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


def swing_plus_unit():
    """Per-(batter, stand) swing quality = mean xrv_grade over 2024-25."""
    swp = pd.read_parquet(DATA / "xrv_swings.parquet", columns=KEY + ["game_year", "xrv_grade"])
    return (swp[swp.game_year.isin(SEASONS)].groupby(KEY)["xrv_grade"].mean()
            .rename("swing_plus").reset_index())


# ------------------------------------------------------------------ season-wide (the null foil)
def season_wide():
    cr = pd.read_parquet(DATA / "adjustability.parquet",
                         columns=KEY + ["n_swings", "adjustability", "adj_count"])
    rep = pd.read_parquet(DATA / "repertoire_scores.parquet", columns=KEY + ["repertoire_plus", "k"])
    u = cr.merge(rep, on=KEY).merge(swing_plus_unit(), on=KEY)
    w = u["n_swings"].to_numpy(float)                                   # usage-weight stances -> batter
    cols = ["adjustability", "adj_count", "repertoire_plus", "k", "swing_plus"]
    for c in cols:
        u[c] *= w
    g = u.groupby("batter_id").agg(**{"n_swings": ("n_swings", "sum"),
                                      **{c: (c, "sum") for c in cols}}).reset_index()
    for c in cols:
        g[c] /= g["n_swings"]

    s = pd.read_parquet(DATA / "swings_model.parquet",
                        columns=["batter_id", "game_year", "delta_run_exp", "xwoba"])
    s = s[s.game_year.isin(SEASONS)]
    prod = pd.concat([s.groupby("batter_id")["delta_run_exp"].mean().rename("rv_per_swing"),
                      s.dropna(subset=["xwoba"]).groupby("batter_id")["xwoba"].mean().rename("xwobacon")],
                     axis=1).reset_index()
    df = g.merge(prod, on="batter_id", how="inner")
    df = df[df["n_swings"] >= MIN_SWINGS].copy()
    df["logn"] = np.log(df["n_swings"])
    preds = ["adjustability", "adj_count", "repertoire_plus", "k", "swing_plus", "logn"]
    z = lambda c: (df[c] - df[c].mean()) / df[c].std()

    L = ["## Test 1 — Season-wide (the null foil)\n",
         f"Batter level, usage-weighted stances, MLB {SEASONS[0]}-{SEASONS[1]}, >={MIN_SWINGS} tracked "
         f"swings (**n={len(df)}**). Realized production; predictors standardized. **Expected null** — "
         "a two-strike skill washes out over a full season; this rules out 'adjusters are just good "
         "hitters'.\n"]
    for outcome, desc in [("rv_per_swing", "mean delta_run_exp/swing"), ("xwobacon", "xwOBA on contact")]:
        d = df.dropna(subset=[outcome])
        zz = lambda c: (d[c] - d[c].mean()) / d[c].std()
        X = np.column_stack([np.ones(len(d))] + [zz(c).to_numpy() for c in preds])
        y = ((d[outcome] - d[outcome].mean()) / d[outcome].std()).to_numpy()
        tab, r2, n = ols(y, X, ["intercept"] + preds)
        L += [f"### {outcome} ({desc}) — n={n}, R²={r2:.3f}", tab.to_markdown(index=False), ""]
        print(f"\n[season] {outcome}  (n={n}, R^2={r2:.3f})\n{tab.to_string(index=False)}")
    return L


# ------------------------------------------------------------------ conditional two-strike (the signal)
def two_strike():
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
    s["zone_pitch"] = s["pitch_type"].astype(str) + "|" + s["plate_zone"].astype(str)  # finer matched cell
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
    pen = pd.DataFrame(rows)

    cr = pd.read_parquet(DATA / "adjustability.parquet",
                         columns=KEY + ["n_swings", "adjustability", "adj_count", "adj_pitch"])
    rep = pd.read_parquet(DATA / "repertoire_scores.parquet", columns=KEY + ["repertoire_plus"])
    df = pen.merge(cr, on=KEY).merge(rep, on=KEY).merge(swing_plus_unit(), on=KEY)
    df["logn"] = np.log(df["n_swings"])
    preds = ["adj_count", "adj_pitch", "swing_plus", "repertoire_plus", "logn"]

    L = ["## Test 2 — Conditional two-strike (the signal)\n",
         f"(batter, stand) level, 2024-25, >={MIN_SWINGS} swings (**n={len(df)}**). Realized-RV "
         "two-strike penalty regressed on the adjustment axes + controls + an **adj_count × swing_plus "
         "interaction** (does the protection scale with swing quality?). Penalty via **COARSE FE** "
         "(`penalty_*`, within 3x3-location x pitch-group) and **MATCHED** (`matched_*`, own early "
         "swings, same pitch_type x zone). Predictors standardized; the interaction is the product of "
         "the standardized adj_count and swing_plus. **Positive `adj_count` on run-value penalty = "
         "count adjustment is net protective.**\n",
         f"Matched coverage: mean {df.coverage.mean()*100:.0f}% of 2-strike swings matched (median "
         f"{df.coverage.median()*100:.0f}%). League avg penalties — FE rv {df.penalty_rv.mean():+.4f}, "
         f"matched rv {df.matched_rv.mean():+.4f} run value/swing at 2K (everyone drops).\n"]
    outs = [("penalty_rv", "COARSE FE, run value — higher = more resilient"),
            ("matched_rv", "MATCHED, run value — higher = more resilient"),
            ("penalty_whiff", "COARSE FE, whiff — lower = fewer extra whiffs"),
            ("matched_whiff", "MATCHED, whiff — lower = fewer extra whiffs")]
    for outcome, arrow in outs:
        d = df.dropna(subset=[outcome])
        zz = lambda c: ((d[c] - d[c].mean()) / d[c].std()).to_numpy()
        Z = {c: zz(c) for c in preds}
        inter = Z["adj_count"] * Z["swing_plus"]
        X = np.column_stack([np.ones(len(d))] + [Z[c] for c in preds] + [inter])
        y = ((d[outcome] - d[outcome].mean()) / d[outcome].std()).to_numpy()
        tab, r2, n = ols(y, X, ["intercept"] + preds + ["adj_count:swing_plus"])
        L += [f"### {outcome}  ({arrow})  — n={n}, R²={r2:.3f}", tab.to_markdown(index=False), ""]
        print(f"\n[2K] {outcome}  (n={n}, R^2={r2:.3f})\n{tab.to_string(index=False)}")

    axes = ["adj_count", "adj_pitch", "adjustability"]
    cors = {f"{a}|{o}": round(float(df[a].corr(df[o])), 3)
            for o in ["penalty_rv", "matched_rv", "penalty_whiff", "matched_whiff"] for a in axes}
    L += ["### Zero-order correlations of each adjustment axis with the two-strike penalties",
          "\n".join(f"- {k}: {v:+.3f}" for k, v in cors.items()), ""]
    return L


def main():
    L = ["# Adjustment-payoff test — does swing adjustability improve outcomes?\n",
         "Realized run value (`delta_run_exp`) throughout; xRV enters only as the `swing_plus` quality "
         "control (see script header for why the model's expected xrv is NOT used as the outcome). "
         "Observational draft — caveats in `src/payoff.py`.\n"]
    L += two_strike() + season_wide()
    (ROOT / "results" / "payoff.md").write_text("\n".join(L), encoding="utf-8")
    print("\nwrote results/payoff.md")


if __name__ == "__main__":
    main()
