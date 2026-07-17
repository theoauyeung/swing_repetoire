"""Adjustment-payoff regression (DRAFT / Facet-2 test).

Does adjustability predict better production, after controlling for repertoire and swing quality?
Hitter-level OLS:  production ~ count_adj + repertoire_plus + k + swing_plus (+ log n_swings),
all predictors standardized so coefficients are comparable (std-beta / partial-association units).

  production   = per-swing run value (mean delta_run_exp, the RE24 anchor) and xwOBAcon (mean xwoba
                 on balls in play), MLB 2024-25 — both LOCAL from swings_model (no DB needed).
  count_adj    = context-responsiveness / adjustability (context_response.parquet)
  repertoire_plus, k = repertoire width + shape count (repertoire_scores.parquet)  [controls]
  swing_plus   = mean xrv_grade, a swing-quality proxy (xrv_swings.parquet)         [quality control]

Per-(batter, stand) metrics are usage-weighted to batter level.

CAVEATS — this is a draft, not a causal claim:
  - observational; no random assignment of "adjustability".
  - production is contemporaneous with the swing metrics (same seasons) -> association, not effect.
  - stances pooled by usage weight; switch hitters approximated.
  - the adjustability construct itself is still under review (docs/adjustability-decontamination.md:
    tilt≈0, vert ambiguous) — treat coefficients as directional, not final.
  - swing_plus (model xRV) and delta_run_exp (realized RV) are related though not identical
    (batter-level r≈0.37); read swing_plus as a quality control, not an independent instrument.
  - a seasonal wOBA / wRC+ variant (season_stats_hitting) needs the DB (VPN) — add when reachable.

Output: prints the tables; writes results/payoff_regression.md (committed).

Run:  python src/payoff_regression.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEASONS = [2024, 2025]
MIN_SWINGS = 400   # matches the context_response qualification


def ols(y, X, names):
    """Plain OLS with classical SEs on standardized inputs. Returns (tidy table, R^2, n)."""
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    covb = (sse / (n - p)) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(covb))
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), n - p)
    r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
    tab = pd.DataFrame({"term": names, "std_beta": beta.round(3), "se": se.round(3),
                        "t": t.round(2), "p": pval.round(4)})
    return tab, r2, n


def batter_level():
    """Usage-weight the per-(batter, stand) predictors to batter level."""
    cr = pd.read_parquet(DATA / "context_response.parquet",
                         columns=["batter_id", "batter_stand", "n_swings", "count_adj"])
    rep = pd.read_parquet(DATA / "repertoire_scores.parquet",
                          columns=["batter_id", "batter_stand", "repertoire_plus", "k"])
    swp = pd.read_parquet(DATA / "xrv_swings.parquet",
                          columns=["batter_id", "batter_stand", "game_year", "xrv_grade"])
    swp = (swp[swp.game_year.isin(SEASONS)].groupby(["batter_id", "batter_stand"])["xrv_grade"]
           .mean().rename("swing_plus").reset_index())
    u = cr.merge(rep, on=["batter_id", "batter_stand"]).merge(swp, on=["batter_id", "batter_stand"])
    w = u["n_swings"].to_numpy(float)
    cols = ["count_adj", "repertoire_plus", "k", "swing_plus"]
    for c in cols:
        u[c] *= w
    g = u.groupby("batter_id").agg(**{"n_swings": ("n_swings", "sum"),
                                      **{c: (c, "sum") for c in cols}}).reset_index()
    for c in cols:
        g[c] /= g["n_swings"]
    return g


def production_local():
    """Batter-level production from swings_model: mean delta_run_exp (all swings) + xwOBAcon (BIP)."""
    s = pd.read_parquet(DATA / "swings_model.parquet",
                        columns=["batter_id", "game_year", "delta_run_exp", "xwoba"])
    s = s[s.game_year.isin(SEASONS)]
    rv = s.groupby("batter_id")["delta_run_exp"].mean().rename("rv_per_swing")
    xw = s.dropna(subset=["xwoba"]).groupby("batter_id")["xwoba"].mean().rename("xwobacon")
    return pd.concat([rv, xw], axis=1).reset_index()


def main():
    df = batter_level().merge(production_local(), on="batter_id", how="inner")
    df = df[df["n_swings"] >= MIN_SWINGS].copy()
    df["logn"] = np.log(df["n_swings"])
    preds = ["count_adj", "repertoire_plus", "k", "swing_plus", "logn"]
    z = lambda s: (s - s.mean()) / s.std()

    L = ["# Adjustment-payoff regression (DRAFT)\n",
         f"Hitter-level OLS, MLB {SEASONS[0]}-{SEASONS[1]}, batters with ≥{MIN_SWINGS} tracked swings "
         f"(**n = {len(df)}**). Predictors standardized (std-beta, comparable across terms). Production "
         "is local per-swing run value. Per-(batter, stand) metrics usage-weighted to batter level. "
         "**Observational, contemporaneous — association, not causal effect.** Caveats in "
         "`src/payoff_regression.py`.\n",
         "Question: does `count_adj` (adjustability) carry a positive partial association with "
         "production after controlling for repertoire width (`repertoire_plus`), shape count (`k`), "
         "swing quality (`swing_plus`), and playing time (`logn`)?\n"]

    for outcome, desc in [("rv_per_swing", "mean delta_run_exp per swing"), ("xwobacon", "xwOBA on contact")]:
        d = df.dropna(subset=[outcome])
        X = np.column_stack([np.ones(len(d))] + [z(d[c]).to_numpy() for c in preds])
        tab, r2, n = ols(z(d[outcome]).to_numpy(), X, ["intercept"] + preds)
        L += [f"## Outcome: {outcome} ({desc})  — n={n}, R²={r2:.3f}", tab.to_markdown(index=False), ""]
        print(f"\nOutcome: {outcome}  (n={n}, R^2={r2:.3f})")
        print(tab.to_string(index=False))

    cors = {c: round(float(df["count_adj"].corr(df[c])), 3)
            for c in ["rv_per_swing", "xwobacon", "repertoire_plus", "swing_plus", "k"]}
    L += ["## count_adj zero-order correlations", "\n".join(f"- {k}: {v:+.3f}" for k, v in cors.items()), ""]
    (ROOT / "results" / "payoff_regression.md").write_text("\n".join(L), encoding="utf-8")
    print("\nzero-order count_adj corrs:", cors)
    print("wrote results/payoff_regression.md")


if __name__ == "__main__":
    main()
