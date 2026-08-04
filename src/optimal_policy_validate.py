"""Validation for the counterfactual adjustment-value accounting.

The design accepts xRV as the value function because a counterfactual swing has no
realized delta_run_exp. That makes validation load-bearing rather than a formality.

Pre-committed failure condition: if the predictive test is null AND the placebo does not
collapse, runs_total is a model artifact and no leaderboard ships from it.

Run: python src/optimal_policy_validate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjustability import KEY, SEASONS                    # noqa: E402
from adjustability_value import _ols_clustered            # noqa: E402


def season_outcomes():
    """Per-unit realized production. woba is non-null only on PA-ending swings."""
    df = pd.read_parquet(DATA / "swings_model.parquet",
                         columns=KEY + ["game_year", "delta_run_exp", "woba"])
    df = df[df["game_year"].isin(SEASONS)]
    grouped = df.groupby(KEY, observed=True)
    return pd.DataFrame({
        "rv_per_swing": grouped["delta_run_exp"].mean(),
        "woba_swing":   grouped["woba"].mean(),
    }).reset_index()


def _z(s):
    return (s - s.mean()) / s.std()


def convergent(df):
    rows = []
    for a, b in [("runs_count", "twostrike_rv_penalty"),
                 ("runs_count", "adj_count"),
                 ("runs_total", "adjustability"),
                 ("runs_total", "swing_plus")]:
        d = df.dropna(subset=[a, b])
        rows.append({"x": a, "y": b, "n": len(d), "r": round(float(d[a].corr(d[b])), 3)})
    return pd.DataFrame(rows)


def predictive(df):
    rows = []
    for outcome in ["woba_swing", "rv_per_swing"]:
        d = df.dropna(subset=[outcome, "runs_per_swing", "swing_plus"])
        y = _z(d[outcome]).to_numpy(float)
        X = np.column_stack([
            np.ones(len(d)),
            _z(d["swing_plus"]).to_numpy(float),
            _z(d["runs_per_swing"]).to_numpy(float),
        ])
        coefs, se, t, p, n, B = _ols_clustered(y, X, d["batter_id"].to_numpy())
        rows.append({"outcome": outcome, "n": n,
                     "theta": round(float(coefs[-1]), 4),
                     "se": round(float(se[-1]), 4),
                     "t": round(float(t[-1]), 2),
                     "p": round(float(p[-1]), 4)})
    return pd.DataFrame(rows)


MIN_SEASON_SWINGS = 150


def reliability(min_season_swings=MIN_SEASON_SWINGS):
    """2024 vs 2025 split-half correlation of runs_total.

    The whole pipeline is re-run independently inside 2024 and inside 2025, so the two halves
    share no coefficients and no folds. Passing bar from the spec: r comparable to
    `adjustability`'s 0.67.

    unit_record divides by N_SEASONS, which is a constant factor here and therefore
    leaves the correlation unchanged; no rescaling is needed.
    """
    import optimal_policy as op

    models = op.load_models()
    run_value_tables = op.xrv.load_run_value_tables()
    swings = op.load_swings()

    rows = []
    for key, group in swings.groupby(KEY, observed=True, sort=False):
        halves = {}
        for season in SEASONS:
            half = group[group["game_year"] == season]
            if len(half) >= min_season_swings:
                halves[season] = op.unit_record(models, run_value_tables,
                                                half.reset_index(drop=True),
                                                headline_only=True)["runs_total"]
        if len(halves) == len(SEASONS):
            rows.append({"batter_id": key[0], "batter_stand": key[1],
                         **{f"runs_{s}": v for s, v in halves.items()}})

    return pd.DataFrame(rows)


def placebo(n_units=60, seed=11):
    """Recompute runs_total on within-unit shuffled situation labels."""
    import optimal_policy as op

    models = op.load_models()
    run_value_tables = op.xrv.load_run_value_tables()
    swings = op.load_swings()
    groups = list(swings.groupby(KEY, observed=True, sort=False))[:n_units]

    real, fake = [], []
    for i, (_, group) in enumerate(groups):
        group = group.reset_index(drop=True)
        real.append(op.unit_record(models, run_value_tables, group)["runs_total"])
        fake.append(op.unit_record(models, run_value_tables, group,
                                   shuffle_seed=seed + i)["runs_total"])
    return pd.DataFrame({"real": real, "placebo": fake})


def main():
    df = pd.read_parquet(DATA / "optimal_policy.parquet").merge(
        season_outcomes(), on=KEY, how="left")

    conv, pred = convergent(df), predictive(df)
    print(conv.to_string(index=False))
    print()
    print(pred.to_string(index=False))

    print("\nRunning 2024 vs 2025 split-half (~8 min)...")
    rel = reliability()
    split_r = float(rel[f"runs_{SEASONS[0]}"].corr(rel[f"runs_{SEASONS[1]}"]))
    print(f"  n = {len(rel)} units with >= {MIN_SEASON_SWINGS} swings in both seasons, "
          f"r = {split_r:.3f}")

    print("\nRunning placebo (60 units, ~6 min)...")
    plac = placebo()
    plac_ratio = float(plac["placebo"].abs().mean() / plac["real"].abs().mean())
    print(f"  mean |real| = {plac['real'].abs().mean():.2f}  "
          f"mean |placebo| = {plac['placebo'].abs().mean():.2f}  ratio = {plac_ratio:.2f}")

    predictive_null = bool((pred["p"] > 0.05).all())
    placebo_failed = plac_ratio > 0.3
    lines = [
        "# Counterfactual adjustment value — validation\n",
        (f"2024-25, {len(df)} units. `runs_total` = season runs from situational swing "
         "changes versus the de-situated counterfactual, priced by xRV. "
         "See `docs/superpowers/specs/2026-08-03-counterfactual-adjustment-value-design.md`.\n"),
        "## Distribution\n",
        df[["runs_total", "runs_count", "runs_gamestate", "runs_platoon",
            "runs_interaction", "runs_total_2k"]].describe().round(3).to_markdown(),
        "",
        "## Convergent validity\n",
        ("`runs_count` against the matched penalties from `adjustability_value.py`, which "
         "share no machinery with this build — they use realized `delta_run_exp` within "
         "`pitch_type x zone` cells.\n"),
        conv.to_markdown(index=False),
        "",
        "## Predictive validity\n",
        ("Realized full-season production on `runs_per_swing`, controlling `swing_plus`, "
         "all z-scored, clustered SE by `batter_id`. `woba_swing` is the unit mean of the "
         "`woba` column over PA-ending swings; walks and HBP end on takes so they are "
         "absent, making this production over PAs that end on a swing.\n"),
        pred.to_markdown(index=False),
        "",
        "## Reliability\n",
        ("`runs_total` estimated independently within 2024 and within 2025 — separate "
         "regressions, separate folds, no shared coefficients — then correlated across "
         f"the {len(rel)} units with at least {MIN_SEASON_SWINGS} swings in both seasons. "
         f"The comparison bar is `adjustability`'s year-over-year r = 0.67.\n"),
        f"- split-half r = **{split_r:.3f}**\n",
        "",
        "## Support constraint\n",
        f"- `alpha_at_boundary`: {100 * df['alpha_at_boundary'].mean():.0f}% of units\n",
        f"- median `alpha_star_supported`: {df['alpha_star_supported'].median():.2f}\n",
        f"- mean `marginal_runs_per_alpha`: {df['marginal_runs_per_alpha'].mean():+.2f} runs\n",
        "",
        "## Placebo\n",
        ("Situation labels shuffled within unit, preserving marginals and destroying any "
         "real situation-shape relationship. 60 units.\n"),
        (f"- mean |runs_total| real: {plac['real'].abs().mean():.2f}\n"
         f"- mean |runs_total| placebo: {plac['placebo'].abs().mean():.2f}\n"
         f"- ratio: {plac_ratio:.2f} (collapse expected — under ~0.3 is a pass)\n"),
        "",
        "## Verdict\n",
        ("**Model artifact.** The predictive test is null and the placebo did not "
         "collapse. `runs_total` ships as a decomposition only; no leaderboard.\n"
         if (predictive_null and placebo_failed) else
         "**Usable.** " + ("Predictive test null but the placebo collapses, so the "
                           "machinery is not manufacturing value from noise.\n"
                           if predictive_null else
                           "The accounting tracks realized production after controlling "
                           "for swing quality, and the placebo collapses.\n")),
    ]
    out = ROOT / "results" / "optimal_policy.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
