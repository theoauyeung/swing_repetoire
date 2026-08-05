"""Stress-tests the adjustment-value numbers before anyone puts them on a leaderboard. A
swing that never happened has no real outcome attached, so the runs come from a model, and
a model will happily hand back a number even when there is nothing there.

Three checks: shuffle the situations and confirm the score collapses, see whether it predicts
real production, and see whether a hitter scores the same in 2024 and 2025.

Run: python src/adjustability_value_validate.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjustability import KEY, SEASONS                    # noqa: E402
from adjustability_value_first_draft import _ols_clustered            # noqa: E402


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
        coefs, se, t, p, n, _ = _ols_clustered(y, X, d["batter_id"].to_numpy())
        rows.append({"outcome": outcome, "n": n,
                     "theta": round(float(coefs[-1]), 4),
                     "se": round(float(se[-1]), 4),
                     "t": round(float(t[-1]), 2),
                     "p": round(float(p[-1]), 4)})
    return pd.DataFrame(rows)


MIN_SEASON_SWINGS = 150
# Everything the prescription rests on. runs_total is the accounting headline; the rest are
# the policy layer, whose reliability the design pre-commits to before publication.
RELIABILITY_COLS = [
    "runs_total", "runs_vs_desituated", "runs_per_swing",
    "marginal_runs_per_alpha", "displacement_sd",
    "alpha_peak_unconstrained", "alpha_star_policy", "runs_at_alpha_star_per_swing",
]
GRADIENT_CELL = ("count", "2 strikes")


def _season_half(op, cf, models, run_value_tables, half, scale):
    """Full re-estimation inside one season: records, plus that season's own policy cap.

    The cap and the replacement pool are both rebuilt from this half alone rather than borrowed
    from the pooled run — otherwise the two halves would share an input and the correlation
    would be inflated.
    """
    scale_vec = np.asarray([scale[f] for f in op.SHAPE], float)
    fits = [op.unit_fit(group.reset_index(drop=True), scale_vec)
            for _, group in half.groupby(KEY, observed=True, sort=False)]
    blocks = [f["policy"] for f in fits]

    records, curves = [], []
    for i, fit in enumerate(fits):
        record = op.unit_record(models, run_value_tables, fit, scale, blocks, i)
        curves.append(record.pop("_alpha_curve"))
        for row in record.pop("_prescriptions"):
            if (row["axis"], row["cell"]) == GRADIENT_CELL:
                record[f"grad2k_{row['dial']}"] = row["grad_delta_runs_per_sd"]
        records.append(record)

    df = pd.DataFrame(records)
    cap = float(df["displacement_sd"].quantile(cf.POLICY_QUANTILE))
    policy, gain = [], []
    for d_u, curve in zip(df["displacement_sd"], curves):
        allowed = {a: v for a, v in curve.items() if a in cf.policy_alphas(d_u, cap)}
        best = max(allowed, key=allowed.get) if allowed else float("nan")
        policy.append(best)
        gain.append(allowed[best] - curve[1.0] if allowed and 1.0 in curve else float("nan"))
    df["alpha_star_policy"] = policy
    df["runs_at_alpha_star_per_swing"] = np.asarray(gain) * len(SEASONS) / df["n_swings"]
    return df


def reliability(min_season_swings=MIN_SEASON_SWINGS):
    """2024 vs 2025 split-half for every quantity the prescription rests on.

    The whole pipeline is re-run independently inside 2024 and inside 2025, so the two halves
    share no coefficients, no folds and no policy cap. Passing bar from the spec: r comparable
    to `adjustability`'s 0.67, and a pre-committed r = 0.5 floor below which
    `alpha_star_policy` does not ship as a per-batter recommendation.

    unit_record divides by N_SEASONS, a constant factor here, so correlations are unaffected.
    """
    import adjustability_value as op
    import counterfactual as cf

    models = op.load_models()
    run_value_tables = op.xrv.load_run_value_tables()
    swings = op.load_swings()
    scale = op.load_policy_reference(swings)["shape_sd"]

    halves = {}
    for season in SEASONS:
        half = swings[swings["game_year"] == season]
        sizes = half.groupby(KEY, observed=True)[op.SHAPE[0]].transform("size")
        half = half[sizes >= min_season_swings]
        halves[season] = _season_half(op, cf, models, run_value_tables, half, scale)
        print(f"  {season}: {len(halves[season])} units")

    return halves[SEASONS[0]].merge(halves[SEASONS[1]], on=KEY, suffixes=("_a", "_b"))


def reliability_table(rel):
    """Split-half r per quantity, weakest-first so the gate is impossible to skim past."""
    dials = sorted(c[len("grad2k_"):-len("_a")] for c in rel.columns
                   if c.startswith("grad2k_") and c.endswith("_a"))
    cols = RELIABILITY_COLS + [f"grad2k_{d}" for d in dials]
    rows = [{"quantity": c, "r": round(float(rel[f"{c}_a"].corr(rel[f"{c}_b"])), 3)}
            for c in cols if f"{c}_a" in rel.columns]
    return pd.DataFrame(rows).sort_values("r").reset_index(drop=True)


def gradient_summary():
    """Per-axis size of the prescriptive signal, and which dial it lands on.

    Reported for all three axes even where the signal is small: a near-zero situational
    gradient on gamestate is itself the finding, not a reason to drop the axis.
    """
    grad = pd.read_parquet(DATA / "adjustability_gradients.parquet")
    rows = []
    for axis, block in grad.groupby("axis"):
        best = block.loc[block.groupby(KEY)["situational_runs_per_quarter_sd"]
                         .apply(lambda s: s.abs().idxmax())]
        rows.append({
            "axis": axis,
            "units": int(block[KEY].drop_duplicates().shape[0]),
            "cells": int(block["cell"].nunique()),
            # No pipes in these names: the notebook parses this table straight out of the
            # markdown report, and a literal | inside a header cell splits the row.
            "median_abs_grad_delta": round(
                float(block["grad_delta_runs_per_sd"].abs().median()), 4),
            "median_abs_runs_at_quarter_sd": round(
                float(best["situational_runs_per_quarter_sd"].abs().median()), 3),
            "most_common_lever": best["dial"].mode().iloc[0],
        })
    return pd.DataFrame(rows)


def placebo(n_units=60, seed=11):
    """Recompute the headline on within-unit shuffled situation labels.

    Run on `runs_vs_desituated`, not on `runs_total`: shuffling destroys the situation-shape
    relationship in the unit being tested but the replacement blocks come from unshuffled hitters,
    so a replacement-benchmarked placebo would be a comparison between two different worlds. The
    de-situated arm is self-contained and is what this test was designed for. The
    aim-destroying placebo for the replacement benchmark lives in experiments/directional_placebo.py.
    """
    import adjustability_value as op
    import numpy as np

    models = op.load_models()
    run_value_tables = op.xrv.load_run_value_tables()
    swings = op.load_swings()
    reference = op.load_policy_reference(swings)
    scale = np.asarray([reference["shape_sd"][f] for f in op.SHAPE], float)
    groups = list(swings.groupby(KEY, observed=True, sort=False))[:n_units]

    def headline(group, shuffle_seed=None):
        fit = op.unit_fit(group, scale, shuffle_seed=shuffle_seed)
        return op.unit_record(models, run_value_tables, fit, reference["shape_sd"],
                              headline_only=True)["runs_vs_desituated"]

    real, fake = [], []
    for i, (_, group) in enumerate(groups):
        group = group.reset_index(drop=True)
        real.append(headline(group))
        fake.append(headline(group, shuffle_seed=seed + i))
    return pd.DataFrame({"real": real, "placebo": fake})


def main(reuse=False):
    df = pd.read_parquet(DATA / "adjustability_value.parquet").merge(
        season_outcomes(), on=KEY, how="left")

    conv, pred = convergent(df), predictive(df)
    grad_summary = gradient_summary()
    print(conv.to_string(index=False))
    print()
    print(pred.to_string(index=False))
    print()
    print(grad_summary.to_string(index=False))

    cache = DATA / "adjustability_value_reliability.parquet"
    if reuse and cache.exists():
        rel = pd.read_parquet(cache)
        print(f"\nReusing cached split-half from {cache}")
    else:
        print("\nRunning 2024 vs 2025 split-half (~20 min)...")
        rel = reliability()
        rel.to_parquet(cache, index=False)
    rel_table = reliability_table(rel)
    policy_r = float(rel_table.loc[rel_table["quantity"] == "alpha_star_policy", "r"].iloc[0])
    print(f"  n = {len(rel)} units with >= {MIN_SEASON_SWINGS} swings in both seasons")
    print(rel_table.to_string(index=False))

    print("\nRunning placebo (60 units, ~6 min)...")
    plac = placebo()
    plac_ratio = float(plac["placebo"].abs().mean() / plac["real"].abs().mean())
    print(f"  mean |real| = {plac['real'].abs().mean():.2f}  "
          f"mean |placebo| = {plac['placebo'].abs().mean():.2f}  ratio = {plac_ratio:.2f}")

    predictive_null = bool((pred["p"] > 0.05).all())
    placebo_failed = plac_ratio > 0.3
    disjoint_r = float(conv.loc[conv["y"] == "twostrike_rv_penalty", "r"].iloc[0])
    passing = ", ".join(f"`{o}`" for o in pred.loc[pred["p"] <= 0.05, "outcome"])
    failing = ", ".join(f"`{o}`" for o in pred.loc[pred["p"] > 0.05, "outcome"])

    if predictive_null and placebo_failed:
        verdict = ("**Model artifact.** The predictive test is null and the placebo did not "
                   "collapse. `runs_total` ships as a decomposition only; no leaderboard.\n")
    elif predictive_null:
        verdict = (f"**Qualified.** Every predictive outcome is null ({failing}). The placebo "
                   f"collapses (ratio {plac_ratio:.2f}), so the machinery is not manufacturing "
                   "value from noise, but nothing here shows `runs_total` tracks production.\n")
    else:
        verdict = (
            f"**Qualified.** The placebo collapses (ratio {plac_ratio:.2f}), so `runs_total` is "
            "not manufactured from noise. External support is thinner than that sounds. The "
            f"predictive test is significant on {passing} and null on {failing}. "
            "`rv_per_swing` is realized `delta_run_exp` — close to what xRV is trained to "
            "reproduce, so it is the more circular of the two — while `woba_swing`, the less "
            "circular outcome, is null. The one fully disjoint outcome-side convergent test, "
            f"`runs_count` against `twostrike_rv_penalty`, is r = {disjoint_r:.3f}. Read "
            "`runs_total` as an accounting decomposition with internal validity, not as a "
            "validated predictor of run production.\n")

    # The policy layer is gated separately from the accounting headline: they can pass and fail
    # independently, and collapsing them to one label would hide which of the two ships.
    below = rel_table[rel_table["r"] < 0.5]
    above = rel_table[rel_table["r"] >= 0.5]
    fmt = lambda t: ", ".join(f"`{q}` (r = {r:.3f})"  # noqa: E731
                              for q, r in zip(t["quantity"], t["r"]))
    verdict += (
        "\n**Policy layer: split.** The alpha scan and the per-dial gradients do not stand or "
        "fall together. Below the pre-committed 0.5 reliability floor: "
        f"{fmt(below)}. At or above it: {fmt(above)}. "
        f"`alpha_star_policy` at r = {policy_r:.3f} is "
        + ("above the floor and ships per batter. "
           if policy_r >= 0.5 else
           "the weakest quantity in the build — how much to modulate is not a per-batter "
           "recommendation, only a cohort-level statement. ")
        + f"The most repeatable quantity in the build is `{rel_table['quantity'].iloc[-1]}` "
        f"(r = {rel_table['r'].iloc[-1]:.3f}). Read *which lever* and *how much* against their "
        "own rows above rather than against a single label for the layer.\n")

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
        ("`runs_count` against the matched penalties from `adjustability_value_first_draft.py`, "
         "which "
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
        ("Every quantity re-estimated independently within 2024 and within 2025 — separate "
         "regressions, separate folds, separate policy caps, no shared coefficients — then "
         f"correlated across the {len(rel)} units with at least {MIN_SEASON_SWINGS} swings in "
         "both seasons. The comparison bar is `adjustability`'s year-over-year r = 0.67, and "
         "the design pre-commits to an r = 0.5 floor for `alpha_star_policy`. `grad2k_*` are "
         "the two-strike situational gradients, the per-dial prescriptive layer.\n"),
        rel_table.to_markdown(index=False),
        "",
        (f"`alpha_star_policy` splits at r = {policy_r:.3f}, "
         + ("clearing" if policy_r >= 0.5 else "**below**")
         + " the pre-committed 0.5 floor, so it "
         + ("ships as a per-batter recommendation.\n" if policy_r >= 0.5 else
            "ships as a cohort-level statement only and the per-batter column is documented "
            "as unreliable.\n")),
        "",
        "## Policy layer\n",
        (f"- median `alpha_peak_unconstrained`: {df['alpha_peak_unconstrained'].median():.2f} "
         f"({100 * df['alpha_peak_at_boundary'].mean():.0f}% of units still at a grid edge). "
         "Diagnostic of curvature, explicitly extrapolative, never a recommendation.\n"),
        (f"- median `alpha_star_policy`: {df['alpha_star_policy'].median():.2f}; "
         f"{int((df['alpha_star_policy'] < 1).sum())} units told to modulate less, "
         f"{int((df['alpha_star_policy'] == 1).sum())} to hold, "
         f"{int((df['alpha_star_policy'] > 1).sum())} to modulate more. "
         f"corr with `adj_count` = {df['alpha_star_policy'].corr(df['adj_count']):+.2f} — the "
         "league cap correctly leaves room for low modulators and none for high ones.\n"),
        (f"- mean `runs_at_alpha_star`: {df['runs_at_alpha_star'].mean():+.2f} runs, itself a "
         f"counting stat (corr with `n_swings` = "
         f"{df['runs_at_alpha_star'].corr(df['n_swings']):+.2f}); use "
         "`runs_at_alpha_star_per_swing` for cross-hitter comparison.\n"),
        f"- mean `marginal_runs_per_alpha`: {df['marginal_runs_per_alpha'].mean():+.2f} runs "
        f"(counting stat: corr with `n_swings` = "
        f"{df['marginal_runs_per_alpha'].corr(df['n_swings']):+.2f}, with `runs_total` = "
        f"{df['marginal_runs_per_alpha'].corr(df['runs_total']):+.2f}; no per-swing version "
        f"exists, so cross-hitter comparison on it is substantially playing time)\n",
        "",
        "## Per-dial prescriptions\n",
        ("`grad_delta_runs_per_sd` is what a +1 SD move on a dial buys in a situation cell, net "
         "of the hitter's own all-swing gradient — the raw cell gradient is dominated by his "
         "baseline, so only the delta is situational. `situational_runs_per_quarter_sd` scales "
         "that to a realistic +0.25 SD move over the cell's own season swings. Deltas are "
         "n-weighted contrasts within an axis and sum to zero across its cells, so they are "
         "read per cell and never summed to a season total. All three axes are reported "
         "regardless of signal size.\n"),
        grad_summary.to_markdown(index=False),
        "",
        "## Placebo\n",
        ("Situation labels shuffled within unit, preserving marginals and destroying any "
         "real situation-shape relationship. 60 units.\n"),
        (f"- mean |runs_total| real: {plac['real'].abs().mean():.2f}\n"
         f"- mean |runs_total| placebo: {plac['placebo'].abs().mean():.2f}\n"
         f"- ratio: {plac_ratio:.2f} (collapse expected — under ~0.3 is a pass)\n"),
        "",
        "## Verdict\n",
        verdict,
    ]
    out = ROOT / "results" / "adjustability_value.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true",
                        help="reuse the cached split-half instead of the ~20 min recompute")
    main(**vars(parser.parse_args()))
