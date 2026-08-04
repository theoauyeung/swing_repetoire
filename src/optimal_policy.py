"""Optimal adjustment policy — counterfactual value of situational swing changes.

For every swing we ask: what would this hitter's swing have looked like on this exact
pitch if the count, base state and matchup had not moved him? Price both the actual and
the counterfactual swing through xRV, difference, and sum over the season.

    value = sum over swings of [ xRV(your swing) - xRV(your de-situated swing) ]

Both arms are FITTED values from the same per-unit regression, so execution noise cancels
and the difference is purely the situation-attributable component of the swing.

Why this replaces the old within-situation contrast
---------------------------------------------------
The previous build measured "2 strikes vs 0-1 strikes" and the analogous base-state and
platoon contrasts. That estimand is defined only on swings inside the contrasted
situations, so it cannot cover a season, and its reference point is the hitter's own
easy-count baseline — which is scaled by how good he is. Judge's early-count baseline is
enormous, so any two-strike degradation read as a large loss. The empirical-Bayes
shrinkage and talent baseline that used to live here existed to patch that; the
counterfactual estimand does not have the problem, so they are gone.

Output: data/optimal_policy.parquet
Run   : python src/optimal_policy.py            (full, ~25 min)
        python src/optimal_policy.py --verify   (scoring correctness check only)
        python src/optimal_policy.py --limit 25 (fast subset for iteration)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import counterfactual as cf                                          # noqa: E402
import xRV_model as xrv                                              # noqa: E402
from adjustability import KEY, MIN_SWINGS, SEASONS                   # noqa: E402
from adjustability import add_context, location_design               # noqa: E402

SHAPE = xrv.SHAPE_FEATURES
LOAD_COLS = KEY + [
    "play_id", "game_year", "batter_full_name", "balls", "strikes", "outs_when_up",
    "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type", "pitcher_throws",
    "on_1b_id", "on_2b_id", "on_3b_id", "delta_run_exp", "woba",
] + SHAPE


def load_models():
    """The three persisted boosters from src/xRV_model.py."""
    specs = [("p_bip", XGBClassifier), ("p_foul", XGBClassifier), ("v_bip", XGBRegressor)]
    models = {}
    for name, cls in specs:
        model = cls(enable_categorical=True)
        model.load_model(DATA / "xrv_models" / f"{name}.json")
        models[name] = model
    return models


def load_swings():
    """2024-25 swings for qualifying units, with pitch_type categories fixed league-wide.

    The cast happens once on the full frame. build_features calls astype("category"),
    which infers categories from whatever it is handed — scoring per-unit frames would
    give each unit its own category codes and silently misalign every prediction.
    """
    df = pd.read_parquet(DATA / "swings_model.parquet", columns=LOAD_COLS)
    df = df[df["game_year"].isin(SEASONS)]
    df["pitch_type"] = df["pitch_type"].astype("category")
    df = add_context(df).dropna(subset=SHAPE)
    sizes = df.groupby(KEY, observed=True)[SHAPE[0]].transform("size")
    return df[sizes >= MIN_SWINGS].reset_index(drop=True)


def score_shapes(models, run_value_tables, group, shape_matrix):
    """Per-swing xRV for `group`'s pitches with its shape columns overwritten.

    assemble_xrv (not the neutral variant) so two-strike stakes stay live: the mechanism
    under test is whether compression buys enough contact to pay for the strikeout risk.
    """
    frame = group.copy()
    for j, feature in enumerate(SHAPE):
        frame[feature] = shape_matrix[:, j]
    enriched = xrv.build_features(frame)
    prob_bip  = models["p_bip"].predict_proba(enriched[xrv.FEATURES])[:, 1]
    prob_foul = models["p_foul"].predict_proba(enriched[xrv.FEATURES])[:, 1]
    value_bip = models["v_bip"].predict(enriched[xrv.FEATURES])
    return xrv.assemble_xrv(enriched, prob_bip, prob_foul, value_bip, run_value_tables)


def verify_scoring(models, run_value_tables, swings, n_units=5):
    """Score OBSERVED shapes and compare to the published xrv column.

    This is the category-alignment guard. If pitch_type codes were misaligned the
    deviation would be large and obvious.
    """
    units = list(dict.fromkeys(zip(swings["batter_id"], swings["batter_stand"])))[:n_units]
    published = pd.read_parquet(DATA / "xrv_swings.parquet", columns=["play_id", "xrv"])
    worst = 0.0
    for batter_id, stand in units:
        group = swings[(swings["batter_id"] == batter_id)
                       & (swings["batter_stand"] == stand)]
        got = score_shapes(models, run_value_tables, group, group[SHAPE].to_numpy(float))
        merged = (pd.DataFrame({"play_id": group["play_id"].to_numpy(), "got": got})
                  .merge(published, on="play_id", how="inner"))
        if merged.empty:
            raise AssertionError(f"no play_id overlap for {batter_id} {stand}")
        worst = max(worst, float((merged["got"] - merged["xrv"]).abs().max()))
    return worst


N_SEASONS = len(SEASONS)
AXIS_NAMES = list(cf.AXES)


def unit_record(models, run_value_tables, group, headline_only=False):
    """One unit's counterfactual run accounting.

    headline_only stops after the two-arm headline, skipping the axis decomposition,
    the alpha scan and the finite difference. Those are 12 of the 14 scoring passes,
    and the split-half reliability check needs only `runs_total`.
    """
    location = location_design(group)
    design, axis_slices = cf.build_design(group, location)
    observed = group[SHAPE].to_numpy(float)

    fits = cf.crossfit_shapes(design, observed)
    shape_actual = cf.predict_oof(design, fits, len(SHAPE))
    design_cf = cf.desituate(design, axis_slices, AXIS_NAMES)
    shape_cf = cf.predict_oof(design_cf, fits, len(SHAPE))

    def season_runs(shape_matrix, baseline):
        return float((score_shapes(models, run_value_tables, group, shape_matrix)
                      - baseline).sum() / N_SEASONS)

    xrv_cf = score_shapes(models, run_value_tables, group, shape_cf)
    xrv_actual = score_shapes(models, run_value_tables, group, shape_actual)

    per_swing_delta = xrv_actual - xrv_cf
    is_two_strike = group["strikes"].to_numpy() == 2

    record = {
        "batter_id":    group["batter_id"].iloc[0],
        "batter_stand": group["batter_stand"].iloc[0],
        "n_swings":     len(group),
        "runs_total":     float(per_swing_delta.sum() / N_SEASONS),
        "runs_total_2k":  float(per_swing_delta[is_two_strike].sum() / N_SEASONS),
        "runs_per_swing": float(per_swing_delta.mean()),
        "xrv_actual_mean": float(xrv_actual.mean()),
        "xrv_cf_mean":     float(xrv_cf.mean()),
    }
    if headline_only:
        return record

    axis_sum = 0.0
    for axis in AXIS_NAMES:
        others = [a for a in AXIS_NAMES if a != axis]
        design_one = cf.desituate(design, axis_slices, others)
        shape_one = cf.predict_oof(design_one, fits, len(SHAPE))
        value = season_runs(shape_one, xrv_cf)
        record[f"runs_{axis}"] = value
        axis_sum += value
    record["runs_interaction"] = record["runs_total"] - axis_sum

    env = cf.envelope(observed)
    admissible = cf.admissible_alphas(shape_cf, shape_actual, env)
    curve = {a: season_runs(cf.blend(shape_cf, shape_actual, a), xrv_cf) for a in admissible}
    best = max(curve, key=curve.get) if curve else float("nan")
    record["alpha_star_supported"] = best
    record["alpha_at_boundary"] = bool(
        admissible and best in (min(admissible), max(admissible))
    )

    lo = season_runs(cf.blend(shape_cf, shape_actual, 0.9), xrv_cf)
    hi = season_runs(cf.blend(shape_cf, shape_actual, 1.1), xrv_cf)
    record["marginal_runs_per_alpha"] = (hi - lo) / 0.2
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    models = load_models()
    run_value_tables = xrv.load_run_value_tables()
    swings = load_swings()
    print(f"{len(swings):,} swings, "
          f"{swings.groupby(KEY, observed=True).ngroups} units")

    if args.verify:
        worst = verify_scoring(models, run_value_tables, swings)
        print(f"max |scored - published xrv| = {worst:.3e}")
        print("PASS" if worst < 1e-5 else "FAIL — pitch_type categories misaligned")
        return

    groups = list(swings.groupby(KEY, observed=True, sort=False))
    if args.limit:
        groups = groups[:args.limit]

    records = []
    for i, (_, group) in enumerate(groups, 1):
        records.append(unit_record(models, run_value_tables, group.reset_index(drop=True)))
        if i % 25 == 0:
            print(f"  {i}/{len(groups)} units")
    df = pd.DataFrame(records)

    adj = pd.read_parquet(DATA / "adjustability.parquet", columns=KEY + [
        "label", "adjustability", "adj_count", "adjustability_plus",
        "adjustability_pctile", "swing_plus",
        "twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty",
    ])
    cards = pd.read_parquet(DATA / "shape_cards.parquet",
                            columns=KEY + ["role", "archetype_name", "grade"])
    primary = (cards[cards["role"] == "primary"][KEY + ["archetype_name", "grade"]]
               .rename(columns={"grade": "primary_grade"}))
    rep = pd.read_parquet(DATA / "repertoire_scores.parquet",
                          columns=KEY + ["repertoire_pctile", "effective_shapes"])

    df = (df.merge(adj, on=KEY, how="left")
            .merge(primary, on=KEY, how="left")
            .merge(rep, on=KEY, how="left"))

    out = DATA / ("optimal_policy_subset.parquet" if args.limit else "optimal_policy.parquet")
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")

    print("\n=== Season runs from situational adjustment ===")
    for col in ["runs_total", "runs_count", "runs_gamestate", "runs_platoon",
                "runs_interaction", "runs_total_2k"]:
        s = df[col]
        print(f"  {col:<18} mean={s.mean():+6.2f}  median={s.median():+6.2f}  "
              f"sd={s.std():5.2f}  range=[{s.min():+6.1f}, {s.max():+6.1f}]")
    resid = df["runs_interaction"].abs()
    print(f"\n  |interaction| mean={resid.mean():.2f}  max={resid.max():.2f}  "
          f"median % of |total|={100 * (resid / df['runs_total'].abs().clip(lower=0.1)).median():.0f}%")
    print(f"  alpha at boundary: {100 * df['alpha_at_boundary'].mean():.0f}% of units")

    print("\n=== Spot check ===")
    spot = df[df["label"].str.contains("Judge|Arraez|Schwarber|Teoscar", na=False)]
    print(spot[["label", "batter_stand", "runs_total", "runs_count", "runs_total_2k",
                "alpha_star_supported", "marginal_runs_per_alpha"]].to_string(index=False))


if __name__ == "__main__":
    main()
