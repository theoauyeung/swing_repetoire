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


if __name__ == "__main__":
    main()
