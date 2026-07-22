"""Swing adjustability — how much a hitter's swing tracks the situation, net of pitch location.

Single-regression build. Per (batter, stand) unit (2024-25, >= MIN_SWINGS), for each of 3 volitional
trait dials (bat_speed, swing_length, swing_path_tilt) we fit ONE regression of the dial on

    dial ~ pitch-location surface  +  situation (count, game state, pitch type)

and read adjustability off as the INCREMENTAL variance the situation explains once location is already
in the model — adjusted R^2 of the full model minus the location-only model — averaged over the 3
dials and floored at 0. 0 = the situation adds nothing to a swing already pinned by where the pitch
is; higher = his swing systematically moves with the situation.

Why one joint regression per hitter (not a pooled league model): a pooled model shares one situation
coefficient across everyone, so it reflects a hitter's situation MIX, not his responsiveness. Per-hitter
responsiveness needs hitter x situation terms, which by Frisch-Waugh-Lovell is exactly this per-unit
joint fit. It replaces the earlier two-stage build (global location residualization, then a separate
per-hitter situation R^2): one model is easier to read and fits each hitter's OWN location relationship
rather than a single league-average surface.

The two attack angles are left out of the dials — they are the most location-forced. adjustability is
an UNSIGNED magnitude (a variance share, not a direction).

The headline is the whole situation over a location-only baseline. Each per-axis number is that axis's
UNIQUE contribution — the variance it adds on top of location AND the other two situation axes — so
adj_count holds pitch type fixed and is not two-strike pitch-mix relabeled:

  adjustability  = all of the situation (count + game state + pitch), over location-only  -- headline
  adj_count      = unique add of the count      (balls, strikes),         net of location + game state + pitch
  adj_gamestate  = unique add of the game state (base state, outs),         net of location + count + pitch
  adj_pitch      = unique add of the pitch type (FB / breaking / offspeed), net of location + count + game state

Input:  data/swings_model.parquet
Output: data/adjustability.parquet   (headline column: adjustability)

Run:  python src/adjustability.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KEY = ["batter_id", "batter_stand"]
SEASONS = [2024, 2025]
MIN_SWINGS = 400
DIALS = ["bat_speed", "swing_length", "swing_path_tilt"]
PITCH_GROUP = {"FF": "FB", "SI": "FB", "FC": "FB", "SL": "brk", "CU": "brk", "KC": "brk", "ST": "brk",
               "SV": "brk", "SC": "brk", "KN": "brk", "CH": "off", "FS": "off", "FO": "off"}


def add_context(df):
    pull = np.where(df["batter_stand"].to_numpy() == "L", 1.0, -1.0)
    df["px"] = df["plate_x"] * pull
    df["pz"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    df = df.dropna(subset=["px", "pz"] + DIALS).copy()
    df["pitch_group"] = df["pitch_type"].map(PITCH_GROUP).fillna("other")
    on1, on2, on3 = df["on_1b_id"].notna(), df["on_2b_id"].notna(), df["on_3b_id"].notna()
    df["base_state"] = np.where(on2 | on3, "risp", np.where(on1, "on1", "empty"))
    return df


def location_design(g):
    """Pitch-location nuisance surface: intercept + linear + quadratic + interaction in (px, pz)."""
    px, pz = g.px.to_numpy(float), g.pz.to_numpy(float)
    return np.column_stack([np.ones(len(g)), px, pz, px ** 2, pz ** 2, px * pz])


def adj_r2(y, X):
    """Adjusted R^2 of y on design X (X INCLUDES its own intercept column). Not clipped here — the
    caller differences two adj_r2's and floors the increment."""
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = float(((y - X @ b) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    p = k - 1                                     # predictors excluding the intercept
    if sst <= 0 or n - p - 1 <= 0:
        return 0.0
    return 1 - (sse / sst) * (n - 1) / (n - p - 1)


def dummies(g, cols):
    return pd.get_dummies(g[cols].astype(str), drop_first=True).to_numpy(float)


def main():
    df = add_context(pd.read_parquet(
        DATA / "swings_model.parquet",
        columns=["game_year", "batter_full_name"] + KEY + ["balls", "strikes", "outs_when_up",
                 "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type",
                 "on_1b_id", "on_2b_id", "on_3b_id"] + DIALS).query("game_year in @SEASONS"))
    axes = {"count": ["balls", "strikes"], "gamestate": ["base_state", "outs_when_up"],
            "pitch": ["pitch_group"]}
    all_cols = [c for cols in axes.values() for c in cols]

    rows = []
    for (bid, stand), g in df.groupby(KEY, sort=False):
        if len(g) < MIN_SWINGS:
            continue
        L = location_design(g)
        ys = {d: g[d].to_numpy(float) for d in DIALS}

        def adjr2_with(cols):
            """Per-dial adjusted R^2 of [location + these situation cols] (location-only if cols empty)."""
            X = np.column_stack([L, dummies(g, cols)]) if cols else L
            return {d: adj_r2(ys[d], X) for d in DIALS}

        a_full = adjr2_with(all_cols)

        def mean_gain(baseline):
            """Mean over dials of (full model - baseline model) adjusted R^2, each floored at 0."""
            return float(np.mean([max(0.0, a_full[d] - baseline[d]) for d in DIALS]))

        # headline = whole situation over location-only; per-axis = unique add net of location + the OTHER axes
        row = {"batter_id": bid, "batter_stand": stand, "label": g["batter_full_name"].iloc[0],
               "n_swings": len(g), "adjustability": round(mean_gain(adjr2_with([])), 4)}
        for ax, cols in axes.items():
            others = [c for a, cc in axes.items() if a != ax for c in cc]
            row[f"adj_{ax}"] = round(mean_gain(adjr2_with(others)), 4)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("adjustability", ascending=False).reset_index(drop=True)
    out["adjustability_pctile"] = (out["adjustability"].rank(pct=True) * 100).round(1)
    out.to_parquet(DATA / "adjustability.parquet", index=False)
    print(f"{len(out)} hitters, >= {MIN_SWINGS} swings, {SEASONS}")
    print(f"adjustability: mean {out.adjustability.mean():.3f}, median {out.adjustability.median():.3f}, "
          f"max {out.adjustability.max():.3f}")
    print(f"per-axis means -> count {out.adj_count.mean():.3f}  gamestate {out.adj_gamestate.mean():.3f}  "
          f"pitch {out.adj_pitch.mean():.3f}")
    print("\ntop 8:\n", out.head(8)[["label", "batter_stand", "n_swings", "adjustability",
                                      "adj_count", "adj_gamestate", "adj_pitch"]].to_string(index=False))
    print("\nbottom 5:\n", out.tail(5)[["label", "batter_stand", "adjustability",
                                        "adj_count", "adj_gamestate", "adj_pitch"]].to_string(index=False))


if __name__ == "__main__":
    main()
