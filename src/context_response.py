"""Swing adjustability — how much a hitter changes his swing to fit the situation.

Per (batter, stand) unit (2024-25, >= MIN_SWINGS): of the variation in a hitter's swing that is NOT
explained by pitch location, how much is explained by the SITUATION — the count, the game state, and
the pitch type? High = he systematically reshapes his swing by situation (adjustable); ~0 = same
swing regardless / only random variation.

Swing = 3 dials: bat_speed, swing_length, swing_path_tilt (the trait dials; the two attack angles are
left out — they are the most location-forced).

Location guard: BEFORE measuring adjustment, each dial is residualized on continuous pitch location
(league-wide: dial ~ plate_x, plate_z, squares, interaction). We then measure adjustment on the
LEFTOVER, so "he swung differently because the pitch was in a different spot" is removed up front and
never counts as adjustment.

Adjustment is measured as adjusted R^2 (share of dial variance the situation explains, penalized for
predictor count), averaged over the 3 dials, clipped at 0:
  adjustability  = situation overall (count + game state + pitch type)
  adj_count      = count only        (balls, strikes)          -- clean (count does not force geometry)
  adj_gamestate  = game state only   (base state, outs)         -- clean
  adj_pitch      = pitch type only   (FB / breaking / offspeed) -- partly forced by pitch movement; read with care

Input:  data/swings_model.parquet
Output: data/context_response.parquet

Run:  python src/context_response.py
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


def residualize_on_location(df):
    """Strip the part of each dial that pitch location predicts (league-wide), so location can't
    masquerade as adjustment. Leaves dial+'_r' columns."""
    X = np.column_stack([np.ones(len(df)), df.px, df.pz, df.px ** 2, df.pz ** 2, df.px * df.pz])
    for d in DIALS:
        b, *_ = np.linalg.lstsq(X, df[d].to_numpy(float), rcond=None)
        df[d + "_r"] = df[d].to_numpy(float) - X @ b
    return df


def adj_r2(y, D):
    """Adjusted R^2 of y on dummy design D (no intercept col); how much of y's variance the
    situation explains, penalized for the number of predictors. Clipped at 0."""
    n = len(y)
    X = np.column_stack([np.ones(n), D]) if D.shape[1] else np.ones((n, 1))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = float(((y - X @ b) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    p = X.shape[1] - 1
    if sst <= 0 or n - p - 1 <= 0:
        return 0.0
    return max(0.0, 1 - (1 - (1 - sse / sst)) * (n - 1) / (n - p - 1))


def dummies(g, cols):
    return pd.get_dummies(g[cols].astype(str), drop_first=True).to_numpy(float)


def main():
    df = add_context(pd.read_parquet(
        DATA / "swings_model.parquet",
        columns=["game_year", "batter_full_name"] + KEY + ["balls", "strikes", "outs_when_up",
                 "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type",
                 "on_1b_id", "on_2b_id", "on_3b_id"] + DIALS).query("game_year in @SEASONS"))
    df = residualize_on_location(df)
    axes = {"count": ["balls", "strikes"], "gamestate": ["base_state", "outs_when_up"],
            "pitch": ["pitch_group"]}
    all_cols = [c for cols in axes.values() for c in cols]
    rd = [d + "_r" for d in DIALS]

    rows = []
    for (bid, stand), g in df.groupby(KEY, sort=False):
        if len(g) < MIN_SWINGS:
            continue
        overall = np.mean([adj_r2(g[d].to_numpy(), dummies(g, all_cols)) for d in rd])
        per = {f"adj_{ax}": np.mean([adj_r2(g[d].to_numpy(), dummies(g, cols)) for d in rd])
               for ax, cols in axes.items()}
        rows.append({"batter_id": bid, "batter_stand": stand, "label": g["batter_full_name"].iloc[0],
                     "n_swings": len(g), "adjustability": round(overall, 4),
                     **{k: round(v, 4) for k, v in per.items()}})

    out = pd.DataFrame(rows).sort_values("adjustability", ascending=False).reset_index(drop=True)
    out["adjustability_pctile"] = (out["adjustability"].rank(pct=True) * 100).round(1)
    out.to_parquet(DATA / "context_response.parquet", index=False)
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
