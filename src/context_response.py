"""Context-responsiveness (adjustability) v2 — cluster-free, per-dial slopes.

The paper's adjustability metric: does a hitter change HOW they swing with the strategic context,
holding fixed the context that mechanically dictates geometry? v1 (mutual information over shape
CLUSTERS) read ~0 count-response, but the diagnostic (experiments/adjustability_count_diagnostic.py)
showed the signal is real but HIDDEN — it lives in bat_speed / swing_length, which the 5-feature
clustering under-weights vs the location-forced attack angles. So v2 drops clusters entirely and
measures the count effect DIRECTLY on the volitional dials, conditional on pitch location + type.

Method (per (batter, stand) unit, 2024-25, >= MIN_SWINGS swings):
  For each volitional dial (bat_speed, swing_length, swing_path_tilt), the within-location×pitch-type
  fixed-effects slope of the dial on `strikes` (0/1/2) — i.e. how much the dial moves per added strike
  once we've held the pitch's location and type fixed. Location is the FORCED channel and can't be
  cleaned; count does not change the geometry needed to meet a pitch at location X, so a
  within-location count slope is volitional (see docs/adjustability-decontamination.md). Slopes are
  reported in cohort-SD units (`_d`, per strike) with a t-stat (`_z`). The attack angles are excluded
  (they are pitch-forced; that's the whole point). Base-out (RISP) is a secondary axis, estimated the
  same way but additionally holding count fixed, and reported apart (it is ~5x weaker than count).

  All dials move the same way under pressure (slower/shorter/flatter), so the composite
  `count_adj = mean(-_d)` is signed so **higher = compresses more under two-strike pressure**.

Input:  data/swings_model.parquet, src/repertoire_reference.json (cohort SDs)
Output: data/context_response.parquet, data/context_response_catalog.md

Run:  python src/context_response.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KEY = ["batter_id", "batter_stand"]
SEASONS = [2024, 2025]
MIN_SWINGS = 400        # per unit; conditioning on location cells needs a bit more than v1's 300
DIALS = ["bat_speed", "swing_length", "swing_path_tilt"]   # volitional trait dials (forced angles excluded)
SHORT = {"bat_speed": "batspeed", "swing_length": "length", "swing_path_tilt": "tilt"}
_sd = json.loads((ROOT / "src" / "repertoire_reference.json").read_text())["feature_sd"]
SD = {d: _sd[d] for d in DIALS}


def add_context(df):
    pull = np.where(df["batter_stand"].to_numpy() == "L", 1.0, -1.0)
    df["px"] = df["plate_x"] * pull
    df["pz"] = (df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"])
    df = df.dropna(subset=["px", "pz"] + DIALS).copy()
    lh = np.select([df.px > 0.3, df.px < -0.3], ["in", "out"], default="mid")
    lv = np.select([df.pz > 0.66, df.pz < 0.33], ["hi", "lo"], default="mid")
    grp = df["pitch_type"].map({"FF": "FB", "SI": "FB", "FC": "FB", "SL": "brk", "CU": "brk",
                                "KC": "brk", "ST": "brk", "SV": "brk", "SC": "brk", "KN": "brk",
                                "CH": "off", "FS": "off", "FO": "off"}).fillna("other")
    df["loc_pitch"] = pd.Series(lh, index=df.index) + "_" + pd.Series(lv, index=df.index) + "|" + grp
    df["risp"] = ((df["on_2b_id"].notna()) | (df["on_3b_id"].notna())).astype(float)
    return df


def fe_slope(g, feat, treat, cells):
    """Fixed-effects slope of `feat` on `treat` within `cells` (demean both within cell), + t-stat.
    Returns (slope, se). slope is the within-cell OLS coefficient; SE from the within residuals."""
    grp = g.groupby(cells)
    rt = g[treat].to_numpy() - grp[treat].transform("mean").to_numpy()
    rf = g[feat].to_numpy() - grp[feat].transform("mean").to_numpy()
    Sxx = float((rt * rt).sum())
    ncells = grp.ngroups
    dof = len(g) - ncells - 1
    if Sxx <= 0 or dof <= 0:
        return np.nan, np.nan
    slope = float((rf * rt).sum() / Sxx)
    sse = float(((rf - slope * rt) ** 2).sum())
    se = float(np.sqrt(sse / dof / Sxx))
    return slope, se


def main():
    df = add_context(pd.read_parquet(
        DATA / "swings_model.parquet",
        columns=["game_year", "batter_full_name"] + KEY + ["strikes", "plate_x", "plate_z",
                 "sz_top", "sz_bot", "pitch_type", "on_2b_id", "on_3b_id"] + DIALS)
        .query("game_year in @SEASONS"))

    rows = []
    for (bid, stand), g in df.groupby(KEY, sort=False):
        if len(g) < MIN_SWINGS:
            continue
        row = {"batter_id": bid, "batter_stand": stand,
               "label": g["batter_full_name"].iloc[0], "n_swings": len(g)}
        d_std = []
        for dial in DIALS:                                    # count adjustment | location, pitch
            sl, se = fe_slope(g, dial, "strikes", "loc_pitch")
            row[f"cnt_{SHORT[dial]}_d"] = round(sl / SD[dial], 3)
            row[f"cnt_{SHORT[dial]}_z"] = round(sl / se, 2) if se and se > 0 else np.nan
            d_std.append(sl / SD[dial])
        row["count_adj"] = round(-float(np.mean(d_std)), 3)   # signed: higher = compresses under strikes
        for dial in DIALS:                                    # base-out | location, pitch, count (secondary)
            sl, se = fe_slope(g, dial, "risp", ["loc_pitch", "strikes"])
            row[f"base_{SHORT[dial]}_d"] = round(sl / SD[dial], 3) if not np.isnan(sl) else np.nan
        base_d = [row[f"base_{SHORT[dial]}_d"] for dial in DIALS]
        row["base_adj"] = round(-float(np.nanmean(base_d)), 3) if np.isfinite(base_d).any() else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out["count_adj_pctile"] = (out["count_adj"].rank(pct=True) * 100).round(1)
    out = out.sort_values("count_adj", ascending=False).reset_index(drop=True)
    out.to_parquet(DATA / "context_response.parquet", index=False)
    write_catalog(out)
    print(f"Wrote context_response.parquet ({len(out)} units, >= {MIN_SWINGS} swings, {SEASONS})")


def write_catalog(a):
    show = ["label", "batter_stand", "n_swings", "count_adj", "count_adj_pctile",
            "cnt_batspeed_d", "cnt_length_d", "cnt_tilt_d", "base_adj"]
    L = ["# Context-responsiveness (adjustability) v2 — cluster-free per-dial slopes\n",
         "How much a hitter changes HOW they swing with the count, holding pitch location + type "
         "fixed. Measured directly on the volitional dials (bat_speed, swing_length, swing_path_tilt) "
         "— NOT on shape clusters, which are dominated by the location-forced attack angles and hid "
         "this signal in v1 (see docs/adjustability-decontamination.md). `cnt_*_d` = within-location "
         "fixed-effects slope of that dial on `strikes`, in cohort-SD units per added strike (negative "
         "= slower/shorter/flatter under pressure); `cnt_*_z` = t-stat. `count_adj` = mean(-_d), "
         f"signed so **higher = compresses more under two-strike pressure**. Units: 2024-25, "
         f">= {MIN_SWINGS} swings.\n",
         "**Guardrail:** count is the clean instrument *because* it doesn't dictate geometry; location "
         "is held fixed but its own forced/chosen split is not identifiable, so we don't headline it. "
         "`base_adj` (runners on 2nd/3rd, holding location+pitch+count) is exploratory — ~5x weaker.\n",
         f"- Units scored: **{len(a)}**  ·  count_adj mean {a.count_adj.mean():.3f}, "
         f"median {a.count_adj.median():.3f}, max {a.count_adj.max():.3f}\n",
         "## Most adjustable (compress most under two-strike pressure)",
         a.head(20)[show].to_markdown(index=False) + "\n",
         "## Least adjustable (swing the same regardless of count)",
         a.tail(20).sort_values("count_adj")[show].to_markdown(index=False) + "\n"]
    (DATA / "context_response_catalog.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:5]))


if __name__ == "__main__":
    main()
