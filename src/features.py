"""M2 prep: filter to competitive tracked swings and build the model-ready swing table.

Competitive swing (operational definition, no DB flag exists):
  - bat-tracked (5 shape features present)
  - not a bunt (is_bunt == 0)
  - bat_speed >= 50 mph  (removes checked/emergency/defensive swings)

Also builds `horz_attack_angle_pull` (+ = pull side, both hands). horz_attack_angle is already a
BATTER-RELATIVE metric — raw + = toward the OPPOSITE field for both L and R (verified against
bearing_angle: corr(raw horz, pull) = -0.47 RHH / -0.45 LHH, i.e. pull = negative raw for both),
so the pull frame is a uniform negation, NOT a per-hand mirror. (The old `*(L?-1:1)` mirror left
RHH inverted — RHH "pull" was actually oppo — and only LHH came out right; see worklog 2026-07-09.)
Note plate_x, by contrast, IS absolute (catcher frame) and needs a real per-hand flip — that lives
in xRV_model.build_features / cards.py, not here.

Input:  data/swings_2024_2026_mlb.parquet
Output: data/swings_model.parquet  (competitive tracked swings, features + context + value)

Run:  python src/features.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SHAPE = ["swing_path_tilt", "swing_length", "bat_speed", "vert_attack_angle", "horz_attack_angle"]
BAT_SPEED_MIN = 50.0
HORZ_ABS_MAX = 45.0        # |horz_attack_angle| beyond this = failed measurement
VERT_BOUNDS = (-45.0, 75.0)  # vert_attack_angle physical bounds for a competitive swing

KEEP = [
    "play_id", "game_pk", "game_date", "game_year",
    "batter_id", "batter_full_name", "batter_stand", "pitcher_id", "pitcher_throws",
    "balls", "strikes", "outs_when_up", "plate_x", "plate_z", "plate_zone", "pitch_type",
    "on_1b_id", "on_2b_id", "on_3b_id",
    "delta_run_exp", "woba", "xwoba", "exit_velo", "launch_angle", "bearing_angle",
    "pitch_outcome", "pa_outcome", "is_whiff", "is_contact", "is_bip",
    "ball_bat_intercept_y",  # timing descriptor (not a shape feature)
    "sz_top", "sz_bot", "height",
] + SHAPE


def main():
    df = pd.read_parquet(DATA / "swings_2024_2026_mlb.parquet")
    n0 = len(df)

    # filter funnel
    tracked = df["bat_speed"].notna() & df[SHAPE].notna().all(axis=1)
    d1 = df[tracked]
    d2 = d1[d1["is_bunt"] == 0]
    d3 = d2[d2["bat_speed"] >= BAT_SPEED_MIN]
    ok_angle = (d3["horz_attack_angle"].abs() <= HORZ_ABS_MAX) & \
               d3["vert_attack_angle"].between(*VERT_BOUNDS)
    d4 = d3[ok_angle].copy()

    print("Filter funnel:")
    print(f"  all swings                 : {n0:>10,}")
    print(f"  bat-tracked (5 feats)      : {len(d1):>10,}  ({len(d1)/n0*100:.1f}%)")
    print(f"  - bunts removed            : {len(d2):>10,}  (-{len(d1)-len(d2):,})")
    print(f"  - bat_speed >= {BAT_SPEED_MIN:g}         : {len(d3):>10,}  (-{len(d2)-len(d3):,})")
    print(f"  - angle artifacts dropped  : {len(d4):>10,}  (-{len(d3)-len(d4):,})")

    # pull frame (+ = pull side, both hands). horz_attack_angle is batter-relative with raw + = oppo,
    # so pull is a uniform negation (validated vs bearing_angle) — NOT a per-hand mirror.
    d4["horz_attack_angle_pull"] = -d4["horz_attack_angle"]
    out = d4[KEEP + ["horz_attack_angle_pull"]]
    dest = DATA / "swings_model.parquet"
    out.to_parquet(dest, index=False)
    print(f"\nWrote {dest} ({dest.stat().st_size/1e6:.1f} MB) | {len(out):,} swings | "
          f"{out.batter_id.nunique()} batters")

    # cohort feasibility after filtering
    per = out.groupby("batter_id").size()
    print("\nPer-batter competitive-swing counts (2024-26 pooled):")
    for thr in (100, 200, 300, 500):
        print(f"  >= {thr:>3}: {int((per >= thr).sum())} batters")
    q = per.quantile([.25, .5, .75]).astype(int)
    print(f"  p25={q[.25]}, median={q[.5]}, p75={q[.75]}, max={per.max()}")

    # residual artifact check (NOT dropped — user filters only; flag for decision)
    print("\nResidual extreme-angle check on the competitive set (candidates for a further trim):")
    hz = out["horz_attack_angle"].abs()
    vt = out["vert_attack_angle"]
    print(f"  |horz_attack_angle| > 45 : {int((hz > 45).sum()):,} ({(hz > 45).mean()*100:.2f}%)")
    print(f"  vert_attack_angle < -30  : {int((vt < -30).sum()):,} ({(vt < -30).mean()*100:.2f}%)")
    print(f"  vert_attack_angle >  60  : {int((vt > 60).sum()):,} ({(vt > 60).mean()*100:.2f}%)")
    print("\n  shape feature ranges after filtering:")
    print(out[SHAPE].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
