"""Throws out the swings that aren't real swings — bunts, checked swings, emergency hacks — and
keeps the ones a hitter actually meant. Everything downstream assumes he was trying to hit the
ball hard, so a defensive flail left in the data would read as a deliberate shape change.

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
    all_swings = pd.read_parquet(DATA / "swings_2024_2026_mlb.parquet")
    total_swing_count = len(all_swings)

    # filter funnel — each step narrows the set; names reflect what survived
    all_shape_features_present = (
        all_swings["bat_speed"].notna()
        & all_swings[SHAPE].notna().all(axis=1)
    )
    bat_tracked_swings = all_swings[all_shape_features_present]
    non_bunt_swings    = bat_tracked_swings[bat_tracked_swings["is_bunt"] == 0]
    fast_enough_swings = non_bunt_swings[non_bunt_swings["bat_speed"] >= BAT_SPEED_MIN]

    horz_within_bounds = fast_enough_swings["horz_attack_angle"].abs() <= HORZ_ABS_MAX
    vert_within_bounds = fast_enough_swings["vert_attack_angle"].between(*VERT_BOUNDS)
    competitive_swings = fast_enough_swings[horz_within_bounds & vert_within_bounds].copy()

    print("Filter funnel:")
    print(f"  all swings                 : {total_swing_count:>10,}")
    print(f"  bat-tracked (5 feats)      : {len(bat_tracked_swings):>10,}  ({len(bat_tracked_swings)/total_swing_count*100:.1f}%)")
    print(f"  - bunts removed            : {len(non_bunt_swings):>10,}  (-{len(bat_tracked_swings)-len(non_bunt_swings):,})")
    print(f"  - bat_speed >= {BAT_SPEED_MIN:g}         : {len(fast_enough_swings):>10,}  (-{len(non_bunt_swings)-len(fast_enough_swings):,})")
    print(f"  - angle artifacts dropped  : {len(competitive_swings):>10,}  (-{len(fast_enough_swings)-len(competitive_swings):,})")

    # pull frame (+ = pull side, both hands). horz_attack_angle is batter-relative with raw + = oppo,
    # so pull is a uniform negation (validated vs bearing_angle) — NOT a per-hand mirror.
    competitive_swings["horz_attack_angle_pull"] = -competitive_swings["horz_attack_angle"]

    output_columns = KEEP + ["horz_attack_angle_pull"]
    output_swings  = competitive_swings[output_columns]
    output_path    = DATA / "swings_model.parquet"
    output_swings.to_parquet(output_path, index=False)
    print(f"\nWrote {output_path} ({output_path.stat().st_size/1e6:.1f} MB) | {len(output_swings):,} swings | "
          f"{output_swings.batter_id.nunique()} batters")

    # cohort feasibility after filtering
    swings_per_batter = output_swings.groupby("batter_id").size()
    print("\nPer-batter competitive-swing counts (2024-26 pooled):")
    for threshold in (100, 200, 300, 500):
        print(f"  >= {threshold:>3}: {int((swings_per_batter >= threshold).sum())} batters")

    # quantile summary for the per-batter distribution
    quartiles = swings_per_batter.quantile([.25, .5, .75]).astype(int)
    print(f"  p25={quartiles[.25]}, median={quartiles[.5]}, p75={quartiles[.75]}, max={swings_per_batter.max()}")

    # residual artifact check (NOT dropped — user filters only; flag for decision)
    print("\nResidual extreme-angle check on the competitive set (candidates for a further trim):")
    horz_abs = output_swings["horz_attack_angle"].abs()
    vert     = output_swings["vert_attack_angle"]
    print(f"  |horz_attack_angle| > 45 : {int((horz_abs > 45).sum()):,} ({(horz_abs > 45).mean()*100:.2f}%)")
    print(f"  vert_attack_angle < -30  : {int((vert < -30).sum()):,} ({(vert < -30).mean()*100:.2f}%)")
    print(f"  vert_attack_angle >  60  : {int((vert > 60).sum()):,} ({(vert > 60).mean()*100:.2f}%)")

    # compute summary stats then select only the relevant columns for display
    print("\n  shape feature ranges after filtering:")
    shape_stats_full    = output_swings[SHAPE].describe().T
    shape_stats_display = shape_stats_full[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(shape_stats_display.round(2).to_string())


if __name__ == "__main__":
    main()
