"""M1 extract: pull all MLB competitive swings (2024-2026) with bat-tracking columns.

Pulls the superset of *all* swings (is_swing=1), not just bat-tracked ones, so the
missingness audit can compare tracked vs untracked. Bat-tracking columns are null where
the swing was not tracked; `has_bat_tracking` flags the tracked subset.

Joins pbp_raw + pbp_descriptions (1:1 on play_id) + players (batter anthropometry).
Writes: data/swings_2024_2026_mlb.parquet, data/sample_1000.csv, data/profile.md

Run:  python src/extract.py
Creds (BIOMECH_DB_*) resolve from ~/.claude/.env per the mlb-db-analysis skill.
"""
import os
import re
import warnings
from pathlib import Path
import pandas as pd
import mysql.connector

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SHAPE_FEATURES = ["swing_path_tilt", "swing_length", "bat_speed",
                  "vert_attack_angle", "horz_attack_angle"]
INTERCEPT = ["ball_bat_intercept_x", "ball_bat_intercept_y", "ball_intercept_z", "ball_bat_miss"]
SEASONS = (2024, 2025, 2026)


def get_secret(name):
    env_value = os.environ.get(name)
    if env_value:
        return env_value
    env_file = Path.home() / ".claude" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.match(rf'^\s*{re.escape(name)}\s*=\s*(.+)$', line)
            if match:
                return match.group(1).strip().strip('"').strip("'")
    return None


def connect():
    return mysql.connector.connect(
        host=get_secret("BIOMECH_DB_HOST"),
        port=int(get_secret("BIOMECH_DB_PORT") or 3306),
        user=get_secret("BIOMECH_DB_USER"),
        password=get_secret("BIOMECH_DB_PASS"),
        database="mlb_db",
    )


QUERY = f"""
SELECT
  r.play_id, r.game_pk, r.game_date, r.game_year,
  r.batter_id, r.batter_full_name, r.batter_stand, r.pitcher_id, r.pitcher_throws,
  r.at_bat_number, r.pitch_number, r.pitch_type,
  r.balls, r.strikes, r.outs_when_up,
  r.plate_x, r.plate_z, r.plate_zone,
  r.{', r.'.join(SHAPE_FEATURES)},
  r.{', r.'.join(INTERCEPT)},
  r.exit_velo, r.launch_angle, r.bb_type, r.bearing_angle,
  r.woba, r.xwoba, r.pa_outcome, r.pitch_outcome,
  CASE WHEN r.pitch_outcome_explanation IN ('foul_bunt','missed_bunt','bunt_foul_tip')
         OR r.pa_outcome = 'sac_bunt'
         OR r.pa_outcome_explanation LIKE '%bunt%'
       THEN 1 ELSE 0 END AS is_bunt,
  d.is_swing, d.is_whiff, d.is_contact, d.is_bip, d.delta_run_exp,
  d.on_1b_id, d.on_2b_id, d.on_3b_id,
  p.sz_top, p.sz_bot, p.height
FROM pbp_raw r
JOIN pbp_descriptions d ON d.play_id = r.play_id
LEFT JOIN players p ON p.mlbam_id = r.batter_id
WHERE r.level_id = 1
  AND r.game_year IN ({','.join(map(str, SEASONS))})
  AND d.is_swing = 1
"""


def main():
    print("Connecting and querying (all MLB swings 2024-2026)...")
    connection = connect()
    swings = pd.read_sql(QUERY, connection)
    connection.close()
    swings["has_bat_tracking"] = swings["bat_speed"].notna()
    print(f"Pulled {len(swings):,} swings | {swings.batter_id.nunique()} batters")

    parquet_path = DATA / "swings_2024_2026_mlb.parquet"
    swings.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({parquet_path.stat().st_size/1e6:.1f} MB)")

    swings.sample(min(1000, len(swings)), random_state=7).to_csv(DATA / "sample_1000.csv", index=False)
    print(f"Wrote {DATA/'sample_1000.csv'}")

    write_profile(swings)
    print(f"Wrote {DATA/'profile.md'}")


def write_profile(swings):
    lines = []
    lines.append("# Data profile — swings_2024_2026_mlb.parquet\n")
    lines.append(f"- Total swings (is_swing=1, MLB): **{len(swings):,}**")
    lines.append(f"- Distinct batters: **{swings.batter_id.nunique()}**")
    lines.append(f"- Bat-tracked swings: **{int(swings.has_bat_tracking.sum()):,}** "
                 f"({swings.has_bat_tracking.mean()*100:.1f}%)\n")

    lines.append("## Swings & tracking rate by year")
    by_year = swings.groupby("game_year").agg(
        swing_count=("play_id", "size"),
        tracked=("has_bat_tracking", "sum"),
    )
    by_year["tracked_rate"] = (by_year["tracked"] / by_year["swing_count"] * 100).round(1)
    lines.append(by_year.to_markdown() + "\n")

    lines.append("## Tracking rate by pitch_outcome (X=in play, S=strike/foul, others)")
    by_outcome = swings.groupby("pitch_outcome").agg(
        swing_count=("play_id", "size"),
        tracked=("has_bat_tracking", "sum"),
    )
    by_outcome["tracked_rate"] = (by_outcome["tracked"] / by_outcome["swing_count"] * 100).round(1)
    lines.append(by_outcome.sort_values("swing_count", ascending=False).to_markdown() + "\n")

    lines.append("## Per-batter POOLED tracked-swing counts (cohort feasibility)")
    swings_per_batter = swings[swings.has_bat_tracking].groupby("batter_id").size()
    lines.append(f"- batters with >=1 tracked swing: {swings_per_batter.size}")
    for threshold in (100, 200, 300, 500):
        lines.append(f"- batters with >= {threshold} tracked swings (2024-26 pooled): "
                     f"**{int((swings_per_batter >= threshold).sum())}**")
    quartile_values = swings_per_batter.quantile([.25, .5, .75, .9]).astype(int)
    lines.append(f"- distribution: p25={quartile_values[.25]}, median={quartile_values[.5]}, "
                 f"p75={quartile_values[.75]}, p90={quartile_values[.9]}, max={swings_per_batter.max()}\n")

    lines.append("## Null rate by column (%)")
    null_rates = (swings.isna().mean() * 100).round(1).sort_values(ascending=False)
    null_rates = null_rates[null_rates > 0]
    lines.append(null_rates.to_frame("null_%").to_markdown() + "\n")

    lines.append("## Shape features — summary (tracked subset)")
    shape_stats = swings.loc[swings.has_bat_tracking, SHAPE_FEATURES + INTERCEPT].describe().T
    shape_stats = shape_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    lines.append(shape_stats.to_markdown() + "\n")

    lines.append("## Outcome columns — summary (tracked subset)")
    outcome_cols = ["exit_velo", "launch_angle", "bearing_angle", "woba", "xwoba", "delta_run_exp"]
    outcome_stats = swings.loc[swings.has_bat_tracking, outcome_cols].describe().T
    outcome_stats = outcome_stats[["count", "mean", "std", "min", "50%", "max"]].round(3)
    lines.append(outcome_stats.to_markdown() + "\n")

    (DATA / "profile.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
