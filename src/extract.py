"""M1 extract: pull all MLB competitive swings (2024-2026) with bat-tracking columns.

Pulls the superset of *all* swings (is_swing=1), not just bat-tracked ones, so the
missingness audit can compare tracked vs untracked. Bat-tracking columns are null where
the swing was not tracked; `has_bat_tracking` flags the tracked subset.

Joins pbp_raw + pbp_descriptions (1:1 on play_id) + players (batter anthropometry).
Writes: data/swings_2024_2026_mlb.parquet, data/sample_1000.csv, data/profile.md

Run:  python src/extract.py
Creds (BIOMECH_DB_*) resolve from ~/.claude/.env per the mlb-db-analysis skill.
"""
import os, re, warnings
from pathlib import Path
import numpy as np
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
    v = os.environ.get(name)
    if v:
        return v
    ef = Path.home() / ".claude" / ".env"
    if ef.exists():
        for line in ef.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(rf'^\s*{re.escape(name)}\s*=\s*(.+)$', line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
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
  r.batter_id, r.batter_full_name, r.batter_stand, r.pitcher_id,
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
    cn = connect()
    df = pd.read_sql(QUERY, cn)
    cn.close()
    df["has_bat_tracking"] = df["bat_speed"].notna()
    print(f"Pulled {len(df):,} swings | {df.batter_id.nunique()} batters")

    out = DATA / "swings_2024_2026_mlb.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    df.sample(min(1000, len(df)), random_state=7).to_csv(DATA / "sample_1000.csv", index=False)
    print(f"Wrote {DATA/'sample_1000.csv'}")

    write_profile(df)
    print(f"Wrote {DATA/'profile.md'}")


def write_profile(df):
    L = []
    w = L.append
    w("# Data profile — swings_2024_2026_mlb.parquet\n")
    w(f"- Total swings (is_swing=1, MLB): **{len(df):,}**")
    w(f"- Distinct batters: **{df.batter_id.nunique()}**")
    w(f"- Bat-tracked swings: **{int(df.has_bat_tracking.sum()):,}** "
      f"({df.has_bat_tracking.mean()*100:.1f}%)\n")

    w("## Swings & tracking rate by year")
    by_year = df.groupby("game_year").agg(
        swings=("play_id", "size"),
        tracked=("has_bat_tracking", "sum"),
    )
    by_year["tracked_rate"] = (by_year["tracked"] / by_year["swings"] * 100).round(1)
    w(by_year.to_markdown() + "\n")

    w("## Tracking rate by pitch_outcome (X=in play, S=strike/foul, others)")
    by_out = df.groupby("pitch_outcome").agg(
        swings=("play_id", "size"),
        tracked=("has_bat_tracking", "sum"),
    )
    by_out["tracked_rate"] = (by_out["tracked"] / by_out["swings"] * 100).round(1)
    w(by_out.sort_values("swings", ascending=False).to_markdown() + "\n")

    w("## Per-batter POOLED tracked-swing counts (cohort feasibility)")
    per = df[df.has_bat_tracking].groupby("batter_id").size()
    w(f"- batters with >=1 tracked swing: {per.size}")
    for thr in (100, 200, 300, 500):
        w(f"- batters with >= {thr} tracked swings (2024-26 pooled): **{int((per >= thr).sum())}**")
    q = per.quantile([.25, .5, .75, .9]).astype(int)
    w(f"- distribution: p25={q[.25]}, median={q[.5]}, p75={q[.75]}, p90={q[.9]}, max={per.max()}\n")

    w("## Null rate by column (%)")
    nulls = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    nulls = nulls[nulls > 0]
    w(nulls.to_frame("null_%").to_markdown() + "\n")

    w("## Shape features — summary (tracked subset)")
    desc = df.loc[df.has_bat_tracking, SHAPE_FEATURES + INTERCEPT].describe().T
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    w(desc.to_markdown() + "\n")

    w("## Outcome columns — summary (tracked subset)")
    oc = ["exit_velo", "launch_angle", "bearing_angle", "woba", "xwoba", "delta_run_exp"]
    od = df.loc[df.has_bat_tracking, oc].describe().T
    od = od[["count", "mean", "std", "min", "50%", "max"]].round(3)
    w(od.to_markdown() + "\n")

    (DATA / "profile.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
