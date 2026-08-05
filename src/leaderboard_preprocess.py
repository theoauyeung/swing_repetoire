"""Boils the big swing files down to small per-hitter summaries so the R leaderboard script can
read them. R chokes trying to open the full files on this machine, so all the counting happens
here in Python first.

Arrow's read_parquet hangs on macOS (errno 89) when multiple large files are open
simultaneously. This script does all the heavy lifting in Python/DuckDB and writes
four small (< 200 KB) parquets that R reads without touching the large files.

Run:  python src/leaderboard_preprocess.py
      (or called by the notebook REGEN cell before launching R)
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import db

# Output paths — named here so the SQL blocks stay readable
batter_names_output        = str(ROOT / "data" / "r_batter_names.parquet")
swing_plus_by_shape_output = str(ROOT / "data" / "r_swing_plus_shape.parquet")
swing_plus_by_batter_output = str(ROOT / "data" / "r_swing_plus_batter.parquet")
swing_plus_by_unit_output  = str(ROOT / "data" / "r_swing_plus_unit.parquet")

connection = db.connect("swings", "xrv_swings", "cluster_assignments")

# 1. Distinct batter names  (replaces 80 MB swings_model.parquet read in R)
connection.sql(f"""
    COPY (SELECT DISTINCT batter_id, batter_full_name FROM swings)
    TO '{batter_names_output}' (FORMAT PARQUET)
""")

# 2. Swing+ by shape  (replaces cluster_assignments + xrv_swings join + aggregation in R)
#    UsageProp denominator includes sub-MIN_CLUSTER_SWINGS shapes, matching R's drop_last logic.
connection.sql(f"""
    COPY (
        WITH shape AS (
            SELECT ca.batter_id, ca.batter_stand, ca.cluster,
                   COUNT(*)                    AS Swings,
                   ROUND(AVG(x.xrv_grade), 1) AS SwingPlus
            FROM cluster_assignments ca
            JOIN xrv_swings x USING (play_id)
            WHERE x.game_year IN (2024, 2025)
            GROUP BY ca.batter_id, ca.batter_stand, ca.cluster
        ),
        unit AS (
            SELECT batter_id, batter_stand, SUM(Swings) AS total_swings
            FROM shape GROUP BY batter_id, batter_stand
        )
        SELECT s.batter_id, s.batter_stand, s.cluster,
               s.Swings, s.SwingPlus,
               -- Cast to DOUBLE before dividing so integer division doesn't truncate the proportion
               s.Swings::DOUBLE / u.total_swings AS UsageProp
        FROM shape s JOIN unit u USING (batter_id, batter_stand)
    ) TO '{swing_plus_by_shape_output}' (FORMAT PARQUET)
""")

# 3. Swing+ by batter  (replaces 52 MB xrv_swings.parquet read + aggregation in R)
connection.sql(f"""
    COPY (
        SELECT batter_id,
               COUNT(*)                    AS Swings,
               ROUND(AVG(xrv_grade), 1)   AS SwingPlus
        FROM xrv_swings
        WHERE game_year IN (2024, 2025)
        GROUP BY batter_id
    ) TO '{swing_plus_by_batter_output}' (FORMAT PARQUET)
""")

# 4. Swing+ by unit (batter × stand)  (replaces cluster_assignments + xrv_swings join in R)
connection.sql(f"""
    COPY (
        SELECT ca.batter_id, ca.batter_stand,
               ROUND(AVG(x.xrv_grade), 1) AS SwingPlus
        FROM cluster_assignments ca
        JOIN xrv_swings x USING (play_id)
        WHERE x.game_year IN (2024, 2025)
        GROUP BY ca.batter_id, ca.batter_stand
    ) TO '{swing_plus_by_unit_output}' (FORMAT PARQUET)
""")

connection.close()
print("leaderboard_preprocess: done")
