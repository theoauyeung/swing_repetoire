"""Opens every project data file as a queryable table so you can ask questions in SQL instead of
loading files by hand. Handy for one-off checks like "who are the top 10 by adjustability".

Usage
-----
    import db
    con = db.connect()
    con.sql("SELECT * FROM swings LIMIT 5").df()
    con.sql("SELECT batter_id, adjustability FROM adjustability ORDER BY adjustability DESC").df()
    con.sql("SELECT * FROM swings JOIN cluster_assignments USING (play_id) LIMIT 10").df()

Views (parquet)
---------------
    swings_raw          — data/swings_2024_2026_mlb.parquet       (1.04M rows, 48 cols — full DB extract)
    swings              — data/swings_model.parquet                (796k rows,  40 cols — competitive swings)
    cluster_assignments — data/cluster_assignments.parquet         (780k rows,   5 cols — per-swing cluster labels)
    cluster_summary     — data/cluster_summary.parquet             (1592 rows,  12 cols — per-(unit, cluster) centroids)
    batter_repertoire   — data/batter_repertoire.parquet           ( 703 rows,  11 cols — per-unit k and diversity scalars)
    shape_archetypes    — data/shape_archetypes.parquet            (1592 rows,  14 cols — per-cluster archetype labels)
    archetype_lexicon   — data/archetype_lexicon.parquet           (   3 rows,   8 cols — league-level archetype definitions)
    shape_cards         — data/shape_cards.parquet                 (1592 rows,  18 cols — per-cluster swing ID cards)
    repertoire_scores   — data/repertoire_scores.parquet           ( 703 rows,  15 cols — Repertoire+ and pctile)
    adjustability       — data/adjustability.parquet               ( 471 rows,  28 cols — adj_count, adj_pitch, adj_gamestate, adj_platoon; twostrike/risp/dp/platoon rv+whiff penalties; swing_plus[xrv_grade_neutral])
    xrv_swings          — data/xrv_swings.parquet                  (796k rows,  15 cols — xrv_grade (count-inclusive), xrv_grade_neutral (count-stripped))

Views (CSV — xRV building blocks)
----------------------------------
    re24                — data/re24.csv              (  8 rows — RE24 matrix)
    count_values        — data/count_values.csv      ( 12 rows — count run values)
    count_transitions   — data/count_transitions.csv ( 72 rows — count transition probabilities)
    linear_weights      — data/linear_weights.csv    (  9 rows — linear weights by outcome)
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent

_PARQUETS = {
    "swings_raw":          "data/swings_2024_2026_mlb.parquet",
    "swings":              "data/swings_model.parquet",
    "cluster_assignments": "data/cluster_assignments.parquet",
    "cluster_summary":     "data/cluster_summary.parquet",
    "batter_repertoire":   "data/batter_repertoire.parquet",
    "shape_archetypes":    "data/shape_archetypes.parquet",
    "archetype_lexicon":   "data/archetype_lexicon.parquet",
    "shape_cards":         "data/shape_cards.parquet",
    "repertoire_scores":   "data/repertoire_scores.parquet",
    "adjustability":       "data/adjustability.parquet",  # 471 rows, 28 cols
    "xrv_swings":          "data/xrv_swings.parquet",
}

_CSVS = {
    "re24":               "data/re24.csv",
    "count_values":       "data/count_values.csv",
    "count_transitions":  "data/count_transitions.csv",
    "linear_weights":     "data/linear_weights.csv",
}


def connect(*only: str) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with views over project parquets and CSVs.

    Pass view names to restrict which views are created (avoids reading metadata
    for large files you don't need):

        con = db.connect("adjustability", "repertoire_scores")
        con = db.connect()  # all views (original behavior)
    """
    con = duckdb.connect()

    # Filter to only the requested views; if no names given, include everything
    requested_parquets = {}
    for view_name, relative_path in _PARQUETS.items():
        if not only or view_name in only:
            requested_parquets[view_name] = relative_path

    requested_csvs = {}
    for view_name, relative_path in _CSVS.items():
        if not only or view_name in only:
            requested_csvs[view_name] = relative_path

    for view_name, relative_path in requested_parquets.items():
        absolute_path = str(ROOT / relative_path)
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{absolute_path}')")

    for view_name, relative_path in requested_csvs.items():
        absolute_path = str(ROOT / relative_path)
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{absolute_path}')")

    return con
