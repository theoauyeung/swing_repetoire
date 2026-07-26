"""Local DuckDB connection with views over all project parquets and CSVs.

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
    adjustability       — data/adjustability.parquet               ( 471 rows,   9 cols — adj_count, adj_pitch, adj_gamestate)
    xrv_swings          — data/xrv_swings.parquet                  (796k rows,  13 cols — per-swing xRV grades)

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
    "adjustability":       "data/adjustability.parquet",
    "xrv_swings":          "data/xrv_swings.parquet",
}

_CSVS = {
    "re24":               "data/re24.csv",
    "count_values":       "data/count_values.csv",
    "count_transitions":  "data/count_transitions.csv",
    "linear_weights":     "data/linear_weights.csv",
}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name, rel in _PARQUETS.items():
        path = str(ROOT / rel)
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    for name, rel in _CSVS.items():
        path = str(ROOT / rel)
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")
    return con
