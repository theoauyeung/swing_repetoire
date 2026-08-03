"""Optimal adjustment policy — annualized situational run values.

Translates per-swing matched penalties (from adjustability.parquet) into
seasonal run totals by multiplying by actual 2024-25 per-season exposure
counts (from swings_model.parquet). Joins primary archetype and repertoire
width for HTE analysis in the notebook.

Output: data/optimal_policy.parquet
Run   : python src/optimal_policy.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEASONS = [2024, 2025]
N_SEASONS = len(SEASONS)


def compute_exposure(sw: pd.DataFrame) -> pd.DataFrame:
    """Per-(batter, stand) annual swing counts for each situation axis."""
    sw = sw[sw["game_year"].isin(SEASONS)].copy()
    sw["any_runner"] = (
        sw["on_1b_id"].notna() | sw["on_2b_id"].notna() | sw["on_3b_id"].notna()
    )
    sw["same_hand"] = sw["batter_stand"] == sw["pitcher_throws"]

    key = ["batter_id", "batter_stand"]
    n_2k = (
        sw[sw["strikes"] == 2]
        .groupby(key)
        .size()
        .reset_index(name="n_2k_total")
    )
    n_gs = (
        sw[sw["any_runner"]]
        .groupby(key)
        .size()
        .reset_index(name="n_gamestate_total")
    )
    n_pl = (
        sw[sw["same_hand"]]
        .groupby(key)
        .size()
        .reset_index(name="n_platoon_total")
    )

    exp = n_2k.merge(n_gs, on=key, how="outer").merge(n_pl, on=key, how="outer")
    for col in ["n_2k_total", "n_gamestate_total", "n_platoon_total"]:
        exp[col] = exp[col].fillna(0)
    exp["n_2k_per_season"] = exp["n_2k_total"] / N_SEASONS
    exp["n_gamestate_per_season"] = exp["n_gamestate_total"] / N_SEASONS
    exp["n_platoon_per_season"] = exp["n_platoon_total"] / N_SEASONS
    return exp[key + ["n_2k_per_season", "n_gamestate_per_season", "n_platoon_per_season"]]


def load_primary(cards: pd.DataFrame) -> pd.DataFrame:
    """Primary shape archetype and grade per (batter, stand)."""
    primary = cards[cards["role"] == "primary"][
        ["batter_id", "batter_stand", "archetype_name", "grade"]
    ].rename(columns={"grade": "primary_grade"})
    return primary


def main():
    print("Loading data...")
    adj = pd.read_parquet(
        DATA / "adjustability.parquet",
        columns=[
            "batter_id", "batter_stand", "label", "n_swings",
            "adj_count", "adjustability_plus", "adjustability_pctile", "swing_plus",
            "twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty",
        ],
    )
    sw = pd.read_parquet(
        DATA / "swings_model.parquet",
        columns=[
            "batter_id", "batter_stand", "game_year", "strikes",
            "on_1b_id", "on_2b_id", "on_3b_id", "pitcher_throws",
        ],
    )
    cards = pd.read_parquet(
        DATA / "shape_cards.parquet",
        columns=["batter_id", "batter_stand", "role", "archetype_name", "grade"],
    )
    rep = pd.read_parquet(
        DATA / "repertoire_scores.parquet",
        columns=["batter_id", "batter_stand", "repertoire_pctile", "effective_shapes"],
    )

    print("Computing exposure counts...")
    exp = compute_exposure(sw)

    key = ["batter_id", "batter_stand"]
    df = adj.merge(exp, on=key, how="left")

    print("Computing seasonal run values...")
    df["season_runs_2k"] = df["twostrike_rv_penalty"] * df["n_2k_per_season"]
    df["season_runs_gamestate"] = df["gamestate_rv_penalty"] * df["n_gamestate_per_season"]
    # Switch-hitter units have null platoon_rv_penalty — propagate NaN
    df["season_runs_platoon"] = df["platoon_rv_penalty"] * df["n_platoon_per_season"]
    # Total sums non-null axes; switch hitters get 2K + GS only
    df["season_runs_total"] = df[["season_runs_2k", "season_runs_gamestate", "season_runs_platoon"]].sum(
        axis=1, min_count=1
    )

    print("Joining archetype and repertoire...")
    primary = load_primary(cards)
    df = df.merge(primary, on=key, how="left")
    df = df.merge(rep, on=key, how="left")

    out_cols = [
        "batter_id", "batter_stand", "label", "n_swings",
        "adj_count", "adjustability_plus", "adjustability_pctile", "swing_plus",
        "twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty",
        "n_2k_per_season", "n_gamestate_per_season", "n_platoon_per_season",
        "season_runs_2k", "season_runs_gamestate", "season_runs_platoon", "season_runs_total",
        "archetype_name", "primary_grade",
        "repertoire_pctile", "effective_shapes",
    ]
    df = df[out_cols]

    out = DATA / "optimal_policy.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows → {out}")

    print("\n=== Season runs distribution ===")
    sr = df["season_runs_total"].dropna()
    print(f"  mean={sr.mean():.2f}  median={sr.median():.2f}  min={sr.min():.2f}  max={sr.max():.2f}")
    print("\n=== Spot check ===")
    spot = df[df["label"].str.contains("Judge|Arraez", na=False)]
    print(spot[["label", "batter_stand", "n_2k_per_season", "season_runs_2k",
                "season_runs_total", "adj_count", "archetype_name", "primary_grade"]].to_string(index=False))


if __name__ == "__main__":
    main()
