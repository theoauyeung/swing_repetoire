"""Optimal adjustment policy — annualized situational run values (partially pooled).

Translates matched situational penalties into seasonal run totals by multiplying
by actual 2024-25 per-season exposure counts. Penalties are empirical-Bayes
shrunk toward a between-batter model prediction before being annualized, because
the raw single-season penalty barely repeats (two-strike YoY r ≈ 0.15) while the
adjustability trait driving it does (r ≈ 0.67). Without shrinkage roughly 85% of
the seasonal-run spread is noise.

Two distinct quantities are produced per axis:
  season_runs_*  — de-noised estimate of what the hitter's situational run delta
                   actually was (EB posterior × exposure). The leaderboard number.
  adj_runs_*     — the portion attributable to adjustability specifically
                   (θ · z_adj · sd_penalty × exposure). The "what is adjusting
                   worth" number.

No cross-axis total is produced. The three axes overlap heavily (exposure counts
sum to 1.265× actual swings — a two-strike swing with a runner on against a
same-hand pitcher is in all three), so summing them double-counts. A disjoint
8-state decomposition was tested and rejected: only 298 of 471 units retain all
seven non-reference states under the ≥3/≥3 cell filter.

Output: data/optimal_policy.parquet
Run   : python src/optimal_policy.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEASONS = [2024, 2025]
N_SEASONS = len(SEASONS)
KEY = ["batter_id", "batter_stand"]

MIN_CELL = 3        # swings required on each side of a matched cell
MIN_SEASON_W = 30   # matched weight required in a season to enter the reliability fit

# (axis key, comparison-state column, extra matching columns beyond pitch_type × zone)
AXES = [
    ("2k",        "is_2k",       []),
    ("gamestate", "any_runner",  ["strikes"]),
    ("platoon",   "same_hand",   ["strikes"]),
]
PENALTY_COL = {
    "2k":        "twostrike_rv_penalty",
    "gamestate": "gamestate_rv_penalty",
    "platoon":   "platoon_rv_penalty",
}


def add_states(sw: pd.DataFrame) -> pd.DataFrame:
    sw = sw[sw["game_year"].isin(SEASONS)].copy()
    sw["is_2k"] = sw["strikes"] == 2
    sw["any_runner"] = (
        sw["on_1b_id"].notna() | sw["on_2b_id"].notna() | sw["on_3b_id"].notna()
    )
    sw["same_hand"] = sw["batter_stand"] == sw["pitcher_throws"]
    sw["zone"] = sw["plate_zone"].astype("Int64").astype(str)
    return sw.dropna(subset=["delta_run_exp", "pitch_type", "plate_zone"])


def matched_penalty(sw: pd.DataFrame, comp_col: str,
                    extra_cells: list) -> pd.DataFrame:
    """n-weighted mean Δrv (comparison − reference) over matched cells, per unit.

    Returns a frame indexed by KEY with columns [penalty, matched_n], where
    matched_n is the comparison-state swing count the penalty averages over —
    the effective sample size used for empirical-Bayes weighting.
    """
    cells = ["pitch_type", "zone"] + extra_cells
    g = (sw.groupby(KEY + cells + [comp_col], observed=True)["delta_run_exp"]
           .agg(mean="mean", n="size")
           .unstack(comp_col))
    if ("mean", True) not in g.columns or ("mean", False) not in g.columns:
        return pd.DataFrame(columns=["penalty", "matched_n"])

    n_comp, n_ref = g[("n", True)], g[("n", False)]
    ok = (n_comp >= MIN_CELL) & (n_ref >= MIN_CELL)
    d = pd.DataFrame({
        "delta": (g[("mean", True)] - g[("mean", False)])[ok],
        "w": n_comp[ok],
    }).dropna().reset_index()
    if d.empty:
        return pd.DataFrame(columns=["penalty", "matched_n"])

    d["wd"] = d["delta"] * d["w"]
    agg = d.groupby(KEY).agg(wd=("wd", "sum"), matched_n=("w", "sum"))
    agg["penalty"] = agg["wd"] / agg["matched_n"]
    return agg[["penalty", "matched_n"]]


def stabilization_constant(sw: pd.DataFrame, comp_col: str,
                           extra_cells: list) -> tuple:
    """n₀ for empirical-Bayes shrinkage, from year-over-year reliability.

    Reliability of a single-season penalty at median sample n is r, so the
    stabilization point is n₀ = n(1−r)/r. Units with a non-positive or trivial
    r get a large n₀, i.e. near-total shrinkage to the model.
    """
    per_season = []
    for year in SEASONS:
        p = matched_penalty(sw[sw["game_year"] == year], comp_col, extra_cells)
        p = p[p["matched_n"] >= MIN_SEASON_W]
        per_season.append(p.rename(columns={
            "penalty": f"p{year}", "matched_n": f"n{year}"}))

    both = per_season[0].join(per_season[1], how="inner")
    if len(both) < 50:
        return float("inf"), float("nan"), len(both)

    r = float(both[f"p{SEASONS[0]}"].corr(both[f"p{SEASONS[1]}"]))
    n_med = float(both[[f"n{y}" for y in SEASONS]].stack().median())
    if not np.isfinite(r) or r <= 0.01:
        return float("inf"), r, len(both)
    return n_med * (1.0 - r) / r, r, len(both)


def fit_axis_model(base: pd.DataFrame, penalty_col: str) -> tuple:
    """Between-batter OLS matching adjustability_value.py Section 2.

    penalty_z ~ composite adjustability_z + swing_plus_z + repertoire_pctile_z

    Composite `adjustability` is the treatment (not adj_count) — it is the
    headline metric and the count axis carries the dominant weight within it.
    Returns (fitted penalty in raw units indexed by KEY, θ, sd_penalty).
    """
    cols = [penalty_col, "adjustability", "swing_plus", "repertoire_pctile"]
    df = base.dropna(subset=cols)
    if len(df) < 50:
        return pd.Series(dtype=float), float("nan"), float("nan")

    def _z(s):
        return (s - s.mean()) / s.std()

    y = _z(df[penalty_col]).to_numpy(float)
    adj_z = _z(df["adjustability"]).to_numpy(float)
    X = np.column_stack([
        np.ones(len(df)),
        _z(df["swing_plus"]).to_numpy(float),
        _z(df["repertoire_pctile"]).to_numpy(float),
        adj_z,
    ])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)

    mu, sd = df[penalty_col].mean(), df[penalty_col].std()
    fitted = pd.Series((X @ coefs) * sd + mu,
                       index=pd.MultiIndex.from_frame(df[KEY]))
    return fitted, float(coefs[-1]), float(sd)


def compute_exposure(sw: pd.DataFrame) -> pd.DataFrame:
    """Per-(batter, stand) annual swing counts for each situation axis."""
    out = None
    for axis, comp_col, _ in AXES:
        n = (sw[sw[comp_col]].groupby(KEY).size()
             .rename(f"n_{axis}_per_season") / N_SEASONS)
        out = n.to_frame() if out is None else out.join(n, how="outer")
    return out.fillna(0.0)


def load_primary(cards: pd.DataFrame) -> pd.DataFrame:
    """Primary shape archetype and grade per (batter, stand)."""
    return cards[cards["role"] == "primary"][
        KEY + ["archetype_name", "grade"]
    ].rename(columns={"grade": "primary_grade"})


def main():
    print("Loading data...")
    adj = pd.read_parquet(
        DATA / "adjustability.parquet",
        columns=KEY + [
            "label", "n_swings", "adjustability", "adj_count",
            "adjustability_plus", "adjustability_pctile", "swing_plus",
            "twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty",
        ],
    )
    sw = add_states(pd.read_parquet(
        DATA / "swings_model.parquet",
        columns=KEY + [
            "game_year", "strikes", "pitch_type", "plate_zone",
            "on_1b_id", "on_2b_id", "on_3b_id", "pitcher_throws", "delta_run_exp",
        ],
    ))
    cards = pd.read_parquet(
        DATA / "shape_cards.parquet",
        columns=KEY + ["role", "archetype_name", "grade"],
    )
    rep = pd.read_parquet(
        DATA / "repertoire_scores.parquet",
        columns=KEY + ["repertoire_pctile", "effective_shapes"],
    )

    print("Computing exposure counts...")
    df = adj.merge(compute_exposure(sw), on=KEY, how="left")
    df = df.merge(rep, on=KEY, how="left")
    idx = pd.MultiIndex.from_frame(df[KEY])

    print("\n=== Empirical-Bayes shrinkage ===")
    for axis, comp_col, extra in AXES:
        pen_col = PENALTY_COL[axis]
        n0, r, n_units = stabilization_constant(sw, comp_col, extra)
        fitted, theta, sd_pen = fit_axis_model(df, pen_col)
        matched = matched_penalty(sw, comp_col, extra)["matched_n"]

        m_n = matched.reindex(idx).to_numpy(float)
        w = np.where(np.isfinite(m_n) & np.isfinite(n0), m_n / (m_n + n0), 0.0)
        fit_v = fitted.reindex(idx).to_numpy(float)
        obs_v = df[pen_col].to_numpy(float)

        eb = np.where(np.isfinite(obs_v) & np.isfinite(fit_v),
                      w * obs_v + (1.0 - w) * fit_v, np.nan)
        df[f"{pen_col}_eb"] = eb
        df[f"eb_weight_{axis}"] = np.where(np.isfinite(obs_v), w, np.nan)

        exposure = df[f"n_{axis}_per_season"]
        df[f"season_runs_{axis}"] = eb * exposure
        # Adjustability-attributable component: θ · z_adj · sd_penalty × exposure
        adj_z = (df["adjustability"] - df["adjustability"].mean()) / df["adjustability"].std()
        df[f"adj_runs_{axis}"] = theta * adj_z * sd_pen * exposure

        print(f"  {axis:<10} YoY r={r:.3f} (n={n_units})  n0={n0:,.0f}  "
              f"mean EB weight={np.nanmean(df[f'eb_weight_{axis}']):.3f}  θ={theta:+.4f}")

    print("Joining archetype...")
    df = df.merge(load_primary(cards), on=KEY, how="left")

    out_cols = (
        KEY + ["label", "n_swings", "adjustability", "adj_count",
               "adjustability_plus", "adjustability_pctile", "swing_plus"]
        + [PENALTY_COL[a] for a, _, _ in AXES]
        + [f"{PENALTY_COL[a]}_eb" for a, _, _ in AXES]
        + [f"eb_weight_{a}" for a, _, _ in AXES]
        + [f"n_{a}_per_season" for a, _, _ in AXES]
        + [f"season_runs_{a}" for a, _, _ in AXES]
        + [f"adj_runs_{a}" for a, _, _ in AXES]
        + ["archetype_name", "primary_grade", "repertoire_pctile", "effective_shapes"]
    )
    df = df[out_cols]

    out = DATA / "optimal_policy.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} rows → {out}")

    print("\n=== Two-strike season runs: raw vs shrunk ===")
    raw = df["twostrike_rv_penalty"] * df["n_2k_per_season"]
    for name, s in [("raw", raw), ("shrunk", df["season_runs_2k"])]:
        s = s.dropna()
        print(f"  {name:<7} sd={s.std():5.2f}  range=[{s.min():6.1f}, {s.max():5.1f}]")
    ar = df["adj_runs_2k"].dropna()
    print(f"  adjustability-attributable: sd={ar.std():.2f}  "
          f"P90−P10={ar.quantile(0.9) - ar.quantile(0.1):+.2f} runs/season")

    print("\n=== Spot check ===")
    spot = df[df["label"].str.contains("Judge|Arraez", na=False)]
    print(spot[["label", "batter_stand", "n_2k_per_season", "twostrike_rv_penalty",
                "eb_weight_2k", "season_runs_2k", "adj_runs_2k",
                "adjustability"]].to_string(index=False))


if __name__ == "__main__":
    main()
