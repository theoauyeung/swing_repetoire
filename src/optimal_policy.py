"""Optimal adjustment policy — annualized situational run values (additive).

Produces a per-hitter seasonal run total that genuinely sums across the three
situation axes (two-strike, runners on, same-hand matchup).

Why this is not just (penalty x exposure) summed
------------------------------------------------
The published penalties in adjustability.parquet are *marginal* contrasts: each
asks "2K vs not-2K" while ignoring the other two states. A single swing can be
two-strike AND runners-on AND same-hand, so those three exposures overlap
(they sum to 1.265x the actual swing count) and adding them double-counts.

Instead, this script estimates the three effects *jointly*, per hitter, within
matched pitch_type x plate_zone cells:

    delta_run_exp ~ cell FE + b_2k*is_2k + b_gs*any_runner + b_pl*same_hand

Each b is now a partial effect holding the other two states fixed, so a swing
that is both two-strike and runners-on contributes b_2k + b_gs -- counted once,
with both effects. The season total is therefore a true sum:

    total = sum_axis (b_axis * n_axis_per_season)

Cell fixed effects are absorbed by within-cell demeaning (Frisch-Waugh), which
gives coefficients identical to including the dummies explicitly.

Shrinkage
---------
Raw single-season effects barely repeat (two-strike YoY r ~ 0.17) while the
adjustability trait driving them repeats at r ~ 0.67, so most of the raw
seasonal-run spread is noise. Each b is empirical-Bayes shrunk toward a
between-batter model prediction before annualization, with weight
eff_n / (eff_n + n0). eff_n is the Frisch-Waugh residual sum of squares for that
indicator -- the effective sample identifying it -- so hitters with more genuine
exposure keep more of their own measurement. Axes with no within-cell variation
get eff_n = 0 and fall back entirely to the model; their exposure is ~0 anyway
(58 switch-hitter stances have a median same-hand share of exactly 0.0000).

Baseline
--------
Effects are reported net of a talent baseline -- the same between-batter model
with the adjustability term zeroed, i.e. the situational effect expected from
swing quality and repertoire width alone. Two artifacts make a raw or
league-mean-centered figure misleading: delta_run_exp magnitudes are
mechanically larger with runners on base (league mean b_gs = +0.0108, positive
for nearly everyone), and elite hitters lose more run value in tough situations
simply because they have more to lose. Centering on the league mean leaves the
output correlated -0.63 with Swing+; centering on the talent baseline brings
that to +0.05.

So season_runs_* reads as: runs above/below what a hitter of the same swing
quality and repertoire width would produce given the same situational exposure.
0 = as expected for your talent. Algebraically the reported value is

    w * (observed - baseline) + (1 - w) * theta * z_adjustability * sd

so it degrades gracefully: axes with real sample keep the hitter's own measured
deviation, axes with none fall back to the pure adjustability prediction.

Caveat on the total: only the two-strike axis has a significant adjustability
link in this joint specification (theta=+0.125, p=0.0065). Game-state
(p=0.20) and platoon (theta=-0.079, p=0.12) are null, and the negative platoon
point estimate partially cancels the two-strike gain in adj_runs_total. The
total is therefore closer to an outcome measure than a skill measure.

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

AXES = ["2k", "gamestate", "platoon"]
IND = {"2k": "is_2k", "gamestate": "any_runner", "platoon": "same_hand"}
CELLS = ["pitch_type", "zone"]
MIN_SEASON_N = 20.0   # eff_n required in a season to enter the reliability fit


def add_states(sw: pd.DataFrame) -> pd.DataFrame:
    sw = sw[sw["game_year"].isin(SEASONS)].copy()
    sw["is_2k"] = (sw["strikes"] == 2).astype(float)
    sw["any_runner"] = (
        sw["on_1b_id"].notna() | sw["on_2b_id"].notna() | sw["on_3b_id"].notna()
    ).astype(float)
    sw["same_hand"] = (sw["batter_stand"] == sw["pitcher_throws"]).astype(float)
    sw = sw.dropna(subset=["delta_run_exp", "pitch_type", "plate_zone"])
    sw["zone"] = sw["plate_zone"].astype(int).astype(str)
    return sw


def joint_effects(sw: pd.DataFrame) -> pd.DataFrame:
    """Per-unit partial effect of each situation state, within pitch_type x zone cells.

    Returns a frame indexed by KEY with beta_<axis> and eff_n_<axis> columns.
    eff_n is the Frisch-Waugh residual sum of squares identifying that indicator;
    it is 0 when the state never varies within a cell for that hitter.
    """
    cols = [IND[a] for a in AXES]
    rows = []
    for (bid, stand), d in sw.groupby(KEY, sort=False):
        g = d.groupby(CELLS, observed=True)
        y = (d["delta_run_exp"] - g["delta_run_exp"].transform("mean")).to_numpy(float)
        X = np.column_stack([
            (d[c] - g[c].transform("mean")).to_numpy(float) for c in cols
        ])

        rec = {"batter_id": bid, "batter_stand": stand}
        live = [j for j in range(len(cols)) if X[:, j].std() > 1e-8]
        beta = np.full(len(cols), np.nan)
        if live:
            Xl = X[:, live]
            if np.linalg.cond(Xl.T @ Xl) < 1e10:
                bl, *_ = np.linalg.lstsq(Xl, y, rcond=None)
                beta[live] = bl

        for j, axis in enumerate(AXES):
            rec[f"beta_{axis}"] = beta[j]
            others = [k for k in live if k != j]
            if j not in live:
                rec[f"eff_n_{axis}"] = 0.0
            elif not others:
                rec[f"eff_n_{axis}"] = float(X[:, j] @ X[:, j])
            else:
                Z = X[:, others]
                resid = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
                rec[f"eff_n_{axis}"] = float(resid @ resid)
        rows.append(rec)

    return pd.DataFrame(rows).set_index(KEY)


def stabilization(sw: pd.DataFrame) -> dict:
    """n0 per axis from year-over-year reliability: n0 = n(1-r)/r."""
    per_year = {y: joint_effects(sw[sw["game_year"] == y]) for y in SEASONS}
    out = {}
    for axis in AXES:
        a, b = (per_year[y][[f"beta_{axis}", f"eff_n_{axis}"]] for y in SEASONS)
        j = a.join(b, lsuffix="_0", rsuffix="_1").dropna()
        j = j[(j[f"eff_n_{axis}_0"] >= MIN_SEASON_N) & (j[f"eff_n_{axis}_1"] >= MIN_SEASON_N)]
        if len(j) < 50:
            out[axis] = (float("inf"), float("nan"), len(j))
            continue
        r = float(j[f"beta_{axis}_0"].corr(j[f"beta_{axis}_1"]))
        n_med = float(j[[f"eff_n_{axis}_0", f"eff_n_{axis}_1"]].stack().median())
        n0 = float("inf") if not np.isfinite(r) or r <= 0.01 else n_med * (1 - r) / r
        out[axis] = (n0, r, len(j))
    return out


def fit_model(df: pd.DataFrame, target: str) -> tuple:
    """Between-batter OLS: target_z ~ adjustability_z + swing_plus_z + repertoire_pctile_z.

    Composite `adjustability` is the treatment (not adj_count) -- it is the headline
    metric. Returns (fitted values in raw units, theta, sd of target).
    """
    cols = [target, "adjustability", "swing_plus", "repertoire_pctile"]
    d = df.dropna(subset=cols)
    if len(d) < 50:
        return pd.Series(dtype=float), float("nan"), float("nan")

    def _z(s):
        return (s - s.mean()) / s.std()

    y = _z(d[target]).to_numpy(float)
    X = np.column_stack([
        np.ones(len(d)),
        _z(d["swing_plus"]).to_numpy(float),
        _z(d["repertoire_pctile"]).to_numpy(float),
        _z(d["adjustability"]).to_numpy(float),
    ])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu, sd = d[target].mean(), d[target].std()
    idx = pd.MultiIndex.from_frame(d[KEY])
    fitted = pd.Series((X @ coefs) * sd + mu, index=idx)
    # Same prediction with the adjustability term zeroed: the situational effect
    # expected from swing quality and repertoire width alone. Used as the baseline
    # so the reported runs are net of talent, not a restatement of Swing+.
    Xc = X.copy()
    Xc[:, -1] = 0.0
    baseline = pd.Series((Xc @ coefs) * sd + mu, index=idx)
    return fitted, baseline, float(coefs[-1]), float(sd)


def exposure(sw: pd.DataFrame) -> pd.DataFrame:
    out = None
    for axis in AXES:
        n = (sw.groupby(KEY)[IND[axis]].sum() / N_SEASONS).rename(f"n_{axis}_per_season")
        out = n.to_frame() if out is None else out.join(n, how="outer")
    return out.fillna(0.0)


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
    sw = sw.merge(adj[KEY], on=KEY)
    cards = pd.read_parquet(DATA / "shape_cards.parquet",
                            columns=KEY + ["role", "archetype_name", "grade"])
    rep = pd.read_parquet(DATA / "repertoire_scores.parquet",
                          columns=KEY + ["repertoire_pctile", "effective_shapes"])

    print("Estimating joint partial effects (within pitch_type x zone)...")
    eff = joint_effects(sw)
    print("Computing year-over-year reliability...")
    stab = stabilization(sw)

    df = (adj.merge(eff.reset_index(), on=KEY, how="left")
             .merge(exposure(sw).reset_index(), on=KEY, how="left")
             .merge(rep, on=KEY, how="left"))
    idx = pd.MultiIndex.from_frame(df[KEY])

    print("\n=== Empirical-Bayes shrinkage ===")
    for axis in AXES:
        n0, r, n_units = stab[axis]
        fitted, baseline, theta, sd_b = fit_model(df, f"beta_{axis}")

        e_n = df[f"eff_n_{axis}"].to_numpy(float)
        w = np.where(np.isfinite(n0) & (e_n > 0), e_n / (e_n + n0), 0.0)
        fit_v = fitted.reindex(idx).to_numpy(float)
        obs_v = np.nan_to_num(df[f"beta_{axis}"].to_numpy(float))

        eb = w * obs_v + (1 - w) * fit_v
        df[f"beta_{axis}_eb"] = eb
        df[f"eb_weight_{axis}"] = w

        # Net of the talent baseline. delta_run_exp magnitudes scale with base state
        # and with hitter quality, so centering on the league mean would leave the
        # output ~-0.6 correlated with Swing+ (elite hitters have more to lose).
        expo = df[f"n_{axis}_per_season"].to_numpy(float)
        df[f"season_runs_{axis}"] = (eb - baseline.reindex(idx).to_numpy(float)) * expo

        adj_z = ((df["adjustability"] - df["adjustability"].mean())
                 / df["adjustability"].std()).to_numpy(float)
        df[f"adj_runs_{axis}"] = theta * adj_z * sd_b * expo

        print(f"  {axis:<10} YoY r={r:6.3f} (n={n_units})  n0={n0:9,.0f}  "
              f"mean w={w.mean():.3f}  theta={theta:+.4f}")

    df["season_runs_total"] = df[[f"season_runs_{a}" for a in AXES]].sum(axis=1)
    df["adj_runs_total"] = df[[f"adj_runs_{a}" for a in AXES]].sum(axis=1)

    print("Joining archetype...")
    primary = cards[cards["role"] == "primary"][KEY + ["archetype_name", "grade"]] \
        .rename(columns={"grade": "primary_grade"})
    df = df.merge(primary, on=KEY, how="left")

    out_cols = (
        KEY + ["label", "n_swings", "adjustability", "adj_count",
               "adjustability_plus", "adjustability_pctile", "swing_plus",
               "twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty"]
        + [f"beta_{a}" for a in AXES] + [f"beta_{a}_eb" for a in AXES]
        + [f"eb_weight_{a}" for a in AXES] + [f"n_{a}_per_season" for a in AXES]
        + [f"season_runs_{a}" for a in AXES] + ["season_runs_total"]
        + [f"adj_runs_{a}" for a in AXES] + ["adj_runs_total"]
        + ["archetype_name", "primary_grade", "repertoire_pctile", "effective_shapes"]
    )
    df = df[out_cols]

    out = DATA / "optimal_policy.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")

    print("\n=== Seasonal runs (net of talent baseline; 0 = as expected) ===")
    for c in [f"season_runs_{a}" for a in AXES] + ["season_runs_total"]:
        s = df[c]
        print(f"  {c:<22} sd={s.std():5.2f}  range=[{s.min():6.1f}, {s.max():5.1f}]")
    print(f"\n  additivity check: max |total - sum(axes)| = "
          f"{(df['season_runs_total'] - df[[f'season_runs_{a}' for a in AXES]].sum(axis=1)).abs().max():.2e}")
    t = df["adj_runs_total"]
    print(f"  adjustability-attributable total: sd={t.std():.2f}  "
          f"P90-P10={t.quantile(0.9) - t.quantile(0.1):+.2f} runs/season")

    print("\n=== Spot check ===")
    spot = df[df["label"].str.contains("Judge|Arraez", na=False)]
    print(spot[["label", "season_runs_2k", "season_runs_gamestate",
                "season_runs_platoon", "season_runs_total", "adj_runs_total",
                "adjustability"]].to_string(index=False))


if __name__ == "__main__":
    main()
