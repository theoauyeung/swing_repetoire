"""Optimal adjustment policy — counterfactual value of situational swing changes.

For every swing we ask: what would this hitter's swing have looked like on this exact
pitch if the count, base state and matchup had not moved him? Price both the actual and
the counterfactual swing through xRV, difference, and sum over the season.

    value = sum over swings of [ xRV(actual swing) - xRV(counterfactual swing) ]

On top of that accounting sit two prescriptive layers. The alpha scan asks HOW MUCH to
modulate, capped at the level a 90th-percentile league modulator sustains. The per-dial
gradient asks WHICH LEVER: for each situation cell it reports what a +1 SD move on each
shape dial is worth, beside how far the hitter already moves it there.

Output: data/adjustability_value.parquet     (unit level)
        data/adjustability_gradients.parquet (unit x axis x cell x dial)
Run   : python src/adjustability_value.py            (full, ~10 min)
        python src/adjustability_value.py --verify   (scoring correctness check only)
        python src/adjustability_value.py --limit 25 (fast subset for iteration)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import counterfactual as cf                                          # noqa: E402
import xRV_model as xrv                                              # noqa: E402
from adjustability import KEY, MIN_SWINGS, SEASONS                   # noqa: E402
from adjustability import add_context, location_design               # noqa: E402

SHAPE = xrv.SHAPE_FEATURES
POLICY_REF = ROOT / "src" / "adjustability_policy_reference.json"
LOAD_COLS = KEY + [
    "play_id", "game_year", "batter_full_name", "balls", "strikes", "outs_when_up",
    "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type", "pitcher_throws",
    "on_1b_id", "on_2b_id", "on_3b_id", "delta_run_exp", "woba",
] + SHAPE


def load_models():
    """The three persisted boosters from src/xRV_model.py."""
    specs = [("p_bip", XGBClassifier), ("p_foul", XGBClassifier), ("v_bip", XGBRegressor)]
    models = {}
    for name, cls in specs:
        model = cls(enable_categorical=True)
        model.load_model(DATA / "xrv_models" / f"{name}.json")
        models[name] = model
    return models


def load_swings():
    """2024-25 swings for qualifying units, with pitch_type categories fixed league-wide.

    The cast happens once on the full frame. build_features calls astype("category"),
    which infers categories from whatever it is handed — scoring per-unit frames would
    give each unit its own category codes and silently misalign every prediction.
    """
    df = pd.read_parquet(DATA / "swings_model.parquet", columns=LOAD_COLS)
    df = df[df["game_year"].isin(SEASONS)]
    df["pitch_type"] = df["pitch_type"].astype("category")
    df = add_context(df).dropna(subset=SHAPE)
    sizes = df.groupby(KEY, observed=True)[SHAPE[0]].transform("size")
    return df[sizes >= MIN_SWINGS].reset_index(drop=True)


def score_shapes(models, run_value_tables, group, shape_matrix):
    """Per-swing xRV for `group`'s pitches with its shape columns overwritten.

    assemble_xrv (not the neutral variant) so two-strike stakes stay live: the mechanism
    under test is whether compression buys enough contact to pay for the strikeout risk.
    """
    frame = group.copy()
    for j, feature in enumerate(SHAPE):
        frame[feature] = shape_matrix[:, j]
    enriched = xrv.build_features(frame)
    prob_bip  = models["p_bip"].predict_proba(enriched[xrv.FEATURES])[:, 1]
    prob_foul = models["p_foul"].predict_proba(enriched[xrv.FEATURES])[:, 1]
    value_bip = models["v_bip"].predict(enriched[xrv.FEATURES])
    return xrv.assemble_xrv(enriched, prob_bip, prob_foul, value_bip, run_value_tables)


def shape_gradients(models, run_value_tables, group, shape_matrix, scale):
    """Per-swing d(xRV)/d(dial), in runs per +1 league SD, by central difference.

    Probed at the OBSERVED shapes: the advice is about the swings the hitter actually
    takes, and that is where xRV is best supported. The +/- 0.1 SD step keeps the probe
    inside the shape cloud, so this reads a local slope rather than extrapolating.
    """
    out = np.empty_like(shape_matrix)
    for j in range(shape_matrix.shape[1]):
        up, down = shape_matrix.copy(), shape_matrix.copy()
        up[:, j] += cf.GRAD_STEP_SD * scale[j]
        down[:, j] -= cf.GRAD_STEP_SD * scale[j]
        out[:, j] = (score_shapes(models, run_value_tables, group, up)
                     - score_shapes(models, run_value_tables, group, down)) / (2 * cf.GRAD_STEP_SD)
    return out


def prescriptions(group, gradients, shifts, cells):
    """One long row per (situation cell, dial): what the dial pays, and what he already does.

    `grad_runs_per_sd` is what a +1 SD move on that dial buys per swing in this cell.
    `shift_sd` is how far his fitted swing already moves it there. The prescription is the
    gap between them. Both are signed — negative means move the dial down.

    `grad_delta_runs_per_sd` subtracts the hitter's own all-swing gradient, which is what
    makes the advice situational. The raw cell gradient is dominated by his baseline, so
    ranking on it would just surface the largest cell for every hitter; the delta says
    where THIS situation differs from how he normally hits.
    """
    overall = gradients.mean(axis=0)
    rows = []
    for axis, labels in cells.items():
        for cell in pd.unique(labels):
            mask = labels == cell
            n_cell = int(mask.sum())
            if n_cell < cf.MIN_CELL_SWINGS:
                continue
            for j, dial in enumerate(SHAPE):
                grad = float(gradients[mask, j].mean())
                delta = grad - float(overall[j])
                shift = float(shifts[mask, j].mean())
                season_swings = n_cell / N_SEASONS
                rows.append({
                    "batter_id": group["batter_id"].iloc[0],
                    "batter_stand": group["batter_stand"].iloc[0],
                    "axis": axis,
                    "cell": cell,
                    "dial": dial,
                    "n_swings": n_cell,
                    "grad_runs_per_sd": grad,
                    "grad_delta_runs_per_sd": delta,
                    "shift_sd": shift,
                    "runs_from_shift": grad * shift * season_swings,
                    "runs_per_quarter_sd": grad * cf.POLICY_STEP_SD * season_swings,
                    "situational_runs_per_quarter_sd": delta * cf.POLICY_STEP_SD * season_swings,
                })
    return rows


def load_policy_reference(swings):
    """Frozen league scale and displacement cap, or a fresh scale if not yet pegged.

    Follows the repertoire_reference.json precedent so prescriptions stay comparable as
    seasons are added. League aggregates only, no PII. Delete the file to re-peg.
    """
    if POLICY_REF.exists():
        return json.loads(POLICY_REF.read_text())
    return {
        "seasons": SEASONS,
        "shape_features": SHAPE,
        "shape_sd": swings[SHAPE].std(ddof=0).astype(float).to_dict(),
        "policy_quantile": cf.POLICY_QUANTILE,
        "displacement_cap": None,
    }


def verify_scoring(models, run_value_tables, swings, n_units=5):
    """Score OBSERVED shapes and compare to the published xrv column.

    This is the category-alignment guard. If pitch_type codes were misaligned the
    deviation would be large and obvious.
    """
    units = list(dict.fromkeys(zip(swings["batter_id"], swings["batter_stand"])))[:n_units]
    published = pd.read_parquet(DATA / "xrv_swings.parquet", columns=["play_id", "xrv"])
    worst = 0.0
    for batter_id, stand in units:
        group = swings[(swings["batter_id"] == batter_id)
                       & (swings["batter_stand"] == stand)]
        got = score_shapes(models, run_value_tables, group, group[SHAPE].to_numpy(float))
        merged = (pd.DataFrame({"play_id": group["play_id"].to_numpy(), "got": got})
                  .merge(published, on="play_id", how="inner"))
        if merged.empty:
            raise AssertionError(f"no play_id overlap for {batter_id} {stand}")
        worst = max(worst, float((merged["got"] - merged["xrv"]).abs().max()))
    return worst


N_SEASONS = len(SEASONS)
AXIS_NAMES = list(cf.AXES)


def unit_record(models, run_value_tables, group, scale=None,
                headline_only=False, shuffle_seed=None):
    """One unit's counterfactual run accounting, alpha scan and per-dial prescriptions.

    Returns the record plus two private keys the caller must pop: `_alpha_curve` (needed
    only after the cohort-wide displacement cap is known) and `_prescriptions` (long rows
    for the gradient parquet).

    headline_only stops after the two-arm headline, skipping everything that needs extra
    scoring passes. The split-half and placebo checks need only `runs_total`.

    shuffle_seed permutes the situation columns within the unit, destroying any real
    situation-shape relationship while preserving marginals — the placebo.
    """
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        situation_cols = [c for cols in cf.AXES.values() for c in cols]
        group = group.copy()
        group[situation_cols] = group[situation_cols].to_numpy()[rng.permutation(len(group))]

    location = location_design(group)
    design, axis_slices = cf.build_design(group, location)
    observed = group[SHAPE].to_numpy(float)

    fits = cf.crossfit_shapes(design, observed)
    shape_actual = cf.predict_oof(design, fits, len(SHAPE))
    design_cf = cf.desituate(design, axis_slices, AXIS_NAMES)
    shape_cf = cf.predict_oof(design_cf, fits, len(SHAPE))

    def season_runs(shape_matrix, baseline):
        return float((score_shapes(models, run_value_tables, group, shape_matrix)
                      - baseline).sum() / N_SEASONS)

    xrv_cf = score_shapes(models, run_value_tables, group, shape_cf)
    xrv_actual = score_shapes(models, run_value_tables, group, shape_actual)

    per_swing_delta = xrv_actual - xrv_cf
    is_two_strike = group["strikes"].to_numpy() == 2

    record = {
        "batter_id":    group["batter_id"].iloc[0],
        "batter_stand": group["batter_stand"].iloc[0],
        "n_swings":     len(group),
        "runs_total":     float(per_swing_delta.sum() / N_SEASONS),
        "runs_total_2k":  float(per_swing_delta[is_two_strike].sum() / N_SEASONS),
        "runs_per_swing": float(per_swing_delta.mean()),
        "xrv_actual_mean": float(xrv_actual.mean()),
        "xrv_cf_mean":     float(xrv_cf.mean()),
    }
    if headline_only:
        return record

    scale = np.asarray([scale[f] for f in SHAPE], dtype=float)

    axis_shapes = {}
    axis_sum = 0.0
    for axis in AXIS_NAMES:
        others = [a for a in AXIS_NAMES if a != axis]
        design_one = cf.desituate(design, axis_slices, others)
        shape_one = cf.predict_oof(design_one, fits, len(SHAPE))
        axis_shapes[axis] = shape_one
        value = season_runs(shape_one, xrv_cf)
        record[f"runs_{axis}"] = value
        axis_sum += value
    record["runs_interaction"] = record["runs_total"] - axis_sum

    # The observed-percentile envelope stays as a hard rail. It is no longer the binding
    # constraint — it is built from observed shapes, whose execution noise (~1.5 SD) dwarfs
    # the situational displacement (~0.37 SD) being blended, so it almost never rejects.
    env = cf.envelope(observed)
    admissible = cf.admissible_alphas(shape_cf, shape_actual, env)
    curve = {a: season_runs(cf.blend(shape_cf, shape_actual, a), xrv_cf) for a in admissible}
    peak = max(curve, key=curve.get) if curve else float("nan")
    record["alpha_peak_unconstrained"] = peak
    record["alpha_peak_at_boundary"] = bool(
        admissible and peak in (min(admissible), max(admissible))
    )
    record["displacement_sd"] = cf.mean_displacement(shape_actual, shape_cf, scale)

    for axis in AXIS_NAMES:
        axis_curve = {}
        for a in cf.AXIS_ALPHA_GRID:
            shape_a = cf.axis_blend(shape_actual, axis_shapes[axis], shape_cf, a)
            if cf.fraction_outside(shape_a, env) < cf.MAX_OUTSIDE:
                axis_curve[a] = season_runs(shape_a, xrv_cf)
        record[f"alpha_star_{axis}"] = (max(axis_curve, key=axis_curve.get)
                                        if axis_curve else float("nan"))
        # Flatness detector: a near-zero range means the argmax is noise, not a policy.
        record[f"runs_range_{axis}"] = (max(axis_curve.values()) - min(axis_curve.values())
                                        if axis_curve else float("nan"))

    lo = season_runs(cf.blend(shape_cf, shape_actual, 0.9), xrv_cf)
    hi = season_runs(cf.blend(shape_cf, shape_actual, 1.1), xrv_cf)
    record["marginal_runs_per_alpha"] = (hi - lo) / 0.2

    gradients = shape_gradients(models, run_value_tables, group, observed, scale)
    record["_alpha_curve"] = curve
    record["_prescriptions"] = prescriptions(
        group, gradients, (shape_actual - shape_cf) / scale, cf.cell_labels(group))
    return record


def axis_top_levers(gradients):
    """Per unit, the most situational dial within EACH axis, and what moving it is worth.

    One roll-up per axis rather than a single overall argmax: the count axis carries an
    order of magnitude more situational gradient than the other two, so an overall winner
    would read `count` for essentially every hitter and hide the gamestate and platoon
    prescriptions entirely. Values stay signed — negative means move the dial down.
    """
    ranked = gradients.assign(gain=gradients["situational_runs_per_quarter_sd"].abs())
    out = None
    for axis, block in ranked.groupby("axis"):
        best = block.loc[block.groupby(KEY)["gain"].idxmax()]
        levers = best[KEY].copy()
        levers[f"top_lever_{axis}"] = (best["cell"] + "/" + best["dial"]).to_numpy()
        levers[f"top_lever_{axis}_runs"] = best["situational_runs_per_quarter_sd"].to_numpy()
        out = levers if out is None else out.merge(levers, on=KEY, how="outer")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    models = load_models()
    run_value_tables = xrv.load_run_value_tables()
    swings = load_swings()
    print(f"{len(swings):,} swings, "
          f"{swings.groupby(KEY, observed=True).ngroups} units")

    if args.verify:
        worst = verify_scoring(models, run_value_tables, swings)
        print(f"max |scored - published xrv| = {worst:.3e}")
        print("PASS" if worst < 1e-5 else "FAIL — pitch_type categories misaligned")
        return

    reference = load_policy_reference(swings)
    groups = list(swings.groupby(KEY, observed=True, sort=False))
    if args.limit:
        groups = groups[:args.limit]

    records, curves, gradient_rows = [], [], []
    for i, (_, group) in enumerate(groups, 1):
        record = unit_record(models, run_value_tables, group.reset_index(drop=True),
                             scale=reference["shape_sd"])
        curves.append(record.pop("_alpha_curve"))
        gradient_rows.extend(record.pop("_prescriptions"))
        records.append(record)
        if i % 25 == 0:
            print(f"  {i}/{len(groups)} units")
    df = pd.DataFrame(records)

    # Admit only levels of situational modulation major-league hitters actually sustain.
    if reference["displacement_cap"] is None:
        reference["displacement_cap"] = float(
            df["displacement_sd"].quantile(cf.POLICY_QUANTILE))
    cap = reference["displacement_cap"]

    policy, gain = [], []
    for d_u, curve in zip(df["displacement_sd"], curves):
        allowed = {a: v for a, v in curve.items() if a in cf.policy_alphas(d_u, cap)}
        best = max(allowed, key=allowed.get) if allowed else float("nan")
        policy.append(best)
        gain.append(allowed[best] - curve[1.0]
                    if allowed and 1.0 in curve else float("nan"))
    df["alpha_star_policy"] = policy
    df["runs_at_alpha_star"] = gain
    df["runs_at_alpha_star_per_swing"] = df["runs_at_alpha_star"] * N_SEASONS / df["n_swings"]

    adj = pd.read_parquet(DATA / "adjustability.parquet", columns=KEY + [
        "label", "adjustability", "adj_count", "adjustability_plus",
        "adjustability_pctile", "swing_plus",
        "twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty",
    ])
    cards = pd.read_parquet(DATA / "shape_cards.parquet",
                            columns=KEY + ["role", "archetype_name", "grade"])
    primary = (cards[cards["role"] == "primary"][KEY + ["archetype_name", "grade"]]
               .rename(columns={"grade": "primary_grade"}))
    rep = pd.read_parquet(DATA / "repertoire_scores.parquet",
                          columns=KEY + ["repertoire_pctile", "effective_shapes"])

    df = (df.merge(adj, on=KEY, how="left")
            .merge(primary, on=KEY, how="left")
            .merge(rep, on=KEY, how="left"))

    suffix = "_subset" if args.limit else ""
    gradients = pd.DataFrame(gradient_rows).merge(df[KEY + ["label"]], on=KEY, how="left")
    df = df.merge(axis_top_levers(gradients), on=KEY, how="left")

    out = DATA / f"adjustability_value{suffix}.parquet"
    grad_out = DATA / f"adjustability_gradients{suffix}.parquet"
    df.to_parquet(out, index=False)
    gradients.to_parquet(grad_out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")
    print(f"Wrote {len(gradients)} rows -> {grad_out}")

    if not args.limit and not POLICY_REF.exists():
        POLICY_REF.write_text(json.dumps(reference, indent=2) + "\n")
        print(f"Pegged policy reference -> {POLICY_REF}")

    print("\n=== Season runs from situational adjustment ===")
    for col in ["runs_total", "runs_count", "runs_gamestate", "runs_platoon",
                "runs_interaction", "runs_total_2k"]:
        s = df[col]
        print(f"  {col:<18} mean={s.mean():+6.2f}  median={s.median():+6.2f}  "
              f"sd={s.std():5.2f}  range=[{s.min():+6.1f}, {s.max():+6.1f}]")
    resid = df["runs_interaction"].abs()
    print(f"\n  |interaction| mean={resid.mean():.2f}  max={resid.max():.2f}  "
          f"median % of |total|={100 * (resid / df['runs_total'].abs().clip(lower=0.1)).median():.0f}%")

    print("\n=== Policy ===")
    print(f"  displacement cap Q{100 * cf.POLICY_QUANTILE:.0f} = {cap:.3f} SD")
    for col in ["displacement_sd", "alpha_peak_unconstrained", "alpha_star_policy",
                "runs_at_alpha_star", "runs_at_alpha_star_per_swing"]:
        s = df[col]
        print(f"  {col:<28} mean={s.mean():+7.3f}  median={s.median():+7.3f}")
    print(f"  peak still at a grid edge: {100 * df['alpha_peak_at_boundary'].mean():.0f}% of units")
    for axis in AXIS_NAMES:
        print(f"  alpha_star_{axis:<10} median={df[f'alpha_star_{axis}'].median():.2f}  "
              f"runs range median={df[f'runs_range_{axis}'].median():.3f}")

    print("\n=== Situational gradient (cohort mean |cell - own baseline|, runs/swing per SD) ===")
    pivot = (gradients.assign(g=gradients["grad_delta_runs_per_sd"].abs())
             .pivot_table(index="dial", columns="cell", values="g", aggfunc="mean"))
    print(pivot.round(4).to_string())

    print("\n=== Spot check ===")
    spot = df[df["label"].str.contains("Judge|Arraez|Schwarber|Teoscar", na=False)]
    print(spot[["label", "runs_total", "alpha_star_policy", "runs_at_alpha_star"]
               + [f"top_lever_{a}" for a in AXIS_NAMES]].to_string(index=False))


if __name__ == "__main__":
    main()
