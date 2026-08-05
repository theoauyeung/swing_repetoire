"""Scores whether a hitter's situational adjustments are actually worth runs. On every pitch
he saw, we swap in the way a randomly picked other hitter would have adjusted, re-grade both
swings, and add up the difference over a season.

    runs_total = sum over swings of [ xRV(his adjustment) - xRV(a replacement hitter's) ]

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
sys.path.insert(0, str(Path(__file__).resolve().parent / "uncommited"))

import counterfactual as cf                                          # noqa: E402
import db                                                            # noqa: E402
import pitch_controls as pc                                          # noqa: E402
import xRV_model as xrv                                              # noqa: E402
from adjustability import KEY, MIN_SWINGS, SEASONS                   # noqa: E402
from adjustability import add_context, location_design               # noqa: E402

SHAPE = xrv.SHAPE_FEATURES
N_SEASONS = len(SEASONS)
AXIS_NAMES = list(cf.AXES)
POLICY_REF = ROOT / "src" / "uncommited" / "adjustability_policy_reference.json"
LOAD_COLS = KEY + [
    "play_id", "game_pk", "game_year", "batter_full_name", "balls", "strikes", "outs_when_up",
    "plate_x", "plate_z", "sz_top", "sz_bot", "pitch_type", "pitcher_throws",
    "on_1b_id", "on_2b_id", "on_3b_id", "delta_run_exp", "woba",
] + SHAPE


def compute_swing_plus() -> pd.DataFrame:
    """Per-unit mean xrv_grade_neutral over 2024-25. The count-neutral grade is used so
    swing_plus is not confounded by count distribution when it controls for hitter quality
    alongside count-stratified quantities."""
    seasons_sql = "(" + ", ".join(str(season) for season in SEASONS) + ")"
    connection = db.connect("xrv_swings")
    swing_plus_table = connection.sql(f"""
        SELECT batter_id, batter_stand, AVG(xrv_grade_neutral) AS swing_plus
        FROM xrv_swings
        WHERE game_year IN {seasons_sql}
        GROUP BY batter_id, batter_stand
    """).df()
    connection.close()
    return swing_plus_table


def load_models():
    """The three persisted boosters from src/xRV_model.py."""
    model_specs = [("p_bip", XGBClassifier), ("p_foul", XGBClassifier), ("v_bip", XGBRegressor)]
    models = {}
    for name, model_class in model_specs:
        model = model_class(enable_categorical=True)
        model.load_model(DATA / "xrv_models" / f"{name}.json")
        models[name] = model
    return models


def load_swings(with_pitch_chars=True):
    """2024-25 swings for qualifying units, with pitch_type categories fixed league-wide.

    The cast happens once on the full frame. build_features calls astype("category"),
    which infers categories from whatever it is handed — scoring per-unit frames would
    give each unit its own category codes and silently misalign every prediction.
    """
    swings = pd.read_parquet(DATA / "swings_model.parquet", columns=LOAD_COLS)
    swings = swings[swings["game_year"].isin(SEASONS)]
    swings["pitch_type"] = swings["pitch_type"].astype("category")
    swings = add_context(swings).dropna(subset=SHAPE)
    if with_pitch_chars:
        swings = pc.join_pitch_chars(swings)
    swings_per_unit = swings.groupby(KEY, observed=True)[SHAPE[0]].transform("size")
    return swings[swings_per_unit >= MIN_SWINGS].reset_index(drop=True)


def score_shapes(models, run_value_tables, group, shape_matrix):
    """Per-swing xRV for `group`'s pitches with its shape columns overwritten.

    assemble_xrv (not the neutral variant) so two-strike stakes stay live: the mechanism
    under test is whether compression buys enough contact to pay for the strikeout risk.
    """
    frame = group.copy()
    for dial_index, feature in enumerate(SHAPE):
        frame[feature] = shape_matrix[:, dial_index]
    enriched = xrv.build_features(frame)
    prob_bip  = models["p_bip"].predict_proba(enriched[xrv.FEATURES])[:, 1]
    prob_foul = models["p_foul"].predict_proba(enriched[xrv.FEATURES])[:, 1]
    value_bip = models["v_bip"].predict(enriched[xrv.FEATURES])
    return xrv.assemble_xrv(enriched, prob_bip, prob_foul, value_bip, run_value_tables)


def _score_displacement(models, run_value_tables, group, shape_cf, displacement_sd, scale_vector):
    """xRV per swing when a displacement in league-SD units is added to the de-situated shape."""
    shifted_shape = shape_cf + displacement_sd * scale_vector
    return score_shapes(models, run_value_tables, group, shifted_shape)


def _season_runs(models, run_value_tables, group, shape_matrix, baseline_xrv):
    """Season runs `shape_matrix` gains over an already-scored baseline, on the same pitches."""
    scored = score_shapes(models, run_value_tables, group, shape_matrix)
    return float((scored - baseline_xrv).sum() / N_SEASONS)


def shape_gradients(models, run_value_tables, group, shape_matrix, scale):
    """Per-swing d(xRV)/d(dial), in runs per +1 league SD, by central difference.

    Probed at the OBSERVED shapes: the advice is about the swings the hitter actually
    takes, and that is where xRV is best supported. The +/- 0.1 SD step keeps the probe
    inside the shape cloud, so this reads a local slope rather than extrapolating.
    """
    gradients = np.empty_like(shape_matrix)
    for dial_index in range(shape_matrix.shape[1]):
        shape_up, shape_down = shape_matrix.copy(), shape_matrix.copy()
        shape_up[:, dial_index]   += cf.GRAD_STEP_SD * scale[dial_index]
        shape_down[:, dial_index] -= cf.GRAD_STEP_SD * scale[dial_index]
        xrv_up   = score_shapes(models, run_value_tables, group, shape_up)
        xrv_down = score_shapes(models, run_value_tables, group, shape_down)
        gradients[:, dial_index] = (xrv_up - xrv_down) / (2 * cf.GRAD_STEP_SD)
    return gradients


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
    baseline_gradient = gradients.mean(axis=0)
    rows = []
    for axis, labels in cells.items():
        for cell in pd.unique(labels):
            in_cell = labels == cell
            n_cell_swings = int(in_cell.sum())
            if n_cell_swings < cf.MIN_CELL_SWINGS:
                continue
            season_swings = n_cell_swings / N_SEASONS
            for dial_index, dial in enumerate(SHAPE):
                cell_gradient = float(gradients[in_cell, dial_index].mean())
                situational_gradient = cell_gradient - float(baseline_gradient[dial_index])
                fitted_shift = float(shifts[in_cell, dial_index].mean())
                rows.append({
                    "batter_id": group["batter_id"].iloc[0],
                    "batter_stand": group["batter_stand"].iloc[0],
                    "axis": axis,
                    "cell": cell,
                    "dial": dial,
                    "n_swings": n_cell_swings,
                    "grad_runs_per_sd": cell_gradient,
                    "grad_delta_runs_per_sd": situational_gradient,
                    "shift_sd": fitted_shift,
                    "runs_from_shift": cell_gradient * fitted_shift * season_swings,
                    "runs_per_quarter_sd": cell_gradient * cf.POLICY_STEP_SD * season_swings,
                    "situational_runs_per_quarter_sd":
                        situational_gradient * cf.POLICY_STEP_SD * season_swings,
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
    all_units = dict.fromkeys(zip(swings["batter_id"], swings["batter_stand"]))
    units = list(all_units)[:n_units]
    published = pd.read_parquet(DATA / "xrv_swings.parquet", columns=["play_id", "xrv"])
    worst_deviation = 0.0
    for batter_id, stand in units:
        group = swings[(swings["batter_id"] == batter_id)
                       & (swings["batter_stand"] == stand)]
        scored = score_shapes(models, run_value_tables, group, group[SHAPE].to_numpy(float))
        rescored = pd.DataFrame({"play_id": group["play_id"].to_numpy(), "got": scored})
        merged = rescored.merge(published, on="play_id", how="inner")
        if merged.empty:
            raise AssertionError(f"no play_id overlap for {batter_id} {stand}")
        deviation = float((merged["got"] - merged["xrv"]).abs().max())
        worst_deviation = max(worst_deviation, deviation)
    return worst_deviation


def unit_fit(group, scale, shuffle_seed=None):
    """Pass 1: this unit's cross-fitted shapes and its situation->shape policy block.

    The policy block maps a CENTERED situation row to a displacement in league-SD units,
    so it is conformable and comparable across hitters with different dial ranges — that
    is what makes it transferable to another hitter's situations in pass 2.

    shuffle_seed permutes the situation columns within the unit, destroying any real
    situation-shape relationship while preserving marginals — the placebo.
    """
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        group = group.copy()
        shuffled_rows = group[cf.SITUATION_COLS].to_numpy()[rng.permutation(len(group))]
        group[cf.SITUATION_COLS] = shuffled_rows

    design, axis_slices = cf.build_design(group, location_design(group),
                                          pc.control_matrix(group))
    observed = group[SHAPE].to_numpy(float)
    fits = cf.crossfit_shapes(design, observed)
    shape_actual = cf.predict_oof(design, fits, len(SHAPE))
    design_cf = cf.desituate(design, axis_slices, AXIS_NAMES)
    shape_cf = cf.predict_oof(design_cf, fits, len(SHAPE))
    centered, axis_rows = cf.centered_situation(group)
    displacement = (shape_actual - shape_cf) / scale
    return {
        "group": group, "design": design, "axis_slices": axis_slices, "fits": fits,
        "observed": observed, "shape_actual": shape_actual, "shape_cf": shape_cf,
        "displacement": displacement, "centered": centered, "axis_rows": axis_rows,
        "policy": cf.fit_policy(centered, displacement),
    }


def unit_record(models, run_value_tables, fit, scale, blocks=None, index=None,
                n_replacements=cf.N_REPLACEMENTS, headline_only=False):
    """Pass 2: run accounting against the replacement benchmark, alpha scan, per-dial gradients.

    Returns the record plus two private keys the caller must pop: `_alpha_curve` (needed
    only after the cohort-wide displacement cap is known) and `_prescriptions` (long rows
    for the gradient parquet).

    headline_only skips everything needing extra scoring passes; the split-half and
    placebo checks need only the headline arms.
    """
    group, shape_cf = fit["group"], fit["shape_cf"]
    shape_actual, observed = fit["shape_actual"], fit["observed"]
    centered, displacement = fit["centered"], fit["displacement"]
    scale_vector = np.asarray([scale[feature] for feature in SHAPE], dtype=float)

    xrv_cf = score_shapes(models, run_value_tables, group, shape_cf)
    xrv_actual = score_shapes(models, run_value_tables, group, shape_actual)
    is_two_strike = group["strikes"].to_numpy() == 2

    record = {
        "batter_id":    group["batter_id"].iloc[0],
        "batter_stand": group["batter_stand"].iloc[0],
        "n_swings":     len(group),
        "runs_vs_desituated": float((xrv_actual - xrv_cf).sum() / N_SEASONS),
        "displacement_sd":    float(np.linalg.norm(displacement, axis=1).mean()),
        "xrv_actual_mean":    float(xrv_actual.mean()),
        "xrv_cf_mean":        float(xrv_cf.mean()),
    }

    # Replacement arms. Each draw substitutes a randomly chosen other hitter's policy block —
    # wholly for the headline, one axis at a time for the decomposition — and re-scores
    # the same pitches. Averaging over draws is what makes the benchmark "a typical
    # individual" rather than one arbitrary hitter.
    if blocks is not None and n_replacements:
        rng = np.random.default_rng(cf.SEED + index)
        other_units = [j for j in range(len(blocks)) if j != index]
        n_draws = min(n_replacements, len(blocks) - 1)
        replacement_indices = rng.choice(other_units, size=n_draws, replace=False)

        replacement_xrv = np.zeros(len(group))
        replacement_xrv_by_axis = {axis: np.zeros(len(group)) for axis in AXIS_NAMES}
        for j in replacement_indices:
            replacement_xrv += _score_displacement(
                models, run_value_tables, group, shape_cf, centered @ blocks[j], scale_vector)
            for axis in AXIS_NAMES:
                swapped = cf.swap_axis(fit["policy"], blocks[j], fit["axis_rows"], axis)
                replacement_xrv_by_axis[axis] += _score_displacement(
                    models, run_value_tables, group, shape_cf, centered @ swapped, scale_vector)

        n_picks = len(replacement_indices)
        per_swing_delta = xrv_actual - replacement_xrv / n_picks
        record["n_replacements"] = n_picks
        record["runs_total"] = float(per_swing_delta.sum() / N_SEASONS)
        record["runs_total_2k"] = float(per_swing_delta[is_two_strike].sum() / N_SEASONS)
        record["runs_per_swing"] = float(per_swing_delta.mean())
        axis_sum = 0.0
        for axis in AXIS_NAMES:
            axis_delta = xrv_actual - replacement_xrv_by_axis[axis] / n_picks
            axis_runs = float(axis_delta.sum() / N_SEASONS)
            record[f"runs_{axis}"] = axis_runs
            axis_sum += axis_runs
        record["runs_interaction"] = record["runs_total"] - axis_sum

    if headline_only:
        return record

    # Each axis on its own: de-situate the other two and leave this one live.
    axis_shapes = {}
    for axis in AXIS_NAMES:
        other_axes = [a for a in AXIS_NAMES if a != axis]
        design_one_axis = cf.desituate(fit["design"], fit["axis_slices"], other_axes)
        axis_shapes[axis] = cf.predict_oof(design_one_axis, fit["fits"], len(SHAPE))

    # The observed-percentile envelope stays as a hard rail. It is no longer the binding
    # constraint — it is built from observed shapes, whose execution noise (~1.5 SD) dwarfs
    # the situational displacement (~0.37 SD) being blended, so it almost never rejects.
    shape_envelope = cf.envelope(observed)
    admissible = cf.admissible_alphas(shape_cf, shape_actual, shape_envelope)
    alpha_curve = {}
    for alpha in admissible:
        blended_shape = cf.blend(shape_cf, shape_actual, alpha)
        alpha_curve[alpha] = _season_runs(
            models, run_value_tables, group, blended_shape, xrv_cf)
    peak_alpha = max(alpha_curve, key=alpha_curve.get) if alpha_curve else float("nan")
    record["alpha_peak_unconstrained"] = peak_alpha
    record["alpha_peak_at_boundary"] = bool(
        admissible and peak_alpha in (min(admissible), max(admissible))
    )

    for axis in AXIS_NAMES:
        axis_alpha_curve = {}
        for alpha in cf.AXIS_ALPHA_GRID:
            blended_shape = cf.axis_blend(shape_actual, axis_shapes[axis], shape_cf, alpha)
            if cf.fraction_outside(blended_shape, shape_envelope) < cf.MAX_OUTSIDE:
                axis_alpha_curve[alpha] = _season_runs(
                    models, run_value_tables, group, blended_shape, xrv_cf)
        if axis_alpha_curve:
            record[f"alpha_star_{axis}"] = max(axis_alpha_curve, key=axis_alpha_curve.get)
            # Flatness detector: a near-zero range means the argmax is noise, not a policy.
            record[f"runs_range_{axis}"] = (max(axis_alpha_curve.values())
                                            - min(axis_alpha_curve.values()))
        else:
            record[f"alpha_star_{axis}"] = float("nan")
            record[f"runs_range_{axis}"] = float("nan")

    # Slope of the value curve at his current level of modulation (alpha = 1).
    runs_at_alpha_090 = _season_runs(models, run_value_tables, group,
                                     cf.blend(shape_cf, shape_actual, 0.9), xrv_cf)
    runs_at_alpha_110 = _season_runs(models, run_value_tables, group,
                                     cf.blend(shape_cf, shape_actual, 1.1), xrv_cf)
    record["marginal_runs_per_alpha"] = (runs_at_alpha_110 - runs_at_alpha_090) / 0.2

    gradients = shape_gradients(models, run_value_tables, group, observed, scale_vector)
    situational_shift = (shape_actual - shape_cf) / scale_vector
    record["_alpha_curve"] = alpha_curve
    record["_prescriptions"] = prescriptions(
        group, gradients, situational_shift, cf.cell_labels(group))
    return record


def axis_top_levers(gradients):
    """Per unit, the most situational dial within EACH axis, and what moving it is worth.

    One roll-up per axis rather than a single overall argmax: the count axis carries an
    order of magnitude more situational gradient than the other two, so an overall winner
    would read `count` for essentially every hitter and hide the gamestate and platoon
    prescriptions entirely. Values stay signed — negative means move the dial down.
    """
    ranked = gradients.assign(gain=gradients["situational_runs_per_quarter_sd"].abs())
    top_levers = None
    for axis, axis_block in ranked.groupby("axis"):
        best = axis_block.loc[axis_block.groupby(KEY)["gain"].idxmax()]
        axis_levers = best[KEY].copy()
        axis_levers[f"top_lever_{axis}"] = (best["cell"] + "/" + best["dial"]).to_numpy()
        axis_levers[f"top_lever_{axis}_runs"] = best["situational_runs_per_quarter_sd"].to_numpy()
        top_levers = (axis_levers if top_levers is None
                      else top_levers.merge(axis_levers, on=KEY, how="outer"))
    return top_levers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeg", action="store_true",
                        help="recompute the displacement cap and rewrite the reference")
    args = parser.parse_args()

    models = load_models()
    run_value_tables = xrv.load_run_value_tables()
    swings = load_swings()
    print(f"{len(swings):,} swings, "
          f"{swings.groupby(KEY, observed=True).ngroups} units")

    if args.verify:
        worst_deviation = verify_scoring(models, run_value_tables, swings)
        print(f"max |scored - published xrv| = {worst_deviation:.3e}")
        print("PASS" if worst_deviation < 1e-5 else "FAIL — pitch_type categories misaligned")
        return

    reference = load_policy_reference(swings)
    scale = reference["shape_sd"]
    scale_vector = np.asarray([scale[feature] for feature in SHAPE], float)
    groups = list(swings.groupby(KEY, observed=True, sort=False))
    if args.limit:
        groups = groups[:args.limit]

    print(f"pass 1: fitting {len(groups)} unit policies")
    fitted = []
    for _, group in groups:
        fitted.append(unit_fit(group.reset_index(drop=True), scale_vector))
    blocks = [fit["policy"] for fit in fitted]

    print(f"pass 2: scoring arms ({cf.N_REPLACEMENTS} replacements per unit)")
    records, alpha_curves, gradient_rows = [], [], []
    for i, fit in enumerate(fitted):
        record = unit_record(models, run_value_tables, fit, scale, blocks, i)
        alpha_curves.append(record.pop("_alpha_curve"))
        gradient_rows.extend(record.pop("_prescriptions"))
        records.append(record)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(fitted)} units")
    unit_table = pd.DataFrame(records)

    # Admit only levels of situational modulation major-league hitters actually sustain.
    # The cap is a quantile of FITTED displacement, so it is tied to the shape design and
    # must be re-pegged whenever that changes (adding the pitch-characteristic controls
    # shrank displacements, which would leave an old cap non-binding). `shape_sd` is a raw
    # league scale and stays frozen across re-pegs.
    if args.repeg:
        reference["displacement_cap"] = None
    if reference["displacement_cap"] is None:
        reference["displacement_cap"] = float(
            unit_table["displacement_sd"].quantile(cf.POLICY_QUANTILE))
    displacement_cap = reference["displacement_cap"]

    alpha_star_policy, runs_at_alpha_star = [], []
    for unit_displacement_sd, curve in zip(unit_table["displacement_sd"], alpha_curves):
        capped_alphas = cf.policy_alphas(unit_displacement_sd, displacement_cap)
        allowed = {alpha: runs for alpha, runs in curve.items() if alpha in capped_alphas}
        best_alpha = max(allowed, key=allowed.get) if allowed else float("nan")
        alpha_star_policy.append(best_alpha)
        # Gain is measured against alpha = 1.0, what he already does.
        runs_at_alpha_star.append(allowed[best_alpha] - curve[1.0]
                                  if allowed and 1.0 in curve else float("nan"))
    unit_table["alpha_star_policy"] = alpha_star_policy
    unit_table["runs_at_alpha_star"] = runs_at_alpha_star
    unit_table["runs_at_alpha_star_per_swing"] = (
        unit_table["runs_at_alpha_star"] * N_SEASONS / unit_table["n_swings"])

    adjustability = pd.read_parquet(DATA / "adjustability.parquet", columns=KEY + [
        "label", "adjustability", "adj_count", "adjustability_plus",
        "adjustability_pctile",
        "twostrike_rv_penalty", "gamestate_rv_penalty", "platoon_rv_penalty",
    ]).merge(compute_swing_plus(), on=KEY, how="left")
    cards = pd.read_parquet(DATA / "shape_cards.parquet",
                            columns=KEY + ["role", "archetype_name", "grade"])
    primary_shape = (cards[cards["role"] == "primary"][KEY + ["archetype_name", "grade"]]
                     .rename(columns={"grade": "primary_grade"}))
    repertoire = pd.read_parquet(DATA / "repertoire_scores.parquet",
                                 columns=KEY + ["repertoire_pctile", "effective_shapes"])

    unit_table = (unit_table.merge(adjustability, on=KEY, how="left")
                            .merge(primary_shape, on=KEY, how="left")
                            .merge(repertoire, on=KEY, how="left"))

    suffix = "_subset" if args.limit else ""
    gradients = (pd.DataFrame(gradient_rows)
                 .merge(unit_table[KEY + ["label"]], on=KEY, how="left"))
    unit_table = unit_table.merge(axis_top_levers(gradients), on=KEY, how="left")

    value_path = DATA / f"adjustability_value{suffix}.parquet"
    gradient_path = DATA / f"adjustability_gradients{suffix}.parquet"
    unit_table.to_parquet(value_path, index=False)
    gradients.to_parquet(gradient_path, index=False)
    print(f"\nWrote {len(unit_table)} rows -> {value_path}")
    print(f"Wrote {len(gradients)} rows -> {gradient_path}")

    if not args.limit and (args.repeg or not POLICY_REF.exists()):
        POLICY_REF.write_text(json.dumps(reference, indent=2) + "\n")
        print(f"Pegged policy reference -> {POLICY_REF}")

    print("\n=== Season runs vs a replacement hitter's policy ===")
    for col in ["runs_total", "runs_count", "runs_gamestate", "runs_platoon",
                "runs_interaction", "runs_total_2k", "runs_vs_desituated"]:
        values = unit_table[col]
        print(f"  {col:<20} mean={values.mean():+6.2f}  median={values.median():+6.2f}  "
              f"sd={values.std():5.2f}  %pos={100 * (values > 0).mean():5.1f}  "
              f"range=[{values.min():+6.1f}, {values.max():+6.1f}]")
    corr_n_swings = unit_table["runs_total"].corr(unit_table["n_swings"])
    corr_swing_plus = unit_table["runs_total"].corr(unit_table["swing_plus"])
    print(f"\n  corr(runs_total, n_swings)  = {corr_n_swings:+.3f}")
    print(f"  corr(runs_total, swing_plus) = {corr_swing_plus:+.3f}")
    interaction = unit_table["runs_interaction"].abs()
    total = unit_table["runs_total"].abs().clip(lower=0.1)
    print(f"\n  |interaction| mean={interaction.mean():.2f}  max={interaction.max():.2f}  "
          f"median % of |total|={100 * (interaction / total).median():.0f}%")

    print("\n=== Policy ===")
    print(f"  displacement cap Q{100 * cf.POLICY_QUANTILE:.0f} = {displacement_cap:.3f} SD")
    for col in ["displacement_sd", "alpha_peak_unconstrained", "alpha_star_policy",
                "runs_at_alpha_star", "runs_at_alpha_star_per_swing"]:
        values = unit_table[col]
        print(f"  {col:<28} mean={values.mean():+7.3f}  median={values.median():+7.3f}")
    at_boundary = 100 * unit_table["alpha_peak_at_boundary"].mean()
    print(f"  peak still at a grid edge: {at_boundary:.0f}% of units")
    for axis in AXIS_NAMES:
        print(f"  alpha_star_{axis:<10} median={unit_table[f'alpha_star_{axis}'].median():.2f}  "
              f"runs range median={unit_table[f'runs_range_{axis}'].median():.3f}")

    print("\n=== Situational gradient (cohort mean |cell - own baseline|, runs/swing per SD) ===")
    with_abs = gradients.assign(g=gradients["grad_delta_runs_per_sd"].abs())
    gradient_pivot = with_abs.pivot_table(index="dial", columns="cell", values="g", aggfunc="mean")
    print(gradient_pivot.round(4).to_string())

    print("\n=== Spot check ===")
    spot_check = unit_table[unit_table["label"]
                            .str.contains("Judge|Arraez|Schwarber|Teoscar", na=False)]
    print(spot_check[["label", "runs_total", "alpha_star_policy", "runs_at_alpha_star"]
                     + [f"top_lever_{a}" for a in AXIS_NAMES]].to_string(index=False))


if __name__ == "__main__":
    main()
