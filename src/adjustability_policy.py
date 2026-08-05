"""Tells one hitter which part of his swing to change in which spot, and by how much: flatten
it with two strikes, aim more up the middle against a same-handed arm. Every suggestion is
re-graded through the run-value model and thrown out unless he has already shown he can make
that swing.

Input : data/adjustability_gradients.parquet (cell gradients, from adjustability_value.py)
Output: data/adjustability_prescriptions.parquet (unit x axis x cell)
Run   : python src/adjustability_policy.py [--limit N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adjustability_value as av                                     # noqa: E402
import counterfactual as cf                                          # noqa: E402
import xRV_model as xrv                                              # noqa: E402
from adjustability import KEY                                        # noqa: E402

SHAPE = av.SHAPE
# bat_speed is NOT prescribable. Within a hitter his hardest swings whiff less (27% -> 19%),
# make more contact (27% -> 45%) and gain run value monotonically, so xRV's gradient on it is
# positive in 96% of cells while every geometry dial is two-sided. That is reverse causality,
# not a free lever: bat speed measured on a swing mostly reads out whether he was on time, so
# "add bat speed" decodes to "be fooled less often" and is not an adjustment he can choose.
# Same carve swing_label.py already makes for archetypes (bat_speed is state-not-trait, ICC 0.126).
LEVERS = [dial for dial in SHAPE if dial != "bat_speed"]
STEP_GRID = [-0.75, -0.5, -0.3, -0.15, 0.15, 0.3, 0.5, 0.75]   # league-SD units, signed
MAX_CENTROID_SD = 1.0                    # Euclidean, 5-dim, league-SD units
# Ordinal contrasts, one per axis: the direction along which modulation is prescribed.
# Count is ordered in strikes; gamestate is any-runner vs empty, matching how
# `gamestate_rv_penalty` is matched; platoon is same- vs opposite-hand.
CONTRASTS = {
    "count":     {"0 strikes": -1.0, "1 strike": 0.0, "2 strikes": 1.0},
    "gamestate": {"empty": -1.0, "on1": 1.0, "risp": 1.0},
    "platoon":   {"opp-hand": -1.0, "same-hand": 1.0},
}
CENTROID_COLS = {"swing_path_tilt": "swing_path_tilt_mean",
                 "swing_length": "swing_length_mean",
                 "bat_speed": "bat_speed_mean",
                 "vert_attack_angle": "vert_attack_angle_mean",
                 "horz_attack_angle_pull": "horz_attack_angle_mean"}


def load_centroids(scale):
    """Each unit's demonstrated swing shapes, in league-SD units, as a (k, 5) array.

    `horz_attack_angle_mean` is stored raw; SHAPE uses the pull frame, a uniform negation
    for both hands (see CLAUDE.md handedness convention).
    """
    summary = pd.read_parquet(DATA / "cluster_summary.parquet")
    summary["horz_attack_angle_mean"] = -summary["horz_attack_angle_mean"]
    centroid_cols = [CENTROID_COLS[feature] for feature in SHAPE]
    centroids = {}
    for key, unit_block in summary.groupby(KEY, observed=True):
        centroids[key] = unit_block[centroid_cols].to_numpy(float) / scale
    return centroids


def score_candidates(models, run_value_tables, group, candidates):
    """Per-swing xRV for every candidate shape matrix, in ONE batched booster call.

    The per-candidate loop is the cost centre of the search; tiling the pitches and
    scoring once is the same arithmetic at a fraction of the overhead.
    """
    tiled_pitches = pd.concat([group] * len(candidates), ignore_index=True)
    scored = av.score_shapes(models, run_value_tables, tiled_pitches, np.vstack(candidates))
    return scored.reshape(len(candidates), len(group))


def contrast_vector(labels, axis):
    """Per-swing contrast weights for an axis, n-weighted to mean zero.

    Centering is what makes the policy mean-preserving: the weighted average shape shift
    is exactly zero, so the search cannot buy runs by improving the swing overall.
    """
    weights = np.array([CONTRASTS[axis].get(str(cell), np.nan) for cell in labels], float)
    if np.isnan(weights).any():
        return None
    return weights - weights.mean()


def directions(gradient):
    """Each single dial on its own, plus the situational gradient's own mix as a ceiling.

    Single-dial directions are what a coach can say out loud. Signs are left free — the
    step grid is signed — so the search decides whether to raise or lower the dial.
    """
    identity = np.eye(len(SHAPE))
    by_lever = {}
    for dial in LEVERS:
        by_lever[dial] = identity[SHAPE.index(dial)]

    # The gradient ceiling is projected onto the prescribable dials too, otherwise it just
    # rediscovers bat speed and every hitter's best "mix" is the one lever he cannot pull.
    prescribable = np.array([dial in LEVERS for dial in SHAPE], float)
    mix = np.asarray(gradient, float) * prescribable
    mix_norm = np.linalg.norm(mix)
    if mix_norm > 1e-12:
        by_lever["gradient"] = mix / mix_norm
    return by_lever


def feasible(shifted, cell_means_sd, centroids, env):
    """Envelope rail on every swing, repertoire rail on every cell's mean shape."""
    if cf.fraction_outside(shifted, env) >= cf.MAX_OUTSIDE:
        return False
    centroid_gaps = []
    for cell_mean in cell_means_sd:
        centroid_gaps.append(np.linalg.norm(centroids - cell_mean, axis=1).min())
    return bool(max(centroid_gaps) <= MAX_CENTROID_SD)


def _best_candidate(candidate_indices, runs):
    """Index of the highest-scoring candidate, ties going to the first one tried."""
    best_index = candidate_indices[0]
    for index in candidate_indices:
        if runs[index] > runs[best_index]:
            best_index = index
    return best_index


def _held_out_gain(candidate_indices, runs_by_season, seasons):
    """Gain of a step CHOSEN on one season and PRICED on the other, averaged both ways.

    Taking the argmax of ~40 candidates on the same swings that price it is a maximum of
    noisy evaluations and is biased up by construction, so the honest number is out of
    sample. Index 0 is always the status quo, which is what the gain is measured against.
    """
    first, second = seasons
    gains = []
    for priced_on, chosen_on in ((first, second), (second, first)):
        chosen = _best_candidate(candidate_indices, runs_by_season[chosen_on])
        gains.append(runs_by_season[priced_on][chosen] - runs_by_season[priced_on][0])
    return float(np.mean(gains))


def axis_prescriptions(models, run_value_tables, fit, axis, labels, gradient,
                       centroids, env, scale):
    """Best signed step for each dial on one axis, all mean-preserving, vs the status quo."""
    contrast = contrast_vector(labels, axis)
    if contrast is None or not np.any(contrast):
        return []
    group, base = fit["group"], fit["shape_actual"]

    cells, cell_masks = [], []
    for cell in pd.unique(labels):
        in_cell = labels == cell
        if in_cell.sum() >= cf.MIN_CELL_SWINGS:
            cells.append(cell)
            cell_masks.append(in_cell)

    # Candidate 0 is always the status quo, so every gain below is measured against it.
    lever_directions = directions(gradient)
    shifts, candidate_tags = [np.zeros_like(base)], [("status quo", 0.0)]
    for lever, direction in lever_directions.items():
        for step in STEP_GRID:
            shift = np.outer(contrast, direction * step)
            shifted = base + shift * scale
            cell_means_sd = [shifted[mask].mean(axis=0) / scale for mask in cell_masks]
            if feasible(shifted, cell_means_sd, centroids, env):
                shifts.append(shift)
                candidate_tags.append((lever, step))

    candidate_shapes = [base + shift * scale for shift in shifts]
    scored = score_candidates(models, run_value_tables, group, candidate_shapes)
    runs_pooled = scored.sum(axis=1)

    # Per-season sums, so a step can be chosen on one season and priced on the other. The
    # shipped `step_sd` is still the pooled argmax — that is the best estimate of the
    # recommendation — but `runs_gain` is what it is worth on unseen swings.
    years = group["game_year"].to_numpy()
    seasons = [year for year in np.unique(years) if (years == year).sum() >= cf.MIN_CELL_SWINGS]
    runs_by_season = {}
    for year in seasons:
        runs_by_season[year] = scored[:, years == year].sum(axis=1)

    contrast_parts = []
    for cell in cells:
        contrast_parts.append(f"{cell}:{CONTRASTS[axis][str(cell)]:+.0f}")

    rows = []
    for lever in lever_directions:
        candidate_indices = [index for index, (tag, _) in enumerate(candidate_tags)
                             if tag == lever]
        candidate_indices.append(0)
        best = _best_candidate(candidate_indices, runs_pooled)
        held_out = float("nan")
        if len(seasons) == 2:
            held_out = _held_out_gain(candidate_indices, runs_by_season, seasons)
        rows.append({
            "axis": axis, "lever": lever,
            "step_sd": candidate_tags[best][1],
            "runs_gain": held_out,
            "runs_gain_insample": float((runs_pooled[best] - runs_pooled[0]) / av.N_SEASONS),
            "n_feasible": len(candidate_indices) - 1,
            "contrast": " / ".join(contrast_parts),
        })
    return rows


def unit_prescriptions(models, run_value_tables, fit, gradients, centroids, scale):
    group = fit["group"]
    env = cf.envelope(fit["observed"])
    cells = cf.cell_labels(group)
    # The SITUATIONAL gradient, not the absolute one: the raw cell gradient is dominated
    # by the hitter's own baseline, so a direction built from it points at how he hits in
    # general rather than at what this situation asks for.
    cell_gradients = {}
    for (axis, cell), block in gradients.groupby(["axis", "cell"], observed=True):
        cell_gradients[(axis, cell)] = block.set_index("dial")["grad_delta_runs_per_sd"]

    rows = []
    for axis, labels in cells.items():
        # Sum the cells' gradients along the axis contrast, so the direction points the way
        # the situation asks rather than at any one cell.
        axis_gradient_parts = []
        for cell in pd.unique(labels):
            sign = CONTRASTS[axis].get(str(cell), 0.0)
            if (axis, cell) in cell_gradients:
                by_dial = cell_gradients[(axis, cell)].reindex(SHAPE).to_numpy(float)
                axis_gradient_parts.append(sign * by_dial)
        if not axis_gradient_parts:
            continue
        axis_gradient = np.sum(axis_gradient_parts, axis=0)

        for row in axis_prescriptions(models, run_value_tables, fit, axis, labels,
                                      axis_gradient, centroids, env, scale):
            rows.append({
                "batter_id": group["batter_id"].iloc[0],
                "batter_stand": group["batter_stand"].iloc[0],
                "n_swings": len(group), **row,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    models = av.load_models()
    run_value_tables = xrv.load_run_value_tables()
    swings = av.load_swings()
    reference = av.load_policy_reference(swings)
    scale = np.asarray([reference["shape_sd"][feature] for feature in SHAPE], float)
    centroids = load_centroids(scale)
    all_gradients = pd.read_parquet(DATA / "adjustability_gradients.parquet")

    groups = list(swings.groupby(KEY, observed=True, sort=False))
    if args.limit:
        groups = groups[:args.limit]
    print(f"{len(groups)} units, searching {len(STEP_GRID)} steps "
          f"x {len(SHAPE) + 1} directions per cell")

    gradients_by_unit = dict(list(all_gradients.groupby(KEY, observed=True)))
    rows = []
    for i, (key, group) in enumerate(groups, 1):
        if key not in centroids or key not in gradients_by_unit:
            continue
        fit = av.unit_fit(group.reset_index(drop=True), scale)
        rows.extend(unit_prescriptions(models, run_value_tables, fit,
                                       gradients_by_unit[key], centroids[key], scale))
        if i % 25 == 0:
            print(f"  {i}/{len(groups)} units", flush=True)

    prescriptions = pd.DataFrame(rows)
    suffix = "_subset" if args.limit else ""
    output_path = DATA / f"adjustability_prescriptions{suffix}.parquet"
    prescriptions.to_parquet(output_path, index=False)
    print(f"\nWrote {len(prescriptions)} rows -> {output_path}")

    print("\n=== Runs left on the table (mean-preserving contrast), by axis x lever ===")
    for name, col in [("held-out season", "runs_gain"), ("in-sample", "runs_gain_insample")]:
        print(f"\n  {name}:")
        by_lever = prescriptions.pivot_table(index="lever", columns="axis",
                                             values=col, aggfunc="mean")
        print(by_lever.round(3).to_string())

    print("\n=== Prescribed step, in league SD, where the search moves off status quo ===")
    moved = prescriptions[prescriptions["step_sd"] != 0]
    steps = moved.pivot_table(index="lever", columns="axis", values="step_sd", aggfunc="median")
    print(steps.round(3).to_string())

    print("\n=== How often each lever is the axis winner ===")
    winners = prescriptions.loc[
        prescriptions.groupby(KEY + ["axis"])["runs_gain_insample"].idxmax()]
    print(pd.crosstab(winners["lever"], winners["axis"]).to_string())

    at_status_quo = 100 * (prescriptions["step_sd"] == 0).mean()
    print(f"\n  status quo already optimal: {at_status_quo:.1f}% "
          f"of (unit, axis, lever) cells")
    print(f"  held-out gain positive: {100 * (prescriptions['runs_gain'] > 0).mean():.1f}%")
    print(f"  mean feasible steps per lever: {prescriptions['n_feasible'].mean():.1f} "
          f"of {len(STEP_GRID)}")

    # The gradient mix is a ceiling, not advice a coach can give, so the roll-up counts
    # only the single-dial winners.
    print("\n  best single-dial gain per unit-axis, summed over axes:")
    single_dial_winners = winners[winners["lever"] != "gradient"]
    if len(single_dial_winners):
        best_dial_runs = single_dial_winners.groupby(KEY)["runs_gain"].sum()
    else:
        best_dial_runs = pd.Series(dtype=float)
    print(f"    mean {best_dial_runs.mean():+.3f}  median {best_dial_runs.median():+.3f}  "
          f"max {best_dial_runs.max():+.3f} season runs")


if __name__ == "__main__":
    main()
