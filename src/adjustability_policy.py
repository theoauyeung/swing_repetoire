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
LEVERS = [d for d in SHAPE if d != "bat_speed"]
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
    cols = [CENTROID_COLS[f] for f in SHAPE]
    return {key: block[cols].to_numpy(float) / scale
            for key, block in summary.groupby(KEY, observed=True)}


def score_candidates(models, run_value_tables, group, candidates):
    """Per-swing xRV for every candidate shape matrix, in ONE batched booster call.

    The per-candidate loop is the cost centre of the search; tiling the pitches and
    scoring once is the same arithmetic at a fraction of the overhead.
    """
    tiled = pd.concat([group] * len(candidates), ignore_index=True)
    scored = av.score_shapes(models, run_value_tables, tiled, np.vstack(candidates))
    return scored.reshape(len(candidates), len(group))


def contrast_vector(labels, axis):
    """Per-swing contrast weights for an axis, n-weighted to mean zero.

    Centering is what makes the policy mean-preserving: the weighted average shape shift
    is exactly zero, so the search cannot buy runs by improving the swing overall.
    """
    weights = np.array([CONTRASTS[axis].get(str(c), np.nan) for c in labels], float)
    if np.isnan(weights).any():
        return None
    return weights - weights.mean()


def directions(gradient):
    """Each single dial on its own, plus the situational gradient's own mix as a ceiling.

    Single-dial directions are what a coach can say out loud. Signs are left free — the
    step grid is signed — so the search decides whether to raise or lower the dial.
    """
    out = {dial: np.eye(len(SHAPE))[SHAPE.index(dial)] for dial in LEVERS}
    # The gradient ceiling is projected onto the prescribable dials too, otherwise it just
    # rediscovers bat speed and every hitter's best "mix" is the one lever he cannot pull.
    mix = np.array([g if d in LEVERS else 0.0 for d, g in zip(SHAPE, gradient)])
    norm = np.linalg.norm(mix)
    if norm > 1e-12:
        out["gradient"] = mix / norm
    return out


def feasible(shifted, cell_means_sd, centroids, env):
    """Envelope rail on every swing, repertoire rail on every cell's mean shape."""
    if cf.fraction_outside(shifted, env) >= cf.MAX_OUTSIDE:
        return False
    gaps = [np.linalg.norm(centroids - m, axis=1).min() for m in cell_means_sd]
    return bool(max(gaps) <= MAX_CENTROID_SD)


def axis_prescriptions(models, run_value_tables, fit, axis, labels, gradient,
                       centroids, env, scale):
    """Best signed step for each dial on one axis, all mean-preserving, vs the status quo."""
    contrast = contrast_vector(labels, axis)
    if contrast is None or not np.any(contrast):
        return []
    group, base = fit["group"], fit["shape_actual"]
    cells = [c for c in pd.unique(labels) if (labels == c).sum() >= cf.MIN_CELL_SWINGS]
    masks = [labels == c for c in cells]

    shifts, tags = [np.zeros_like(base)], [("status quo", 0.0)]
    for label, direction in directions(gradient).items():
        for step in STEP_GRID:
            shift = np.outer(contrast, direction * step)
            shifted = base + shift * scale
            means = [shifted[m].mean(axis=0) / scale for m in masks]
            if feasible(shifted, means, centroids, env):
                shifts.append(shift)
                tags.append((label, step))

    scored = score_candidates(models, run_value_tables, group,
                              [base + s * scale for s in shifts])
    totals = scored.sum(axis=1)
    # Per-season sums, so the step can be CHOSEN on one season and PRICED on the other.
    # Taking the argmax of ~40 candidates on the same swings that price it is a maximum of
    # noisy evaluations and is biased up by construction; the honest number is out of
    # sample. The shipped `step_sd` is still the pooled argmax — that is the best estimate
    # of the recommendation — but `runs_gain` is what it is worth on unseen swings.
    years = group["game_year"].to_numpy()
    seasons = [y for y in np.unique(years) if (years == y).sum() >= cf.MIN_CELL_SWINGS]
    by_season = {y: scored[:, years == y].sum(axis=1) for y in seasons}

    rows = []
    for label in list(directions(gradient)):
        picks = [i for i, (lab, _) in enumerate(tags) if lab == label] + [0]
        best = max(picks, key=lambda i: totals[i])
        honest = float("nan")
        if len(seasons) == 2:
            a, b = seasons
            honest = float(np.mean([
                by_season[h][max(picks, key=lambda i: by_season[t][i])] - by_season[h][0]
                for h, t in ((a, b), (b, a))]))
        rows.append({
            "axis": axis, "lever": label,
            "step_sd": tags[best][1],
            "runs_gain": honest,
            "runs_gain_insample": float((totals[best] - totals[0]) / av.N_SEASONS),
            "n_feasible": len(picks) - 1,
            "contrast": " / ".join(f"{c}:{CONTRASTS[axis][str(c)]:+.0f}" for c in cells),
        })
    return rows


def unit_prescriptions(models, run_value_tables, fit, gradients, centroids, scale):
    group = fit["group"]
    env = cf.envelope(fit["observed"])
    cells = cf.cell_labels(group)
    # The SITUATIONAL gradient, not the absolute one: the raw cell gradient is dominated
    # by the hitter's own baseline, so a direction built from it points at how he hits in
    # general rather than at what this situation asks for.
    lookup = {(a, c): block.set_index("dial")["grad_delta_runs_per_sd"]
              for (a, c), block in gradients.groupby(["axis", "cell"], observed=True)}

    rows = []
    for axis, labels in cells.items():
        signs = {c: CONTRASTS[axis].get(str(c), 0.0) for c in pd.unique(labels)}
        parts = [signs[c] * lookup[(axis, c)].reindex(SHAPE).to_numpy(float)
                 for c in signs if (axis, c) in lookup]
        if not parts:
            continue
        rows.extend({
            "batter_id": group["batter_id"].iloc[0],
            "batter_stand": group["batter_stand"].iloc[0],
            "n_swings": len(group), **row,
        } for row in axis_prescriptions(models, run_value_tables, fit, axis, labels,
                                        np.sum(parts, axis=0), centroids, env, scale))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    models = av.load_models()
    run_value_tables = xrv.load_run_value_tables()
    swings = av.load_swings()
    reference = av.load_policy_reference(swings)
    scale = np.asarray([reference["shape_sd"][f] for f in SHAPE], float)
    centroids = load_centroids(scale)
    all_gradients = pd.read_parquet(DATA / "adjustability_gradients.parquet")

    groups = list(swings.groupby(KEY, observed=True, sort=False))
    if args.limit:
        groups = groups[:args.limit]
    print(f"{len(groups)} units, searching {len(STEP_GRID)} steps "
          f"x {len(SHAPE) + 1} directions per cell")

    grad_by_unit = dict(list(all_gradients.groupby(KEY, observed=True)))
    rows = []
    for i, (key, group) in enumerate(groups, 1):
        if key not in centroids or key not in grad_by_unit:
            continue
        fit = av.unit_fit(group.reset_index(drop=True), scale)
        rows.extend(unit_prescriptions(models, run_value_tables, fit,
                                       grad_by_unit[key], centroids[key], scale))
        if i % 25 == 0:
            print(f"  {i}/{len(groups)} units", flush=True)

    df = pd.DataFrame(rows)
    suffix = "_subset" if args.limit else ""
    out = DATA / f"adjustability_prescriptions{suffix}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")

    print("\n=== Runs left on the table (mean-preserving contrast), by axis x lever ===")
    for name, col in [("held-out season", "runs_gain"), ("in-sample", "runs_gain_insample")]:
        print(f"\n  {name}:")
        print(df.pivot_table(index="lever", columns="axis", values=col,
                             aggfunc="mean").round(3).to_string())
    print("\n=== Prescribed step, in league SD, where the search moves off status quo ===")
    helped = df[df["step_sd"] != 0]
    print(helped.pivot_table(index="lever", columns="axis", values="step_sd",
                             aggfunc="median").round(3).to_string())
    print("\n=== How often each lever is the axis winner ===")
    best = df.loc[df.groupby(KEY + ["axis"])["runs_gain_insample"].idxmax()]
    print(pd.crosstab(best["lever"], best["axis"]).to_string())
    print(f"\n  status quo already optimal: {100 * (df['step_sd'] == 0).mean():.1f}% "
          f"of (unit, axis, lever) cells")
    print(f"  held-out gain positive: {100 * (df['runs_gain'] > 0).mean():.1f}%")
    print(f"  mean feasible steps per lever: {df['n_feasible'].mean():.1f} "
          f"of {len(STEP_GRID)}")
    print("\n  best single-dial gain per unit-axis, summed over axes:")
    total = (best[best["lever"] != "gradient"].groupby(KEY)["runs_gain"].sum()
             if (best["lever"] != "gradient").any() else pd.Series(dtype=float))
    print(f"    mean {total.mean():+.3f}  median {total.median():+.3f}  "
          f"max {total.max():+.3f} season runs")


if __name__ == "__main__":
    main()
