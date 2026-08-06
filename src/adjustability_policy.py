"""Tells one hitter which part of his swing to change in which spot, and by how much: flatten
it with two strikes, aim more up the middle against a same-handed arm. Every suggestion is
re-graded through the run-value model and thrown out unless he has already shown he can make
that swing.

Two stages, in one file because they are one idea:

  1. SEARCH   per unit x axis x lever, the signed step that buys the most runs under a
              mean-preserving ordinal contrast, subject to an envelope rail and a repertoire
              rail. Slow (~25 min): every candidate shape is re-scored through xRV.
  2. READOUT  turns those steps into instructions in the dial's own units, annotates each
              with how well it repeats, and pools them into a league policy and a
              power/contact policy. Seconds.

`--report-only` runs stage 2 against the parquet stage 1 already wrote, which is what you
want whenever the wording or the pooling changes but the search does not.

The reliability annotation says which of three levels a cell should be read at, and never
excludes: count steps repeat per batter, gamestate steps only hold up as a cohort statement,
platoon sits between. An axis with a weak per-hitter number still has a real league mean.

Input : data/adjustability_gradients.parquet (cell gradients, from adjustability_value.py)
        data/adjustability_policy_reliability.parquet (from experiments/policy_search_gate.py)
Output: data/adjustability_prescriptions.parquet  (unit x axis x lever, the raw search)
        data/adjustability_playbook.parquet       (the same rows, as instructions)
        results/adjustability_playbook.md
Run   : python src/adjustability_policy.py [--limit N] [--report-only]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adjustability_value as av                                     # noqa: E402
from adjustability import KEY, SEASONS                               # noqa: E402

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

# --- stage 2 vocabulary -------------------------------------------------------------------
# How each axis reads out loud. The high pole is the +1 end of the ordinal contrast, so a
# positive `step_sd` means "more of this dial at the high pole than at the low pole".
POLES = {
    "count":     ("with two strikes", "early in the count"),
    "gamestate": ("with runners on", "with the bases empty"),
    "platoon":   ("vs a same-handed arm", "vs an opposite-handed arm"),
}
DIAL_WORDS = {
    "swing_path_tilt":        ("steepen", "flatten"),
    "swing_length":           ("lengthen", "shorten"),
    "vert_attack_angle":      ("swing more uphill", "swing more level"),
    "horz_attack_angle_pull": ("aim more to pull", "aim more to oppo"),
}
# The search works in league SD because that is the only scale on which four different
# quantities are comparable, but nobody can act on "0.5 SD". These convert back to the units
# the dial is actually measured in, using the same frozen league scale the search used.
LEAGUE_SD = json.loads(av.POLICY_REF.read_text())["shape_sd"]
# dial -> (multiplier from the raw feature's unit to the display unit, suffix, name)
UNITS = {
    "swing_path_tilt":        (1.0, "°", "degrees of tilt"),
    "swing_length":           (12.0, '"', "inches of swing length"),
    "vert_attack_angle":      (1.0, "°", "degrees of attack angle"),
    "horz_attack_angle_pull": (1.0, "°", "degrees of aim"),
}
# Where a prescription should be read. Split-half r on the step itself for the per-batter
# number; sign agreement (chance = 0.50) for whether at least the DIRECTION repeats.
STEP_R_BATTER = 0.40
SIGN_AGREE_DIRECTION = 0.55
TYPE_QUANTILE = 0.25


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
    scored = av.runs_for(models, run_value_tables, tiled_pitches, np.vstack(candidates))
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


def feasible(hitter, shifted, cell_means_sd, centroids):
    """Envelope rail on every swing, repertoire rail on every cell's mean shape."""
    if hitter.fraction_outside(shifted) >= av.MAX_OUTSIDE:
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


def axis_prescriptions(models, run_value_tables, hitter, axis, labels, gradient,
                       centroids, scale):
    """Best signed step for each dial on one axis, all mean-preserving, vs the status quo."""
    contrast = contrast_vector(labels, axis)
    if contrast is None or not np.any(contrast):
        return []
    group, base = hitter.swings, hitter.as_he_swung_it

    cells, cell_masks = [], []
    for cell in pd.unique(labels):
        in_cell = labels == cell
        if in_cell.sum() >= av.MIN_CELL_SWINGS:
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
            if feasible(hitter, shifted, cell_means_sd, centroids):
                shifts.append(shift)
                candidate_tags.append((lever, step))

    candidate_shapes = [base + shift * scale for shift in shifts]
    scored = score_candidates(models, run_value_tables, group, candidate_shapes)
    runs_pooled = scored.sum(axis=1)

    # Per-season sums, so a step can be chosen on one season and priced on the other. The
    # shipped `step_sd` is still the pooled argmax — that is the best estimate of the
    # recommendation — but `runs_gain` is what it is worth on unseen swings.
    years = group["game_year"].to_numpy()
    seasons = [year for year in np.unique(years) if (years == year).sum() >= av.MIN_CELL_SWINGS]
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


def unit_prescriptions(models, run_value_tables, hitter, gradients, centroids, scale):
    group = hitter.swings
    cells = hitter.situation_cells()
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

        for row in axis_prescriptions(models, run_value_tables, hitter, axis, labels,
                                      axis_gradient, centroids, scale):
            rows.append({
                "batter_id": group["batter_id"].iloc[0],
                "batter_stand": group["batter_stand"].iloc[0],
                "n_swings": len(group), **row,
            })
    return rows


# ----------------------------------------------------------------------------------------
# Stage 2: the readout
# ----------------------------------------------------------------------------------------


def hitter_types():
    """Power / Balanced / Contact from bat speed and whiff rate over the cohort seasons.

    Same split the 2026-08-04 gradient work used: the two z-scores correlate -0.55, so their
    difference is a real axis rather than a restatement of either one.
    """
    swings = pd.read_parquet(DATA / "swings_model.parquet",
                             columns=KEY + ["game_year", "bat_speed", "is_whiff"])
    swings = swings[swings["game_year"].isin(SEASONS)]
    per_unit = swings.groupby(KEY, observed=True).agg(
        bat_speed_mean=("bat_speed", "mean"), whiff_rate=("is_whiff", "mean")).reset_index()

    power_z = ((per_unit["bat_speed_mean"] - per_unit["bat_speed_mean"].mean())
               / per_unit["bat_speed_mean"].std())
    contact_z = -((per_unit["whiff_rate"] - per_unit["whiff_rate"].mean())
                  / per_unit["whiff_rate"].std())
    per_unit["power_contact_z"] = power_z - contact_z

    low = per_unit["power_contact_z"].quantile(TYPE_QUANTILE)
    high = per_unit["power_contact_z"].quantile(1 - TYPE_QUANTILE)
    per_unit["hitter_type"] = np.select(
        [per_unit["power_contact_z"] >= high, per_unit["power_contact_z"] <= low],
        ["Power", "Contact"], default="Balanced")
    return per_unit


def to_native(lever, step_sd):
    """A step in league SD converted to the dial's own unit. NaN for the gradient mix, which
    moves four dials at once and so has no single unit."""
    if lever not in UNITS:
        return float("nan")
    multiplier, _, _ = UNITS[lever]
    return step_sd * LEAGUE_SD[lever] * multiplier


def instruction(axis, lever, step):
    """One sentence a coach can say, in the dial's own units."""
    high, low = POLES[axis]
    if step == 0:
        return f"{high}: no change from how he swings {low}"
    if lever == "gradient":
        # The gradient mix moves all four dials at once; it prices the ceiling a hitter could
        # reach, not an instruction, so it is described rather than phrased as advice.
        return f"{high}: best mixed move is {abs(step):.2f} SD along his situational gradient"
    up, down = DIAL_WORDS[lever]
    _, suffix, name = UNITS[lever]
    verb = up if step > 0 else down
    return (f"{high}: {verb} by {abs(to_native(lever, step)):.1f}{suffix} "
            f"({name}) relative to {low}")


def read_at(step_r, sign_agree):
    """Which level this (axis, dial) cell should be read at, from the split-half gate."""
    if step_r >= STEP_R_BATTER:
        return "batter"
    if sign_agree >= SIGN_AGREE_DIRECTION:
        return "batter (direction only)"
    return "type / league"


def build_playbook(prescriptions):
    reliability = pd.read_parquet(DATA / "adjustability_policy_reliability.parquet")
    value = pd.read_parquet(DATA / "adjustability_value.parquet")

    playbook = prescriptions.merge(reliability[["axis", "lever", "step_r", "sign_agree",
                                                "runs_gain_r"]], on=["axis", "lever"])
    playbook = playbook.merge(value[KEY + ["label", "swing_plus", "adj_count"]], on=KEY)
    playbook = playbook.merge(hitter_types(), on=KEY)

    playbook["step_native"] = [to_native(lever, step) for lever, step
                               in zip(playbook["lever"], playbook["step_sd"])]
    playbook["step_unit"] = [UNITS[lever][1] if lever in UNITS else ""
                             for lever in playbook["lever"]]
    playbook["instruction"] = [instruction(axis, lever, step) for axis, lever, step
                               in zip(playbook["axis"], playbook["lever"],
                                      playbook["step_sd"])]
    playbook["read_at"] = [read_at(r, a) for r, a
                           in zip(playbook["step_r"], playbook["sign_agree"])]
    return playbook.sort_values(KEY + ["axis", "runs_gain_insample"],
                                ascending=[True, True, True, False])


def pooled_policy(playbook, group_cols):
    """The cohort's own prescription for each axis x dial: direction, size, and what it pays.

    `share_positive` is the vote — among the hitters the search moves, the fraction moved
    toward the high pole of the axis — and is the part that survives at the league level even
    where the per-batter step does not repeat.
    """
    single = playbook[playbook["lever"].isin(LEVERS)].copy()
    # runs_gain is a one-season total, so comparing buckets on it would rank playing time.
    # n_swings spans both seasons.
    single["runs_gain_per_100"] = (single["runs_gain"]
                                   / (single["n_swings"] / len(SEASONS)) * 100)
    rows = []
    for keys, block in single.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        moved = block[block["step_sd"] != 0]
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_units": len(block),
            "share_moved": round(float((block["step_sd"] != 0).mean()), 3),
            # The vote is taken among hitters the search actually moves. Over everyone it is
            # diluted by the ~30% left at the status quo, which are abstentions, not votes.
            "share_positive": round(float((moved["step_sd"] > 0).mean()), 3) if len(moved) else 0.0,
            "median_step": round(float(moved["step_native"].median()), 2) if len(moved) else 0.0,
            "mean_step": round(float(block["step_native"].mean()), 2),
            "unit": block["step_unit"].iloc[0],
            "mean_runs_gain": round(float(block["runs_gain"].mean()), 3),
            "runs_gain_per_100": round(float(block["runs_gain_per_100"].mean()), 3),
            "read_at": block["read_at"].iloc[0],
        })
    return pd.DataFrame(rows)


def markdown_report(playbook, league, by_type):
    lines = ["# Adjustability playbook", "",
             f"{playbook[KEY].drop_duplicates().shape[0]} units, {SEASONS[0]}-{SEASONS[-1]}. "
             "Steps are in each dial's own units (degrees of angle, inches of swing length), "
             "signed toward the high pole of each axis. `mean_runs_gain` is priced on the "
             "held-out season.", ""]

    lines += ["## League policy", "",
              "What the search asks of the average hitter. `share_positive` is the cohort's "
              "vote on direction; read it where the per-batter step does not repeat.", ""]
    for axis in CONTRASTS:
        high, low = POLES[axis]
        block = league[league["axis"] == axis].sort_values("mean_runs_gain", ascending=False)
        lines += [f"### {axis} — {high} vs {low}", "",
                  block.drop(columns=["axis"]).to_markdown(index=False), ""]

    lines += ["## By hitter type", "",
              "Power / Contact are the outer quartiles of z(bat speed) - z(contact rate); "
              "the middle half is Balanced.", ""]
    for axis in CONTRASTS:
        block = by_type[by_type["axis"] == axis]
        pivot = block.pivot(index="lever", columns="hitter_type", values="mean_step")
        gains = block.pivot(index="lever", columns="hitter_type",
                            values="runs_gain_per_100")
        lines += [f"### {axis}", "",
                  "Mean prescribed step (degrees, except swing length in inches):", "",
                  pivot.round(2).to_markdown(), "",
                  "Held-out runs gained per 100 swings:", "",
                  gains.round(3).to_markdown(), ""]

    lines += ["## Where each prescription is readable", "",
              playbook.drop_duplicates(["axis", "lever"])[
                  ["axis", "lever", "step_r", "sign_agree", "runs_gain_r", "read_at"]
              ].sort_values(["axis", "lever"]).to_markdown(index=False), ""]

    lines += ["## Sample batter cards", "",
              "Two per type, taken at each type's median total gain. Picking the largest "
              "totals instead would just surface the hitters with the most swings.", ""]
    # runs_gain is NaN for units that only reach the swing floor in one season, so they have
    # no held-out price and cannot be carded.
    single = playbook[playbook["lever"].isin(LEVERS) & (playbook["step_sd"] != 0)]
    single = single.dropna(subset=["runs_gain"])
    best = single.loc[single.groupby(KEY + ["axis"])["runs_gain"].idxmax()]

    totals = best.groupby(["hitter_type", "label"])["runs_gain"].sum().reset_index()
    shown = []
    for _, block in totals.groupby("hitter_type"):
        ranked = block.sort_values("runs_gain").reset_index(drop=True)
        middle = len(ranked) // 2
        shown.extend(ranked.loc[middle:middle + 1, "label"])
    for label in shown:
        card = best[best["label"] == label]
        lines += [f"### {label} ({card['hitter_type'].iloc[0]})", ""]
        for _, row in card.iterrows():
            lines.append(f"- **{row['axis']}** — {row['instruction']} "
                         f"({row['runs_gain']:+.2f} runs held out, read at {row['read_at']})")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------


def search(limit=None):
    """Stage 1. Re-scores every candidate shape through xRV, so this is the slow half."""
    models, run_value_tables, swings, reference = av.load_data()
    scale = np.asarray([reference["shape_sd"][feature] for feature in SHAPE], float)
    centroids = load_centroids(scale)
    all_gradients = pd.read_parquet(DATA / "adjustability_gradients.parquet")

    groups = list(swings.groupby(KEY, observed=True, sort=False))
    if limit:
        groups = groups[:limit]
    print(f"{len(groups)} units, searching {len(STEP_GRID)} steps "
          f"x {len(SHAPE) + 1} directions per cell")

    gradients_by_unit = dict(list(all_gradients.groupby(KEY, observed=True)))
    rows = []
    for i, (key, group) in enumerate(groups, 1):
        if key not in centroids or key not in gradients_by_unit:
            continue
        hitter = av.learn_how_he_swings(group.reset_index(drop=True), scale)
        rows.extend(unit_prescriptions(models, run_value_tables, hitter,
                                       gradients_by_unit[key], centroids[key], scale))
        if i % 25 == 0:
            print(f"  {i}/{len(groups)} units", flush=True)

    prescriptions = pd.DataFrame(rows)
    suffix = "_subset" if limit else ""
    output_path = DATA / f"adjustability_prescriptions{suffix}.parquet"
    prescriptions.to_parquet(output_path, index=False)
    print(f"\nWrote {len(prescriptions)} rows -> {output_path}")
    return prescriptions


def report(prescriptions):
    """Stage 2. Cheap: no model calls, just the search output turned into instructions."""
    playbook = build_playbook(prescriptions)
    out = DATA / "adjustability_playbook.parquet"
    playbook.to_parquet(out, index=False)
    print(f"\nWrote {len(playbook)} rows -> {out}")

    league = pooled_policy(playbook, ["axis", "lever"])
    by_type = pooled_policy(playbook, ["axis", "lever", "hitter_type"])

    print("\n=== League policy: prescribed step in the dial's own units, and what it pays ===")
    print(league.to_string(index=False))

    print("\n=== By hitter type: mean prescribed step ===")
    for axis in CONTRASTS:
        block = by_type[by_type["axis"] == axis]
        print(f"\n  {axis}:")
        print(block.pivot(index="lever", columns="hitter_type",
                          values="mean_step").round(2).to_string())

    print("\n=== How often each lever is the axis winner ===")
    winners = prescriptions.loc[
        prescriptions.groupby(KEY + ["axis"])["runs_gain_insample"].idxmax()]
    print(pd.crosstab(winners["lever"], winners["axis"]).to_string())

    at_status_quo = 100 * (prescriptions["step_sd"] == 0).mean()
    print(f"\n  status quo already optimal: {at_status_quo:.1f}% of (unit, axis, lever) cells")
    print(f"  held-out gain positive: {100 * (prescriptions['runs_gain'] > 0).mean():.1f}%")
    print(f"  mean feasible steps per lever: {prescriptions['n_feasible'].mean():.1f} "
          f"of {len(STEP_GRID)}")
    counts = playbook.drop_duplicates(KEY)["hitter_type"].value_counts()
    print(f"  hitter types: {counts.to_dict()}")

    # The gradient mix is a ceiling, not advice a coach can give, so the roll-up counts
    # only the single-dial winners.
    print("\n  best single-dial gain per unit-axis, summed over axes:")
    single_dial_winners = winners[winners["lever"] != "gradient"]
    best_dial_runs = single_dial_winners.groupby(KEY)["runs_gain"].sum()
    print(f"    mean {best_dial_runs.mean():+.3f}  median {best_dial_runs.median():+.3f}  "
          f"max {best_dial_runs.max():+.3f} season runs")

    report_path = ROOT / "results" / "adjustability_playbook.md"
    report_path.write_text(markdown_report(playbook, league, by_type))
    print(f"\nWrote {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-only", action="store_true",
                        help="skip the search and re-read the prescriptions parquet")
    args = parser.parse_args()

    if args.report_only:
        suffix = "_subset" if args.limit else ""
        prescriptions = pd.read_parquet(
            DATA / f"adjustability_prescriptions{suffix}.parquet")
    else:
        prescriptions = search(args.limit)
    report(prescriptions)


if __name__ == "__main__":
    main()
