"""Writes a scouting card for each of a hitter's swings: what it looks like next to his main one,
when he goes to it, and what it's worth. One row per swing, so you can read "his B swing is 3 mph
slower and 5 degrees flatter, he uses it with two strikes, and it costs him nothing."

Input:  data/cluster_summary, shape_archetypes, cluster_assignments, xrv_swings, swings_model (parquet)
Outputs:
  data/shape_cards.parquet     one row per (batter, stand, cluster) with all card fields
  data/shape_cards_catalog.md  a few worked example cards

Run:  python src/cards.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# geometry centroid features (pull frame), used for the name-delta phrasing
GEO = ["bat_speed", "vert_attack_angle", "horz_attack_angle_pull", "swing_path_tilt", "swing_length"]
# (column, word for +delta, word for -delta, unit, format) — ordered by interpretive salience
PHRASE = [
    ("bat_speed", "faster", "slower", "mph", "{:+.0f}"),
    ("vert_attack_angle", "uppercut", "flatter", "deg", "{:+.0f}"),
    ("horz_attack_angle_pull", "pull", "oppo", "deg", "{:+.0f}"),
    ("swing_path_tilt", "steeper", "flatter-plane", "deg", "{:+.0f}"),
    ("swing_length", "longer", "shorter", "ft", "{:+.1f}"),
]
DELTA_Z = 0.5         # include a feature in the name phrase only if |delta| >= this many centroid-SDs
FASTBALLS = {"FF", "SI", "FC", "FT", "FA"}
MIN_CTX = 25          # min unit swings in a context bucket to compute a lift
MIN_CTX_SHAPE = 5     # min swings OF THIS SHAPE in the bucket
LIFT_MIN = 0.15       # report a context only if the shape is >= +15% over-indexed there
MIN_STRAT = 5         # min swings per shape per stratum to enter the matched contrast
MATCHED_THIN_N = 30   # matched support below this = flagged thin


def load():
    cluster_summary = pd.read_parquet(DATA / "cluster_summary.parquet")
    archetypes = pd.read_parquet(DATA / "shape_archetypes.parquet",
                                 columns=["batter_id", "batter_stand", "cluster", "label", "archetype_name",
                                          "arch_confidence", "weight", "n"] + GEO)
    cluster_assignments = pd.read_parquet(DATA / "cluster_assignments.parquet",
                                          columns=["play_id", "batter_id", "batter_stand", "cluster"])
    xrv_swings = pd.read_parquet(DATA / "xrv_swings.parquet", columns=["play_id", "xrv", "xrv_grade"])
    swings_model = pd.read_parquet(DATA / "swings_model.parquet",
                                   columns=["play_id", "batter_stand", "balls", "strikes", "plate_x", "plate_z",
                                            "sz_top", "sz_bot", "pitch_type", "outs_when_up",
                                            "on_1b_id", "on_2b_id", "on_3b_id"])
    return cluster_summary, archetypes, cluster_assignments, xrv_swings, swings_model


def build_frame(cluster_assignments, xrv_swings, swings_model):
    """Per-swing frame: cluster + value + context bucket columns (count / location / pitch / base-out
    booleans for the when-label, and a coarse count x height x pitch stratum for the matched contrast)."""
    swings = cluster_assignments.merge(xrv_swings, on="play_id", how="inner").merge(
        swings_model.drop(columns="batter_stand"), on="play_id", how="left")
    balls = swings.balls
    strikes = swings.strikes
    zone_height_normalized = (swings.plate_z - swings.sz_bot) / (swings.sz_top - swings.sz_bot)
    # plate_x is absolute (catcher frame): pull-side/inside = negative for RHH, positive for LHH,
    # so flip RHH. + = pull-side / inside for both hands (validated vs bearing_angle).
    plate_x_pull_relative = swings.plate_x * np.where(swings.batter_stand == "L", 1.0, -1.0)
    is_fastball = swings.pitch_type.isin(FASTBALLS)
    runner_presence = swings[["on_1b_id", "on_2b_id", "on_3b_id"]].notna()

    # when-label context buckets (rich; each needs only marginal support)
    swings["ctx"] = None  # placeholder; buckets held in a dict of masks returned separately
    buckets = {
        "0-0": (balls == 0) & (strikes == 0), "ahead": balls > strikes,
        "behind": (strikes > balls) & (strikes < 2),
        "2-strk": strikes == 2, "3-2": (balls == 3) & (strikes == 2),
        "up": zone_height_normalized > 0.66, "down": zone_height_normalized < 0.33,
        "inside": plate_x_pull_relative > 0.33, "away": plate_x_pull_relative < -0.33,
        "vs FB": is_fastball, "vs offspd": swings.pitch_type.notna() & ~is_fastball,
        "RISP": runner_presence.on_2b_id | runner_presence.on_3b_id,
        "bases empty": ~runner_presence.any(axis=1),
        "2 out": swings.outs_when_up == 2,
    }
    # matched-contrast stratum (coarse, needs JOINT support of two shapes): count3 x height3 x pitch2
    count_bin = np.where(strikes == 2, "2strk", np.where(balls > strikes, "ahead", "other"))
    height_bin = np.where(zone_height_normalized > 0.66, "up",
                          np.where(zone_height_normalized < 0.33, "down", "mid"))
    pitch_bin = np.where(is_fastball, "FB", "OS")
    swings["strat"] = pd.Series(count_bin, index=swings.index) + "|" + height_bin + "|" + pitch_bin
    return swings, buckets


def name_delta(centroid, reference, feature_sds):
    """Readable geometry phrase: top features (by |delta| in centroid-SD units) of centroid vs reference.
    Always emits at least the single biggest delta, even if it clears no threshold, so the phrase
    is never uninformative (a shape can be far from the reference with each feature diffusely
    under the notable bar)."""
    scored_phrases = []
    for col, positive_word, negative_word, unit, fmt in PHRASE:
        delta = centroid[col] - reference[col]
        normalized_delta = abs(delta) / feature_sds[col] if feature_sds[col] > 0 else 0.0
        direction_word = positive_word if delta > 0 else negative_word
        scored_phrases.append((normalized_delta, f"{fmt.format(delta)}{unit} {direction_word}"))
    scored_phrases.sort(reverse=True)
    notable_phrases = [phrase for score, phrase in scored_phrases if score >= DELTA_Z][:3] or [scored_phrases[0][1]]
    return ", ".join(notable_phrases)


def over_index(unit_swings, cluster, base_weight, buckets):
    """Sorted (lift, bucket-name) list of contexts where the shape is over-indexed vs the hitter's
    own baseline usage (lift = P(shape | ctx)/P(shape) - 1 >= LIFT_MIN, with support)."""
    if base_weight <= 0:
        return []
    is_this_cluster = (unit_swings.cluster == cluster).to_numpy()
    over_indexed = []
    for bucket_name, bucket_mask in buckets.items():
        in_bucket = bucket_mask.loc[unit_swings.index].to_numpy()
        total_in_bucket = int(in_bucket.sum())
        cluster_in_bucket = int((in_bucket & is_this_cluster).sum())
        if total_in_bucket < MIN_CTX or cluster_in_bucket < MIN_CTX_SHAPE:
            continue
        lift = (cluster_in_bucket / total_in_bucket) / base_weight - 1.0
        if lift >= LIFT_MIN:
            over_indexed.append((lift, bucket_name))
    over_indexed.sort(reverse=True)
    return over_indexed


def when_label(over_indexed):
    """Full when-phrase with lift percentages (top 3)."""
    return ", ".join(f"{name} {lift * 100:+.0f}%" for lift, name in over_indexed[:3])


def context_tag(over_indexed):
    """Compact situational tag (top 3 over-indexed bucket names) to enrich the archetype label so
    same-archetype shapes separate. Top-2 collides when two shapes share their strongest contexts
    (e.g. Raleigh L's two Uppercut Pulls are both down/offspeed) — the 3rd bucket pulls them apart
    ('...down, offspd, full' vs '...down, offspd, inside'). Shapes deployed identically will still
    tie here; their geometry (name_delta) and value (grade) are what separate them then."""
    return ", ".join(name for _, name in over_indexed[:3])


def matched_contrast(unit_swings, cluster):
    """Within-batter matched xRV contrast of `cluster` vs the primary (0), over strata where both
    appear. Returns (runs_per_100, matched_n, thin_flag)."""
    cluster_swings = unit_swings[unit_swings.cluster == cluster]
    primary_swings = unit_swings[unit_swings.cluster == 0]
    weighted_diff_sum = 0.0
    total_cluster_swings = 0.0
    matched_swing_count = 0.0
    stratum_count = 0
    for stratum, cluster_in_stratum in cluster_swings.groupby("strat"):
        primary_in_stratum = primary_swings[primary_swings.strat == stratum]
        if len(cluster_in_stratum) >= MIN_STRAT and len(primary_in_stratum) >= MIN_STRAT:
            mean_xrv_diff = cluster_in_stratum.xrv.mean() - primary_in_stratum.xrv.mean()
            weighted_diff_sum += mean_xrv_diff * len(cluster_in_stratum)
            total_cluster_swings += len(cluster_in_stratum)
            matched_swing_count += len(cluster_in_stratum)
            stratum_count += 1
    if total_cluster_swings == 0:
        return np.nan, 0, True
    runs_per_100 = (weighted_diff_sum / total_cluster_swings) * 100.0
    is_thin = matched_swing_count < MATCHED_THIN_N or stratum_count < 2
    return runs_per_100, int(matched_swing_count), is_thin


def build_cards(cluster_summary, archetypes, frame, buckets):
    # centroid-SD per feature (spread of shapes across the league) — the scale for "notable" deltas
    feature_sds = {col: archetypes[col].std() for col in GEO}
    cohort_mean = {col: archetypes[col].mean() for col in GEO}   # league reference for naming the primary
    centroids = archetypes.set_index(["batter_id", "batter_stand", "cluster"])

    mean_xrv_grade = frame.groupby(["batter_id", "batter_stand", "cluster"]).xrv_grade.mean()
    rows = []
    for (batter_id, batter_stand), unit_swings in frame.groupby(["batter_id", "batter_stand"], sort=False):
        primary_centroid = centroids.loc[(batter_id, batter_stand, 0)]
        cluster_usage = unit_swings.cluster.value_counts(normalize=True)
        for cluster_id in sorted(unit_swings.cluster.unique()):
            cluster_centroid = centroids.loc[(batter_id, batter_stand, cluster_id)]
            reference_centroid = cohort_mean if cluster_id == 0 else primary_centroid
            runs_per_100, matched_n, is_thin = (
                (np.nan, 0, False) if cluster_id == 0
                else matched_contrast(unit_swings, cluster_id)
            )
            over_indexed = over_index(unit_swings, cluster_id,
                                      float(cluster_usage.get(cluster_id, 0.0)), buckets)
            context_tag_str = context_tag(over_indexed)
            archetype = cluster_centroid["archetype_name"]
            rows.append({
                "batter_id": batter_id, "batter_stand": batter_stand,
                "label": cluster_centroid["label"], "cluster": int(cluster_id),
                "role": "primary" if cluster_id == 0 else "secondary",
                "archetype_name": archetype,
                "context_tag": context_tag_str,
                # cluster 0 is by definition the hitter's primary swing, so label it "Primary"
                # rather than by archetype+situation (archetype_name still holds the true archetype).
                "archetype_detailed": "Primary" if cluster_id == 0 else (
                    f"{archetype} · {context_tag_str}" if context_tag_str else archetype
                ),
                "arch_confidence": cluster_centroid["arch_confidence"],
                "usage": round(float(cluster_usage.get(cluster_id, 0.0)), 3),
                "n": int((unit_swings.cluster == cluster_id).sum()),
                "name_delta": name_delta(cluster_centroid, reference_centroid, feature_sds),
                "when_label": when_label(over_indexed),
                "grade": round(float(mean_xrv_grade.loc[(batter_id, batter_stand, cluster_id)]), 1),
                "matched_runs100": None if cluster_id == 0 else round(runs_per_100, 2),
                "matched_n": matched_n, "matched_thin": is_thin,
                "bat_speed": round(float(cluster_centroid["bat_speed"]), 1),
            })
    return pd.DataFrame(rows)


def write_catalog(cards):
    lines = ["# Swing ID cards (Layer 2) — worked examples\n"]
    for player_name in ["Aaron Judge", "Shohei Ohtani", "Luis Arraez"]:
        player_cards = cards[cards.label.str.startswith(player_name)].sort_values("cluster")
        if player_cards.empty:
            continue
        first_card = player_cards.iloc[0]
        lines.append(f"\n## {first_card.label}  ·  {int(player_cards.n.sum()):,} swings  ·  {len(player_cards)} shapes\n")
        for _, card_row in player_cards.iterrows():
            role_label = "* PRIMARY" if card_row.role == "primary" else "  secondary"
            reference_label = "vs league" if card_row.role == "primary" else "vs primary"
            archetype_label = card_row.archetype_name if card_row.role == "primary" else card_row.archetype_detailed
            lines.append(f"**{role_label} — {archetype_label}**  ({card_row.usage*100:.0f}% usage, {card_row.bat_speed:.0f} mph)")
            lines.append(f"  - shape: {card_row.name_delta} ({reference_label})")
            lines.append(f"  - when : {card_row.when_label or '(no strong over-index)'}")
            value_str = f"grade {card_row.grade:.0f}"
            if card_row.role == "secondary" and pd.notna(card_row.matched_runs100):
                thin_flag = " *(thin)*" if card_row.matched_thin else ""
                value_str += f"  ·  vs primary {card_row.matched_runs100:+.2f} runs/100 (n={card_row.matched_n}){thin_flag}"
            lines.append(f"  - value: {value_str}\n")
    (DATA / "shape_cards_catalog.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    cluster_summary, archetypes, cluster_assignments, xrv_swings, swings_model = load()
    frame, buckets = build_frame(cluster_assignments, xrv_swings, swings_model)
    print(f"Frame: {len(frame):,} swings with value + context")
    cards = build_cards(cluster_summary, archetypes, frame, buckets)
    cards.to_parquet(DATA / "shape_cards.parquet", index=False)
    print(f"Wrote shape_cards.parquet — {len(cards)} shape rows across "
          f"{cards.groupby(['batter_id','batter_stand']).ngroups} units")
    write_catalog(cards)


if __name__ == "__main__":
    main()
