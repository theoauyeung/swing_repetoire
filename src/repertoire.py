"""Measures how many different swings a hitter really has, and how different they are from each
other. A hitter with two swings that look nothing alike scores like one with four that all look
the same.

Input:  data/cluster_summary.parquet, data/swings_model.parquet
Outputs:
  data/repertoire_scores.parquet  one row per (batter, stand): expansiveness, repertoire_plus,
                               repertoire_pctile, per-feature raw spreads
  data/repertoire_catalog.md      human-readable leaderboards

Run:  python src/repertoire.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Frozen reference so the metric is comparable ACROSS seasons (pegged to 2024-25, like OPS+/wRC+
# peg to a fixed league baseline). First run builds it from the 2024-25 swings/cohort and commits
# it here; every later run (2026+ added) reuses these constants instead of re-baselining. Delete
# the file to re-peg. It holds only league-level aggregates (SDs, mean/SD, a percentile grid) — no
# athlete PII — so unlike data/ it lives in-repo and is committed.
REF_PATH = ROOT / "src" / "repertoire_reference.json"
REF_SEASONS = [2024, 2025]

# 5 shape features. cluster_summary stores centroids as `{col}_mean`; swings_model has the raw
FEATURES = ["swing_path_tilt", "swing_length", "bat_speed",
            "vert_attack_angle", "horz_attack_angle"]
UNITS = {"swing_path_tilt": "deg", "swing_length": "ft", "bat_speed": "mph",
         "vert_attack_angle": "deg", "horz_attack_angle": "deg"}
KEY = ["batter_id", "batter_stand"]


def mean_pairwise_dist(centroids, weights, feature_sds):
    """Usage-weighted mean pairwise Euclidean distance between a unit's cluster centroids (each
    feature standardized by the cohort swing-level SD), plus the per-feature mean pairwise gap in
    raw units. This is the count-BLIND spread term; main() multiplies it by effective_shapes."""
    num_clusters    = len(centroids)
    per_feature_gaps = np.zeros(len(feature_sds))
    if num_clusters < 2:
        return 0.0, per_feature_gaps

    standardized_centroids = centroids / feature_sds

    weighted_distance_sum = 0.0
    pair_weight_sum       = 0.0
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            pair_weight = weights[i] * weights[j]
            z_diff      = standardized_centroids[i] - standardized_centroids[j]
            euclidean_distance = float(np.sqrt((z_diff ** 2).sum()))
            raw_gap     = np.abs(centroids[i] - centroids[j])
            weighted_distance_sum += pair_weight * euclidean_distance
            per_feature_gaps      += pair_weight * raw_gap
            pair_weight_sum       += pair_weight

    if pair_weight_sum <= 0:
        return 0.0, per_feature_gaps
    return weighted_distance_sum / pair_weight_sum, per_feature_gaps / pair_weight_sum


def compute_units(cluster_summary, centroid_cols, feature_sds):
    """One row per (batter, stand): expansiveness (count-aware width) + per-feature raw spreads."""
    rows = []
    for (batter_id, stand), group in cluster_summary.groupby(KEY, sort=False):
        group = group.sort_values("cluster")
        centroids = group[centroid_cols].to_numpy(float)
        weights   = group["weight"].to_numpy(float)

        mean_distance, per_feature_gap = mean_pairwise_dist(centroids, weights, feature_sds)
        effective_shape_count = 1.0 / float(np.square(weights).sum())  # inverse-Simpson; k=1 -> 1
        # sqrt tempers the count term so spread and count drive the ranking ~equally
        # (effective_shape_count^1 made count 84% of it); k=1 -> 0 because mean_distance=0
        expansiveness = mean_distance * np.sqrt(effective_shape_count)

        row = {
            "batter_id":         batter_id,
            "batter_stand":      stand,
            "label":             group["label"].iloc[0],
            "k":                 len(group),
            "effective_shapes":  round(effective_shape_count, 2),
            "n_swings":          int(group["n"].sum()),
            "mean_pairwise_dist": round(mean_distance, 4),
            "expansiveness":     round(expansiveness, 4),
        }
        for feature, gap_value in zip(FEATURES, per_feature_gap):
            row[f"spread_{feature}"] = round(gap_value, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_reference(cluster_summary, centroid_cols, swings):
    """Return (feature_sds, ref) using the frozen 2024-25 baseline. Builds + commits it on first run.

    Pegging the SCALE (feature SDs + the z mean/SD + the percentile grid) to a fixed 2024-25 baseline
    is what makes repertoire_plus / repertoire_pctile comparable across seasons — without it, "50 =
    average" and the percentile silently re-reference every time the cohort changes. (Caveat: the
    cluster centroids feeding expansiveness are still pooled over whatever seasons are clustered, so
    a true per-season cross-season plot also needs per-season centroids — separate future work; this
    step removes the scale drift, not the centroid pooling.)"""
    if REF_PATH.exists():
        ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
        feature_sds = np.array([ref["feature_sd"][feature] for feature in FEATURES], float)
        return feature_sds, ref

    ref_swings  = swings[swings["game_year"].isin(REF_SEASONS)]
    feature_sds = ref_swings[FEATURES].std().to_numpy(float)

    ref_expansiveness = compute_units(cluster_summary, centroid_cols, feature_sds)["expansiveness"]
    feature_sd_dict   = {feature: float(sd) for feature, sd in zip(FEATURES, feature_sds)}
    expansiveness_sorted_list = [round(float(value), 4) for value in sorted(ref_expansiveness)]

    ref = {
        "reference_seasons":    REF_SEASONS,
        "n_reference_units":    int(len(ref_expansiveness)),
        "feature_sd":           feature_sd_dict,
        "expansiveness_mean":   float(ref_expansiveness.mean()),
        "expansiveness_std":    float(ref_expansiveness.std()),
        "expansiveness_sorted": expansiveness_sorted_list,
    }
    REF_PATH.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(f"Built + froze repertoire reference -> {REF_PATH} (seasons {REF_SEASONS}, {len(ref_expansiveness)} units)")
    return feature_sds, ref


def main():
    cluster_summary = pd.read_parquet(DATA / "cluster_summary.parquet")
    swings          = pd.read_parquet(DATA / "swings_model.parquet", columns=FEATURES + ["game_year"])
    centroid_cols   = [f"{feature}_mean" for feature in FEATURES]
    feature_sds, ref = resolve_reference(cluster_summary, centroid_cols, swings)

    print(f"Frozen 2024-25 reference: SD per feature + expansiveness mean {ref['expansiveness_mean']:.3f} "
          f"/ SD {ref['expansiveness_std']:.3f} over {ref['n_reference_units']} units")
    for feature, sd_value in zip(FEATURES, feature_sds):
        print(f"  {feature:22s} {sd_value:6.3f} {UNITS[feature]}")

    repertoire = compute_units(cluster_summary, centroid_cols, feature_sds)

    # Scale + rank against the FROZEN 2024-25 reference (not the current cohort), so both are
    # season-stable. repertoire_plus: 50 + 10·z on the frozen mean/SD. pctile: position in the
    # frozen expansiveness distribution.
    z_scores = (repertoire["expansiveness"] - ref["expansiveness_mean"]) / ref["expansiveness_std"]
    repertoire["repertoire_plus"] = (50 + 10 * z_scores).clip(0, 100).round(1)

    ref_sorted_array = np.array(ref["expansiveness_sorted"], float)
    raw_percentile_ranks = (
        np.searchsorted(ref_sorted_array, repertoire["expansiveness"].to_numpy(), side="right")
        / len(ref_sorted_array) * 100
    )
    repertoire["repertoire_pctile"] = raw_percentile_ranks.round(1)

    repertoire = repertoire.sort_values("repertoire_plus", ascending=False).reset_index(drop=True)
    repertoire.to_parquet(DATA / "repertoire_scores.parquet", index=False)
    write_catalog(repertoire, feature_sds)
    print(f"\nWrote repertoire_scores.parquet ({len(repertoire)} units) + repertoire_catalog.md")


def write_catalog(repertoire, feature_sds):
    spread_cols = [f"spread_{feature}" for feature in FEATURES]
    lines = []
    lines.append("# Swing Repertoire+ catalog — count-aware repertoire width (geometry only)\n")
    lines.append("Repertoire+ = **usage-weighted mean pairwise centroid distance × √(effective number of "
                 "shapes)** (`1/Σweight²`, inverse-Simpson), in the 5-feature shape space (bat speed + the "
                 "four angle/length features) standardized by cohort swing-level SD so it is comparable "
                 "across hitters. It rewards BOTH how different a hitter's shapes are AND how many "
                 "(effective) shapes they carry — a 6-shape hitter is functionally wider than a 2-shape one "
                 "even when the 2 are far apart — but the **√** on the count term keeps the two balanced "
                 "(each drives ~half the ranking), so a genuinely wide 2-shape hitter can still out-rank a "
                 "mediocre 5-shape one. **It is purely descriptive — it says nothing about whether "
                 "the shapes are good or valuable.** k=1 (single-shape) hitters score the floor (0). The "
                 "scale (feature SDs + the `50+10·z` constants + the percentile grid) is **pegged to the "
                 "2024-25 baseline** (`src/repertoire_reference.json`) so repertoire_plus / pctile stay "
                 "comparable when later seasons are added.\n")
    lines.append(f"- Cohort: **{len(repertoire)} (batter, stand) units**")
    lines.append(f"- **Lead with `repertoire_pctile` (0-100 rank).** {int((repertoire.k == 1).sum())} single-shape "
                 f"units (24%) pile up at the 0-spread floor, dragging the Repertoire+ mean below the "
                 f"multi-shape median, so `repertoire_plus`'s '50 = average' is skewed by that mass. The "
                 f"percentile is robust to it; Repertoire+ is a monotone transform of the same ranking. "
                 f"Repertoire+ uses the same 0-100 / 50-average scale as Swing+ (50 + 10·z, clipped).")
    lines.append(f"- Expansiveness (mean pairwise dist × √effective shapes) — "
                 f"mean {repertoire.expansiveness.mean():.2f}, median {repertoire.expansiveness.median():.2f}, "
                 f"max {repertoire.expansiveness.max():.2f}")
    lines.append("- **What drives the width:** `swing_length` + the two attack angles dominate; "
                 "`bat_speed` and `swing_path_tilt` contribute least. `horz_attack_angle` is the most "
                 "pitch-reactive feature (batter ICC 0.054), so a horz-driven wide repertoire partly "
                 "reflects pitch-location variety, not genuine swing change.\n")

    sd_table = pd.DataFrame({"feature": FEATURES, "cohort_SD": feature_sds.round(3),
                              "unit": [UNITS[feature] for feature in FEATURES]})
    lines.append("## Cohort swing-level SD (the per-feature scale used to standardize)")
    lines.append(sd_table.to_markdown(index=False) + "\n")

    display_cols = ["label", "n_swings", "k", "effective_shapes", "repertoire_pctile", "repertoire_plus",
                    "expansiveness", "mean_pairwise_dist"] + spread_cols
    lines.append("## Widest repertoires (most expansive repertoires)")
    lines.append(repertoire.head(15)[display_cols].to_markdown(index=False) + "\n")

    multi_shape_units = repertoire[repertoire.k >= 2].sort_values("repertoire_plus")
    lines.append("## Narrowest repertoires (>=2 shapes, least expansive)")
    lines.append(multi_shape_units.head(15)[display_cols].to_markdown(index=False) + "\n")

    lines.append("## Widest on each single axis (usage-weighted mean pairwise gap, raw units)")
    for feature in FEATURES:
        spread_col = f"spread_{feature}"
        top_five   = repertoire[repertoire.k >= 2].sort_values(spread_col, ascending=False).head(5)
        entry_parts = [f"{row.label} ({getattr(row, spread_col):.1f} {UNITS[feature]})"
                       for row in top_five.itertuples()]
        entry_list  = ", ".join(entry_parts)
        lines.append(f"- **{feature}**: {entry_list}")
    lines.append("")

    catalog_text = "\n".join(lines)
    (DATA / "repertoire_catalog.md").write_text(catalog_text, encoding="utf-8")
    print(catalog_text)


if __name__ == "__main__":
    main()
