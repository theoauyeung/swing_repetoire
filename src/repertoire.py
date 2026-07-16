""" Swing Repertoire+ — a purely descriptive measure of how EXPANSIVE a hitter's swing
repertoire is in shape space. Bigger gaps between a unit's cluster centroids (in bat speed
and the angle features) => a wider repertoire. This is geometry only: it says nothing about
whether the shapes are good, valuable, or well-deployed — a wide repertoire and a narrow one are
graded purely on spread.


Definition (per (batter, stand) unit) — a count-aware functional-diversity measure:
  - standardize each of the 5 shape features by the cohort swing-level SD,
  - mean_pairwise_dist = USAGE-WEIGHTED MEAN pairwise Euclidean distance between the unit's
    cluster centroids in that league-z space (pair weight = weight_i * weight_j),
  - effective_shapes = 1 / sum(weight_i^2)  (inverse-Simpson: the usage-EFFECTIVE number of
    shapes, so a rarely-used emergency shape barely counts; k even-weight => effective ~ k),
  - expansiveness = mean_pairwise_dist * sqrt(effective_shapes)  — rewards BOTH how different the
    shapes are AND how many (effective) shapes a hitter carries; k=1 => 0. (Mean pairwise
    distance alone is count-blind — it's the average dissimilarity of two random swings, so a
    hitter with 2 extreme shapes beat one with 6 moderate ones. But `* effective_shapes` overshot
    the other way — count then drove ~84% of the ranking and shape-count groups barely overlapped.
    The `sqrt` tempers the count term so spread and count contribute ~equally: a genuinely wide
    2-shape repertoire can still out-rank a mediocre 5-shape one.)
  - Repertoire+ = 50 + 10 * z(expansiveness), clipped to [0, 100] (50 = league-average width;
    same scale as Swing+), plus a 0-100 percentile.
  - per-feature spread breakdown: usage-weighted mean pairwise |centroid gap| for each feature,
    in raw units (mph / degrees / ft), so you can see WHICH axis makes a hitter expansive.

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


def mean_pairwise_dist(centroids, weights, sd):
    """Usage-weighted mean pairwise Euclidean distance between a unit's cluster centroids (each
    feature standardized by the cohort swing-level SD), plus the per-feature mean pairwise gap in
    raw units. This is the count-BLIND spread term; main() multiplies it by effective_shapes."""
    k = len(centroids)
    per_feat = np.zeros(len(sd))
    if k < 2:
        return 0.0, per_feat
    Z = centroids / sd                       # league-z centroids
    num = den = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            w = weights[i] * weights[j]
            num += w * float(np.sqrt(((Z[i] - Z[j]) ** 2).sum()))
            per_feat += w * np.abs(centroids[i] - centroids[j])
            den += w
    return (num / den, per_feat / den) if den > 0 else (0.0, per_feat)


def compute_units(summary, cen_cols, sd):
    """One row per (batter, stand): expansiveness (count-aware width) + per-feature raw spreads."""
    rows = []
    for (bid, stand), g in summary.groupby(KEY, sort=False):
        g = g.sort_values("cluster")
        centroids = g[cen_cols].to_numpy(float)
        weights = g["weight"].to_numpy(float)
        mpd, per_feat = mean_pairwise_dist(centroids, weights, sd)
        eff = 1.0 / float(np.square(weights).sum())   # effective # shapes (inverse-Simpson); k=1 -> 1
        width = mpd * np.sqrt(eff)                     # count-aware width; sqrt tempers the count term
                                                      # so spread & count drive the ranking ~equally
                                                      # (eff^1 made count 84% of it); k=1 -> 0 (mpd=0)
        row = {"batter_id": bid, "batter_stand": stand,
               "label": g["label"].iloc[0], "k": len(g), "effective_shapes": round(eff, 2),
               "n_swings": int(g["n"].sum()), "mean_pairwise_dist": round(mpd, 4),
               "expansiveness": round(width, 4)}
        row.update({f"spread_{c}": round(v, 2) for c, v in zip(FEATURES, per_feat)})
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_reference(summary, cen_cols, swings):
    """Return (sd, ref) using the frozen 2024-25 baseline. Builds + commits it on first run.

    Pegging the SCALE (feature SDs + the z mean/SD + the percentile grid) to a fixed 2024-25 baseline
    is what makes repertoire_plus / repertoire_pctile comparable across seasons — without it, "50 =
    average" and the percentile silently re-reference every time the cohort changes. (Caveat: the
    cluster centroids feeding expansiveness are still pooled over whatever seasons are clustered, so
    a true per-season cross-season plot also needs per-season centroids — separate future work; this
    step removes the scale drift, not the centroid pooling.)"""
    if REF_PATH.exists():
        ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
        sd = np.array([ref["feature_sd"][f] for f in FEATURES], float)
        return sd, ref

    ref_swings = swings[swings["game_year"].isin(REF_SEASONS)]
    sd = ref_swings[FEATURES].std().to_numpy(float)
    exp = compute_units(summary, cen_cols, sd)["expansiveness"]
    ref = {"reference_seasons": REF_SEASONS,
           "n_reference_units": int(len(exp)),
           "feature_sd": {f: float(s) for f, s in zip(FEATURES, sd)},
           "expansiveness_mean": float(exp.mean()),
           "expansiveness_std": float(exp.std()),
           "expansiveness_sorted": [round(float(v), 4) for v in sorted(exp)]}
    REF_PATH.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(f"Built + froze repertoire reference -> {REF_PATH} (seasons {REF_SEASONS}, {len(exp)} units)")
    return sd, ref


def main():
    summary = pd.read_parquet(DATA / "cluster_summary.parquet")
    swings = pd.read_parquet(DATA / "swings_model.parquet", columns=FEATURES + ["game_year"])
    cen_cols = [f"{c}_mean" for c in FEATURES]
    sd, ref = resolve_reference(summary, cen_cols, swings)
    print(f"Frozen 2024-25 reference: SD per feature + expansiveness mean {ref['expansiveness_mean']:.3f} "
          f"/ SD {ref['expansiveness_std']:.3f} over {ref['n_reference_units']} units")
    for f, s in zip(FEATURES, sd):
        print(f"  {f:22s} {s:6.3f} {UNITS[f]}")

    repertoire = compute_units(summary, cen_cols, sd)
    # Scale + rank against the FROZEN 2024-25 reference (not the current cohort), so both are
    # season-stable. repertoire_plus: 50 + 10·z on the frozen mean/SD. pctile: position in the
    # frozen expansiveness distribution.
    z = (repertoire["expansiveness"] - ref["expansiveness_mean"]) / ref["expansiveness_std"]
    repertoire["repertoire_plus"] = (50 + 10 * z).clip(0, 100).round(1)
    ref_sorted = np.array(ref["expansiveness_sorted"], float)
    repertoire["repertoire_pctile"] = (
        np.searchsorted(ref_sorted, repertoire["expansiveness"].to_numpy(), side="right")
        / len(ref_sorted) * 100).round(1)

    repertoire = repertoire.sort_values("repertoire_plus", ascending=False).reset_index(drop=True)
    repertoire.to_parquet(DATA / "repertoire_scores.parquet", index=False)
    write_catalog(repertoire, sd)
    print(f"\nWrote repertoire_scores.parquet ({len(repertoire)} units) + repertoire_catalog.md")


def write_catalog(a, sd):
    spread_cols = [f"spread_{c}" for c in FEATURES]
    L, w = [], None
    out = L.append
    out("# Swing Repertoire+ catalog — count-aware repertoire width (geometry only)\n")
    out("Repertoire+ = **usage-weighted mean pairwise centroid distance × √(effective number of "
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
    out(f"- Cohort: **{len(a)} (batter, stand) units**")
    out(f"- **Lead with `repertoire_pctile` (0-100 rank).** {int((a.k == 1).sum())} single-shape "
        f"units (24%) pile up at the 0-spread floor, dragging the Repertoire+ mean below the "
        f"multi-shape median, so `repertoire_plus`'s '50 = average' is skewed by that mass. The "
        f"percentile is robust to it; Repertoire+ is a monotone transform of the same ranking. "
        f"Repertoire+ uses the same 0-100 / 50-average scale as Swing+ (50 + 10·z, clipped).")
    out(f"- Expansiveness (mean pairwise dist × √effective shapes) — "
        f"mean {a.expansiveness.mean():.2f}, median {a.expansiveness.median():.2f}, "
        f"max {a.expansiveness.max():.2f}")
    out("- **What drives the width:** `swing_length` + the two attack angles dominate; "
        "`bat_speed` and `swing_path_tilt` contribute least. `horz_attack_angle` is the most "
        "pitch-reactive feature (batter ICC 0.054), so a horz-driven wide repertoire partly "
        "reflects pitch-location variety, not genuine swing change.\n")

    out("## Cohort swing-level SD (the per-feature scale used to standardize)")
    out(pd.DataFrame({"feature": FEATURES, "cohort_SD": sd.round(3),
                      "unit": [UNITS[f] for f in FEATURES]}).to_markdown(index=False) + "\n")

    show = ["label", "n_swings", "k", "effective_shapes", "repertoire_pctile", "repertoire_plus",
            "expansiveness", "mean_pairwise_dist"] + spread_cols
    out("## Widest repertoires (most expansive repertoires)")
    out(a.head(15)[show].to_markdown(index=False) + "\n")

    out("## Narrowest repertoires (>=2 shapes, least expansive)")
    multi = a[a.k >= 2].sort_values("repertoire_plus")
    out(multi.head(15)[show].to_markdown(index=False) + "\n")

    out("## Widest on each single axis (usage-weighted mean pairwise gap, raw units)")
    for c in FEATURES:
        col = f"spread_{c}"
        top = a[a.k >= 2].sort_values(col, ascending=False).head(5)
        pairs = ", ".join(f"{r.label} ({getattr(r, col):.1f} {UNITS[c]})" for r in top.itertuples())
        out(f"- **{c}**: {pairs}")
    out("")

    (DATA / "repertoire_catalog.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
