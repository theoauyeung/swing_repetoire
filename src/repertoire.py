""" Swing Repertoire+ — a purely descriptive measure of how EXPANSIVE a hitter's swing
repertoire is in shape space. Bigger gaps between a unit's cluster centroids (in bat speed
and the angle features) => a wider repertoire. This is geometry only: it says nothing about
whether the shapes are good, valuable, or well-deployed — a wide repertoire and a narrow one are
graded purely on spread.


Definition (per (batter, stand) unit):
  - standardize each of the 5 shape features by the cohort swing-level SD,
  - expansiveness = USAGE-WEIGHTED MEAN pairwise Euclidean distance between the unit's cluster
    centroids in that league-z space (pair weight = weight_i * weight_j); k=1 => 0,
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 5 shape features. cluster_summary stores centroids as `{col}_mean`; swings_model has the raw
FEATURES = ["swing_path_tilt", "swing_length", "bat_speed",
            "vert_attack_angle", "horz_attack_angle"]
UNITS = {"swing_path_tilt": "deg", "swing_length": "ft", "bat_speed": "mph",
         "vert_attack_angle": "deg", "horz_attack_angle": "deg"}
KEY = ["batter_id", "batter_stand"]


def expansiveness(centroids, weights, sd):
    """Usage-weighted mean pairwise Euclidean distance between a unit's cluster centroids, with
    each feature standardized by the cohort swing-level SD"""
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


def main():
    summary = pd.read_parquet(DATA / "cluster_summary.parquet")
    swings = pd.read_parquet(DATA / "swings_model.parquet", columns=FEATURES)
    sd = swings[FEATURES].std().to_numpy(float)
    print("Cohort swing-level SD per feature:")
    for f, s in zip(FEATURES, sd):
        print(f"  {f:22s} {s:6.3f} {UNITS[f]}")

    cen_cols = [f"{c}_mean" for c in FEATURES]
    rows = []
    for (bid, stand), g in summary.groupby(KEY, sort=False):
        g = g.sort_values("cluster")
        centroids = g[cen_cols].to_numpy(float)
        weights = g["weight"].to_numpy(float)
        exp, per_feat = expansiveness(centroids, weights, sd)
        row = {"batter_id": bid, "batter_stand": stand,
               "label": g["label"].iloc[0], "k": len(g),
               "n_swings": int(g["n"].sum()), "expansiveness": round(exp, 4)}
        row.update({f"spread_{c}": round(v, 2) for c, v in zip(FEATURES, per_feat)})
        rows.append(row)

    repertoire = pd.DataFrame(rows)
    z = (repertoire["expansiveness"] - repertoire["expansiveness"].mean()) / repertoire["expansiveness"].std()
    repertoire["repertoire_plus"] = (50 + 10 * z).clip(0, 100).round(1)
    repertoire["repertoire_pctile"] = (repertoire["expansiveness"].rank(pct=True) * 100).round(1)

    repertoire = repertoire.sort_values("repertoire_plus", ascending=False).reset_index(drop=True)
    repertoire.to_parquet(DATA / "repertoire_scores.parquet", index=False)
    write_catalog(repertoire, sd)
    print(f"\nWrote repertoire_scores.parquet ({len(repertoire)} units) + repertoire_catalog.md")


def write_catalog(a, sd):
    spread_cols = [f"spread_{c}" for c in FEATURES]
    L, w = [], None
    out = L.append
    out("# Swing Repertoire+ catalog — repertoire expansiveness (geometry only)\n")
    out("Repertoire+ measures how SPREAD OUT a hitter's swing-shape clusters are in the 5-feature "
        "shape space (bat speed + the four angle/length features), standardized by cohort "
        "swing-level SD so it is comparable across hitters. **It is purely descriptive — it "
        "says nothing about whether the shapes are good or valuable.** k=1 (single-shape) "
        "hitters score the floor (0 spread).\n")
    out(f"- Cohort: **{len(a)} (batter, stand) units**")
    out(f"- **Lead with `repertoire_pctile` (0-100 rank).** {int((a.k == 1).sum())} single-shape "
        f"units (24%) pile up at the 0-spread floor, dragging the Repertoire+ mean below the "
        f"multi-shape median, so `repertoire_plus`'s '50 = average' is skewed by that mass. The "
        f"percentile is robust to it; Repertoire+ is a monotone transform of the same ranking. "
        f"Repertoire+ uses the same 0-100 / 50-average scale as Swing+ (50 + 10·z, clipped).")
    out(f"- Usage-weighted mean pairwise centroid distance (league-SD units) — "
        f"mean {a.expansiveness.mean():.2f}, median {a.expansiveness.median():.2f}, "
        f"max {a.expansiveness.max():.2f}")
    out("- **What drives the width:** `swing_length` + the two attack angles dominate; "
        "`bat_speed` and `swing_path_tilt` contribute least. `horz_attack_angle` is the most "
        "pitch-reactive feature (batter ICC 0.054), so a horz-driven wide repertoire partly "
        "reflects pitch-location variety, not genuine swing change.\n")

    out("## Cohort swing-level SD (the per-feature scale used to standardize)")
    out(pd.DataFrame({"feature": FEATURES, "cohort_SD": sd.round(3),
                      "unit": [UNITS[f] for f in FEATURES]}).to_markdown(index=False) + "\n")

    show = ["label", "n_swings", "k", "repertoire_pctile", "repertoire_plus", "expansiveness"] + spread_cols
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
