"""Gives every swing a name a coach would use — "Uppercut Pull", "Level Oppo" — instead of a
number. The clustering step only knows "this hitter's cluster 1", which means nothing across
hitters; this puts all of them on one shared vocabulary.

Input:  data/cluster_assignments.parquet, data/swings_model.parquet, data/cluster_summary.parquet
Outputs:
  data/shape_archetypes.parquet   one row per (batter, stand, cluster): archetype id/name,
                                   assignment confidence, pull-frame centroid, n, usage weight
  data/archetype_lexicon.parquet  one row per archetype: id, name, raw-unit centroid, n_shapes
  data/archetype_lexicon.md       human-readable catalog

Run:  python src/interpret.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Archetypes are DEFINED by the 4 pure-geometry features. bat_speed is reported as a descriptor
# only (its ICC 0.126 is "state not trait" per research-design.md, so cross-batter it drags the
# vocabulary toward effort/state and collapses the geometry grid — verified: including it splits
# off a degenerate low-effort bin). The per-batter clustering (cluster.py) still uses all 5; this
# is a naming overlay only.
GEO_FEAT = ["swing_path_tilt", "swing_length", "vert_attack_angle", "horz_attack_angle_pull"]
DESCRIPTOR = "bat_speed"
FEAT = GEO_FEAT + [DESCRIPTOR]          # loaded per unit-cluster; only GEO_FEAT defines archetypes
SHORT = {"swing_path_tilt": "tilt", "swing_length": "len", "bat_speed": "bat_speed",
         "vert_attack_angle": "vaa", "horz_attack_angle_pull": "haa_pull"}
K_ARCH = 3            # Three-way vocabulary: Level Oppo / Level Center / Uppercut Pull, the honest
                      # carve of the level-oppo <-> uppercut-pull diagonal. NOTE this is a deliberate
                      # interpretability override of the BIC minimum, which is 2 on the current
                      # MERGE_SEP=1.75 pool (BIC 13188.8 at K=2 vs 13222.2 at K=3 — a slim margin). We
                      # keep 3 because the middle Level-Center band is a real, useful distinction for
                      # readers; K=3 is a shallow local BIC bump, not a degenerate split. At K=3 the
                      # two level components differ mainly in horz attack (haa_pull ~-5.6 vs ~-7.6), so
                      # HAA_OPPO below is tuned to -6.5 to name them apart (Center vs Oppo). Was BIC-min
                      # 3 under MERGE_SEP=2.0; the recluster moved the raw BIC-min to 2.
SEED = 7
N_INIT = 25           # 8 restarts can settle in a worse local optimum; 25 reliably reaches the
                      # global one. The archetype *names* are seed-invariant; only a few
                      # boundary shapes move (seed agreement ~0.93).

# name thresholds on an archetype's raw-unit centroid (degrees)
VAA_FLAT, VAA_STEEP = 3.0, 13.0        # vert_attack_angle: <flat / level / >=uppercut (~6deg = level)
HAA_OPPO, HAA_PULL = -6.5, 5.0         # horz_attack_angle_pull: <oppo / center / >=pull (+ = pull, both hands).
                                       # OPPO boundary moved -5.0 -> -6.5 (2026-07-13) to split the two
                                       # K=3 level components (haa_pull ~-5.6 Center vs ~-7.6 Oppo) that
                                       # collided under the MERGE_SEP=1.75 pool; restores Level Center.


def load_centroids():
    """Per unit-cluster centroid in the pull frame + n + usage weight + display label."""
    cluster_assignments = pd.read_parquet(DATA / "cluster_assignments.parquet",
                                          columns=["play_id", "batter_id", "batter_stand", "cluster"])
    swings_model        = pd.read_parquet(DATA / "swings_model.parquet", columns=["play_id"] + FEAT)
    cluster_summary     = pd.read_parquet(DATA / "cluster_summary.parquet",
                                          columns=["batter_id", "batter_stand", "cluster", "label", "weight"])

    swings_with_assignments = cluster_assignments.merge(swings_model, on="play_id", how="left")

    # Aggregate to one centroid per (batter, stand, cluster) then attach display label and usage weight
    feature_agg = {"n": ("play_id", "size"), **{feature: (feature, "mean") for feature in FEAT}}
    centroids = (swings_with_assignments
                 .groupby(["batter_id", "batter_stand", "cluster"])
                 .agg(**feature_agg)
                 .reset_index()
                 .merge(cluster_summary, on=["batter_id", "batter_stand", "cluster"], how="left"))
    return centroids


def archetype_name(centroid):
    """Reproducible {vertical} {direction} name from a raw-unit centroid dict."""
    vertical_attack_angle       = centroid["vert_attack_angle"]
    horizontal_attack_angle_pull = centroid["horz_attack_angle_pull"]
    vertical_label  = "Flat" if vertical_attack_angle < VAA_FLAT else ("Uppercut" if vertical_attack_angle >= VAA_STEEP else "Level")
    direction_label = "Oppo" if horizontal_attack_angle_pull < HAA_OPPO else ("Pull" if horizontal_attack_angle_pull >= HAA_PULL else "Center")
    return f"{vertical_label} {direction_label}"


def _build_lexicon_row(archetype_index, remapped_archetype_id, raw_means, component_assignments, centroids):
    """Build one row of the archetype lexicon dict from a single GMM component."""
    centroid_dict = dict(zip(GEO_FEAT, raw_means[archetype_index]))
    member_mask   = component_assignments == archetype_index
    return {
        "archetype":      remapped_archetype_id,
        "archetype_name": archetype_name(centroid_dict),
        "n_shapes":       int(member_mask.sum()),
        **{SHORT[feature]: round(centroid_dict[feature], 2) for feature in GEO_FEAT},
        # bat_speed descriptor = mean over the member shapes' centroids
        "bat_speed": round(centroids.loc[member_mask, DESCRIPTOR].mean(), 2),
    }


def fit_archetype(centroids):
    """League-standardize the geometry-centroid pool, fit the archetype GMM, tag every
    unit-cluster. bat_speed is not fitted — reported per archetype as a descriptor.
    Returns (centroids + archetype cols, lexicon dataframe)."""
    feature_means = centroids[GEO_FEAT].mean()
    feature_stds  = centroids[GEO_FEAT].std()
    standardized_geometry = ((centroids[GEO_FEAT] - feature_means) / feature_stds).to_numpy()

    gmm = GaussianMixture(K_ARCH, covariance_type="full", n_init=N_INIT,
                          reg_covar=1e-4, random_state=SEED).fit(standardized_geometry)
    responsibilities = gmm.predict_proba(standardized_geometry)
    raw_means = gmm.means_ * feature_stds.values + feature_means.values  # de-standardize geometry centroids

    # relabel archetypes by prevalence (0 = most common) for stable, meaningful ids
    component_assignments = responsibilities.argmax(axis=1)
    prevalence_order      = pd.Series(component_assignments).value_counts().index.to_numpy()
    remap = np.empty(K_ARCH, dtype=int)
    remap[prevalence_order] = np.arange(K_ARCH)

    lexicon_rows = [
        _build_lexicon_row(archetype_index, remap[archetype_index], raw_means, component_assignments, centroids)
        for archetype_index in range(K_ARCH)
    ]
    lexicon = pd.DataFrame(lexicon_rows).sort_values("archetype").reset_index(drop=True)

    archetype_names = lexicon["archetype_name"]
    if archetype_names.duplicated().any():
        raise ValueError(f"archetype names collide at K_ARCH={K_ARCH}: {archetype_names.tolist()} — "
                         "retune name thresholds or K_ARCH")

    tagged_centroids = centroids.copy()
    tagged_centroids["archetype"]       = remap[component_assignments]
    tagged_centroids["arch_confidence"] = responsibilities.max(axis=1).round(3)
    tagged_centroids = tagged_centroids.merge(lexicon[["archetype", "archetype_name"]], on="archetype", how="left")
    return tagged_centroids, lexicon


def write_catalog(shapes, lexicon):
    lines = []
    lines.append("# Swing-shape archetype lexicon (Layer 1)\n")
    lines.append(f"- {len(shapes):,} unit-clusters across {shapes.groupby(['batter_id','batter_stand']).ngroups} "
                 f"(batter, stand) units, mapped onto **{K_ARCH} league archetypes** (pull frame).")
    lines.append(f"- Naming is algorithmic from the archetype centroid: vertical "
                 f"(Flat < {VAA_FLAT}° VAA <= Level < {VAA_STEEP}° <= Uppercut) x direction "
                 f"(Oppo < {HAA_OPPO}° HAA_pull < Center < {HAA_PULL}° <= Pull).")
    lines.append(f"- Assignment confidence (max archetype responsibility) — median "
                 f"{shapes.arch_confidence.median():.2f}, share > 0.8: {(shapes.arch_confidence > 0.8).mean():.0%}\n")

    lines.append("## The lexicon (raw-unit centroids)")
    lines.append(lexicon[["archetype", "archetype_name", "n_shapes", "tilt", "len", "bat_speed", "vaa", "haa_pull"]]
                 .to_markdown(index=False) + "\n")

    lines.append("## Exemplars per archetype (largest shapes)")
    for _, archetype_row in lexicon.iterrows():
        member_shapes = shapes[shapes.archetype == archetype_row.archetype]
        top_exemplars = member_shapes.sort_values("n", ascending=False).head(6)["label"].tolist()
        lines.append(f"- **{archetype_row.archetype_name}** ({archetype_row.n_shapes} shapes): "
                     + ", ".join(top_exemplars))
    lines.append("")

    catalog_text = "\n".join(lines)
    (DATA / "archetype_lexicon.md").write_text(catalog_text, encoding="utf-8")
    print(catalog_text)


def main():
    centroids = load_centroids()
    print(f"Loaded {len(centroids):,} unit-cluster centroids (pull frame)")
    tagged_centroids, lexicon = fit_archetype(centroids)

    output_columns = (["batter_id", "batter_stand", "cluster", "label", "archetype", "archetype_name",
                        "arch_confidence", "n", "weight"] + FEAT)
    tagged_centroids[output_columns].to_parquet(DATA / "shape_archetypes.parquet", index=False)
    lexicon.to_parquet(DATA / "archetype_lexicon.parquet", index=False)
    write_catalog(tagged_centroids, lexicon)
    print(f"\nWrote shape_archetypes / archetype_lexicon (.parquet) + archetype_lexicon.md")


if __name__ == "__main__":
    main()
