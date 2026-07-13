""" Layer 1 — league swing-shape archetype lexicon.

Per-(batter, stand) clusters are strictly non-comparable (cluster.py); "Cluster 1" is a bare
index. This builds a cross-unit *vocabulary* so every unit-cluster inherits a human-readable
archetype tag ("Uppercut Pull", "Flat Oppo") without making clusters the unit of cross-hitter
comparison. Design sanction: research-design.md Limitation #6 — a post-hoc global reference
embedding may *label* clusters, it does not merge them.

Method:
  - recompute each unit-cluster centroid in the 5-feature *pull frame* (handedness-mirrored
    horz_attack_angle) from cluster_assignments + swings_model, so a lefty's pull swing and a
    righty's pull swing land in the same archetype. (cluster_summary stores the raw, unmirrored
    horz angle, which would put L/R pull swings on opposite sides — hence the recompute.)
  - define archetypes on the 4 pure-geometry features only (tilt, length, VAA, HAA_pull);
    bat_speed is reported as a per-archetype descriptor, NOT a defining axis (it is state-not-trait
    cross-batter, ICC 0.126, and including it collapses the geometry grid into an effort bin),
  - league-standardize the 1,592-centroid pool (z-score each feature across shapes, unweighted:
    each shape is one observation so rare protective shapes get equal say in the vocabulary),
  - fit a full-cov GMM at K_ARCH=3 — Level Oppo / Level Center / Uppercut Pull, the honest carve of
    the level-oppo <-> uppercut-pull diagonal (uppercut swings are pull-side). On the MERGE_SEP=1.75
    pool the raw BIC minimum is 2, so K=3 is a deliberate interpretability override (kept for the
    useful middle band; see K_ARCH note). The two level components separate on horz attack, so the
    OPPO naming boundary is tuned to -6.5 to name them apart,
  - name each archetype algorithmically from its centroid (vertical x direction), so names are
    reproducible regardless of GMM component ordering, and assert they come out unique.

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
    ca = pd.read_parquet(DATA / "cluster_assignments.parquet",
                         columns=["play_id", "batter_id", "batter_stand", "cluster"])
    sm = pd.read_parquet(DATA / "swings_model.parquet", columns=["play_id"] + FEAT)
    lab = pd.read_parquet(DATA / "cluster_summary.parquet",
                          columns=["batter_id", "batter_stand", "cluster", "label", "weight"])
    df = ca.merge(sm, on="play_id", how="left")
    cent = (df.groupby(["batter_id", "batter_stand", "cluster"])
              .agg(n=("play_id", "size"), **{f: (f, "mean") for f in FEAT})
              .reset_index()
              .merge(lab, on=["batter_id", "batter_stand", "cluster"], how="left"))
    return cent


def archetype_name(centroid):
    """Reproducible {vertical} {direction} name from a raw-unit centroid dict."""
    vaa, haa = centroid["vert_attack_angle"], centroid["horz_attack_angle_pull"]
    vert = "Flat" if vaa < VAA_FLAT else ("Uppercut" if vaa >= VAA_STEEP else "Level")
    direction = "Oppo" if haa < HAA_OPPO else ("Pull" if haa >= HAA_PULL else "Center")
    return f"{vert} {direction}"


def fit_archetype(cent):
    """League-standardize the geometry-centroid pool, fit the archetype GMM, tag every
    unit-cluster. bat_speed is not fitted — reported per archetype as a descriptor.
    Returns (cent + archetype cols, lexicon dataframe)."""
    mu, sd = cent[GEO_FEAT].mean(), cent[GEO_FEAT].std()
    Z = ((cent[GEO_FEAT] - mu) / sd).to_numpy()
    gm = GaussianMixture(K_ARCH, covariance_type="full", n_init=N_INIT,
                         reg_covar=1e-4, random_state=SEED).fit(Z)
    resp = gm.predict_proba(Z)
    raw_means = gm.means_ * sd.values + mu.values        # de-standardize geometry centroids

    # relabel archetypes by prevalence (0 = most common) for stable, meaningful ids
    comp = resp.argmax(axis=1)
    order = pd.Series(comp).value_counts().index.to_numpy()
    remap = np.empty(K_ARCH, dtype=int)
    remap[order] = np.arange(K_ARCH)

    lex = []
    for a in range(K_ARCH):
        c = dict(zip(GEO_FEAT, raw_means[a]))
        lex.append({"archetype": remap[a], "archetype_name": archetype_name(c),
                    "n_shapes": int((comp == a).sum()),
                    **{SHORT[f]: round(c[f], 2) for f in GEO_FEAT},
                    # bat_speed descriptor = mean over the member shapes' centroids
                    "bat_speed": round(cent.loc[comp == a, DESCRIPTOR].mean(), 2)})
    lex = pd.DataFrame(lex).sort_values("archetype").reset_index(drop=True)
    names = lex["archetype_name"]
    if names.duplicated().any():
        raise ValueError(f"archetype names collide at K_ARCH={K_ARCH}: {names.tolist()} — "
                         "retune name thresholds or K_ARCH")

    out = cent.copy()
    out["archetype"] = remap[comp]
    out["arch_confidence"] = resp.max(axis=1).round(3)
    out = out.merge(lex[["archetype", "archetype_name"]], on="archetype", how="left")
    return out, lex


def write_catalog(shapes, lex):
    L = []
    w = L.append
    w("# Swing-shape archetype lexicon (Layer 1)\n")
    w(f"- {len(shapes):,} unit-clusters across {shapes.groupby(['batter_id','batter_stand']).ngroups} "
      f"(batter, stand) units, mapped onto **{K_ARCH} league archetypes** (pull frame).")
    w(f"- Naming is algorithmic from the archetype centroid: vertical "
      f"(Flat < {VAA_FLAT}° VAA <= Level < {VAA_STEEP}° <= Uppercut) x direction "
      f"(Oppo < {HAA_OPPO}° HAA_pull < Center < {HAA_PULL}° <= Pull).")
    w(f"- Assignment confidence (max archetype responsibility) — median "
      f"{shapes.arch_confidence.median():.2f}, share > 0.8: {(shapes.arch_confidence > 0.8).mean():.0%}\n")

    w("## The lexicon (raw-unit centroids)")
    w(lex[["archetype", "archetype_name", "n_shapes", "tilt", "len", "bat_speed", "vaa", "haa_pull"]]
      .to_markdown(index=False) + "\n")

    w("## Exemplars per archetype (largest shapes)")
    for _, r in lex.iterrows():
        ex = (shapes[shapes.archetype == r.archetype]
              .sort_values("n", ascending=False).head(6)["label"].tolist())
        w(f"- **{r.archetype_name}** ({r.n_shapes} shapes): " + ", ".join(ex))
    w("")

    (DATA / "archetype_lexicon.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    cent = load_centroids()
    print(f"Loaded {len(cent):,} unit-cluster centroids (pull frame)")
    shapes, lex = fit_archetype(cent)

    keep = (["batter_id", "batter_stand", "cluster", "label", "archetype", "archetype_name",
             "arch_confidence", "n", "weight"] + FEAT)
    shapes[keep].to_parquet(DATA / "shape_archetypes.parquet", index=False)
    lex.to_parquet(DATA / "archetype_lexicon.parquet", index=False)
    write_catalog(shapes, lex)
    print(f"\nWrote shape_archetypes / archetype_lexicon (.parquet) + archetype_lexicon.md")


if __name__ == "__main__":
    main()
