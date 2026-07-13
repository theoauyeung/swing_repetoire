""" per-(batter, stand) swing-shape clustering via Gaussian Mixture Models.

Clustering unit = (batter_id, batter_stand). A switch hitter's left- and right-handed swings
are different movements; pooling them would let stance dominate the GMM (cluster 0 = "bats L",
cluster 1 = "bats R" instead of real swing shapes). So each stance is clustered as its own
"player" — Cal Raleigh L vs Cal Raleigh R. One-way hitters have a single stance and are
unaffected. Only horz_attack_angle is handedness-mirrored; the other four features are not,
which is exactly why an unsplit switch hitter separates on stance.

For each qualifying unit (>= MIN_SWINGS competitive tracked swings, 2024-26 pooled):
  - z-score the 5 shape features within that hitter,
  - fit full-covariance GMMs for increasing k and select k by minimum BIC (early-stopping
    once BIC stops improving),
  - merge component pairs closer than MERGE_SEP into one shape (BIC over-segments at large n
    into large-but-near-duplicate components; the merge collapses those back so each cluster is
    a genuinely distinct shape). Reported k is the post-merge count.
  - assign every swing to its most-likely (merged) component with a responsibility/confidence,
  - cluster 0 = the hitter's primary swing.

Input:  data/swings_model.parquet
Outputs:
  data/cluster_assignments.parquet  one row per swing: play_id, batter_id, batter_stand, cluster, resp_max
  data/cluster_summary.parquet      one row per (batter, stand, cluster): weight, n, raw centroid
  data/batter_repertoire.parquet    one row per (batter, stand): k, bic, usage entropy, effective shapes
  data/cluster_catalog.md           human-readable summary

Run:  python src/cluster.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FEATURES = ["swing_path_tilt", "swing_length", "bat_speed",
            "vert_attack_angle", "horz_attack_angle_pull"]
RAW_CENTROID_COLS = ["swing_path_tilt", "swing_length", "bat_speed",
                     "vert_attack_angle", "horz_attack_angle"] 
MIN_SWINGS = 150      
D = len(FEATURES)
PARAMS_PER_COMP = D + D * (D + 1) // 2   # free params of a full-cov Gaussian in D dims (=20 for D=5)
PATIENCE = 3          # stop searching k once BIC fails to improve this many times in a row
N_INIT = 5            # EM restarts per k (stabilizes the BIC estimate we now rely on)
SEED = 7
MERGE_SEP = 1.75      # post-BIC merge: collapse component pairs closer than this (within-cluster-SD
                      # Mahalanobis) into one shape. Chosen via a threshold sweep (worklog 2026-07-13):
                      # the BIC-component separation distribution is a gapless continuum peaking at
                      # ~2.0, so this is a judgment dial, not a data-pinned value. 1.75 (~19% overlap)
                      # keeps mean k at 2.26 (vs 1.94 at 2.0) and cuts single-shape units 24%->13% for
                      # a richer Facet-2 signal, while still merging the true near-duplicate mass (<1.5).


def fit_batter(X_raw):
    """Fit per-batter GMM, selecting k by minimum BIC.
    Returns labels, resp_max, mu, sd, k, bic."""
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X_raw - mu) / sd
    n = len(X)
    k_max = max(1, n // PARAMS_PER_COMP)  # identifiability bound: >= 1 point per free param per component

    best_k, best_gm, best_bic, since = 1, None, np.inf, 0
    for k in range(1, k_max + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=N_INIT, reg_covar=1e-5, random_state=SEED).fit(X)
        bic = gm.bic(X)
        if bic < best_bic - 1e-6:
            best_k, best_gm, best_bic, since = k, gm, bic, 0
        else:
            since += 1
            if since >= PATIENCE:
                break

    # BIC over-splits at large n, so merge near-duplicate components before finalizing. k is the
    # POST-merge shape count (the reported repertoire size); best_bic is the selected model's BIC.
    resp = best_gm.predict_proba(X)
    labels, resp_max, weights = merge_components(
        X, resp, best_gm.means_, best_gm.covariances_, MERGE_SEP)
    k = len(weights)

    # relabel by descending usage weight -> cluster 0 = primary swing
    order = np.argsort(-weights)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    labels = remap[labels]
    return labels, resp_max, mu, sd, k, best_bic


def _pair_maha(mi, Si, mj, Sj):
    """Bhattacharyya-style Mahalanobis distance between two components, measured against their
    pooled within-cluster covariance (separation in units of the components' own scatter)."""
    d = mi - mj
    return float(np.sqrt(d @ np.linalg.solve(0.5 * (Si + Sj), d)))


def merge_components(X, resp, means0, covs0, thresh):
    """Collapse component pairs closer than `thresh` into one shape, closest pair first, until all
    surviving pairs clear the bar (or one remains). Un-merged singletons keep the GMM's fitted
    (regularized) params; a merged group's params are recomputed empirically from its pooled swings.
    Merged responsibility = sum of member components' responsibilities. Returns
    labels, resp_max, weights in X's (z-scored) frame."""
    comp = resp.argmax(axis=1)                         # original hard component per swing
    member = {g: [g] for g in range(len(means0))}      # surviving group -> original component ids
    reg = 1e-5 * np.eye(X.shape[1])

    def gstats(g):
        ms = member[g]
        if len(ms) == 1:
            return means0[ms[0]], covs0[ms[0]]         # GMM's own PD params (robust for small comps)
        pts = X[np.isin(comp, ms)]
        return pts.mean(0), np.cov(pts.T) + reg

    while len(member) > 1:
        stats = {g: gstats(g) for g in member}
        gs = list(member)
        dist, a, b = min(((_pair_maha(*stats[x], *stats[y]), x, y)
                          for i, x in enumerate(gs) for y in gs[i + 1:]), key=lambda t: t[0])
        if dist >= thresh:
            break
        member[a] += member.pop(b)                     # absorb b into a

    fg = {c: i for i, g in enumerate(member) for c in member[g]}   # original comp -> final label
    labels = np.array([fg[c] for c in comp])
    kf = len(member)
    merged_resp = np.zeros((len(X), kf))
    for c in range(len(means0)):
        merged_resp[:, fg[c]] += resp[:, c]
    resp_max = merged_resp[np.arange(len(X)), labels].round(3)
    weights = np.array([(labels == i).mean() for i in range(kf)])
    return labels, resp_max, weights


def main():
    df = pd.read_parquet(DATA / "swings_model.parquet",
                         columns=["play_id", "batter_id", "batter_full_name", "batter_stand"]
                                 + FEATURES + ["horz_attack_angle"])
    KEY = ["batter_id", "batter_stand"]
    per = df.groupby(KEY).size()
    cohort = per[per >= MIN_SWINGS].index
    df = df[pd.MultiIndex.from_frame(df[KEY]).isin(cohort)].copy()
    # switch hitters = qualifying batters with both stances in the cohort; only they get the
    # (L)/(R) suffix on their display label (one-way hitters keep their bare name).
    stands_per_batter = df.groupby("batter_id")["batter_stand"].nunique()
    switch_ids = set(stands_per_batter[stands_per_batter == 2].index)
    print(f"Cohort: {len(cohort)} (batter, stand) units, {len(df):,} swings (>= {MIN_SWINGS} each); "
          f"{len(switch_ids)} switch hitters clustered as two units each")

    assign_rows, summary_rows, batter_rows = [], [], []
    for i, ((bid, stand), g) in enumerate(df.groupby(KEY, sort=False)):
        X = g[FEATURES].to_numpy(float)
        labels, resp_max, mu, sd, k, bic = fit_batter(X)
        name = g["batter_full_name"].iloc[0]
        label = f"{name} ({stand})" if bid in switch_ids else name

        gg = g.assign(cluster=labels)
        assign_rows.append(pd.DataFrame({
            "play_id": g["play_id"].values, "batter_id": bid, "batter_stand": stand,
            "cluster": labels, "resp_max": resp_max.round(3),
        }))

        weights = np.bincount(labels, minlength=k) / len(labels)
        for c in range(k):
            sub = gg[gg.cluster == c]
            row = {"batter_id": bid, "batter_stand": stand, "batter_full_name": name,
                   "label": label, "cluster": c, "n": len(sub), "weight": round(weights[c], 4)}
            row.update({f"{col}_mean": round(sub[col].mean(), 3) for col in RAW_CENTROID_COLS})
            summary_rows.append(row)

        w = weights[weights > 0]
        entropy = float(-(w * np.log(w)).sum())
        batter_rows.append({"batter_id": bid, "batter_stand": stand, "batter_full_name": name,
                            "label": label, "n_swings": len(labels),
                            "k": k, "bic": round(bic, 1), "min_weight": round(weights.min(), 4),
                            "min_comp_n": int(np.bincount(labels).min()),
                            "usage_entropy": round(entropy, 3),
                            "effective_shapes": round(float(np.exp(entropy)), 2)})
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(cohort)} units")

    assignments = pd.concat(assign_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    repertoire = pd.DataFrame(batter_rows)

    assignments.to_parquet(DATA / "cluster_assignments.parquet", index=False)
    summary.to_parquet(DATA / "cluster_summary.parquet", index=False)
    repertoire.to_parquet(DATA / "batter_repertoire.parquet", index=False)
    write_catalog(repertoire, summary)
    print(f"\nWrote cluster_assignments / cluster_summary / batter_repertoire / cluster_catalog.md")


def write_catalog(rep, summary):
    L = []
    w = L.append
    w("# Swing-shape cluster catalog (per-(batter, stand) GMM)\n")
    w(f"- Cohort: **{len(rep)} (batter, stand) units**, **{int(rep.n_swings.sum()):,} swings** "
      f"(switch hitters contribute one unit per stance)")
    w(f"- Repertoire size (k) — mean {rep.k.mean():.2f}, median {int(rep.k.median())}")
    w(f"- Effective shapes (exp usage-entropy) — mean {rep.effective_shapes.mean():.2f}\n")

    w("## BIC-selection sanity: are any components degenerate? (no occupancy floor is imposed)")
    w(f"- smallest mixture weight across hitters — min {rep.min_weight.min():.3f}, "
      f"median {rep.min_weight.median():.3f}")
    w(f"- smallest component size across hitters — min {int(rep.min_comp_n.min())}, "
      f"median {int(rep.min_comp_n.median())}")
    w(f"- hitters with a component < 15 swings: {int((rep.min_comp_n < 15).sum())} "
      f"| < 3% usage: {int((rep.min_weight < 0.03).sum())}\n")

    w("## Distribution of repertoire size (k) across hitters")
    kd = rep.k.value_counts().sort_index()
    w(kd.to_frame("batters").to_markdown() + "\n")

    w("## Widest repertoires (most effective shapes)")
    top = rep.sort_values("effective_shapes", ascending=False).head(10)
    w(top[["label", "n_swings", "k", "effective_shapes"]]
      .to_markdown(index=False) + "\n")

    w("## Most one-note hitters (fewest effective shapes, >=800 swings)")
    mono = rep[rep.n_swings >= 800].sort_values("effective_shapes").head(10)
    w(mono[["label", "n_swings", "k", "effective_shapes"]].to_markdown(index=False) + "\n")

    # one worked example: highest-k unit's cluster centroids in real units
    ex = rep.sort_values(["k", "n_swings"], ascending=False).iloc[0]
    ex_rows = summary[(summary.batter_id == ex.batter_id) &
                      (summary.batter_stand == ex.batter_stand)].sort_values("cluster")
    w(f"## Example repertoire — {ex.label} (raw-unit cluster centroids)")
    cols = ["cluster", "n", "weight"] + [f"{c}_mean" for c in RAW_CENTROID_COLS]
    w(ex_rows[cols].to_markdown(index=False))

    (DATA / "cluster_catalog.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
