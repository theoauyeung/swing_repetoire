""" per-batter swing-shape clustering via Gaussian Mixture Models.

For each qualifying hitter (>= MIN_SWINGS competitive tracked swings, 2024-26 pooled):
  - z-score the 5 shape features within that hitter,
  - fit full-covariance GMMs for increasing k and select k by minimum BIC (early-stopping
    once BIC stops improving). 
  - assign every swing to its most-likely component (with a responsibility/confidence),
  - cluster 0 = the hitter's primary swing,
  - score how distinct the shapes are via shape_dispersion (usage-weighted mean pairwise
    Mahalanobis distance between cluster centroids, each pair measured against its own
    within-cluster covariance — separation in units of the batter's own swing scatter;
    within-batter, not yet cross-batter comparable).

Input:  data/swings_model.parquet
Outputs:
  data/cluster_assignments.parquet  one row per swing: play_id, batter_id, cluster, resp_max
  data/cluster_summary.parquet      one row per (batter, cluster): weight, n, raw centroid
  data/batter_repertoire.parquet    one row per batter: k, bic, usage entropy, effective shapes,
                                    shape_dispersion (within-batter Mahalanobis separation of shapes)
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
                     "vert_attack_angle", "horz_attack_angle"]  # report raw (unmirrored) centroids
MIN_SWINGS = 300      # cohort threshold (pooled competitive swings)
D = len(FEATURES)
PARAMS_PER_COMP = D + D * (D + 1) // 2   # free params of a full-cov Gaussian in D dims (=20 for D=5)
PATIENCE = 3          # stop searching k once BIC fails to improve this many times in a row
N_INIT = 5            # EM restarts per k (stabilizes the BIC estimate we now rely on)
SEED = 7


def fit_batter(X_raw):
    """Fit per-batter GMM, selecting k by minimum BIC.
    Returns labels, resp_max, mu, sd, k, bic, shape_dispersion."""
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

    gm, k = best_gm, best_k
    resp = gm.predict_proba(X)
    labels = resp.argmax(axis=1)
    resp_max = resp.max(axis=1)
    disp = shape_dispersion(gm.means_, gm.covariances_, gm.weights_)

    # relabel by descending usage weight -> cluster 0 = primary swing
    order = np.argsort(-gm.weights_)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    labels = remap[labels]
    return labels, resp_max, mu, sd, k, best_bic, disp


def shape_dispersion(means, covs, weights):
    """How distinct a hitter's own shapes are: usage-weighted mean pairwise Mahalanobis
    distance between cluster centroids, each pair measured against its own within-cluster
    covariance """
    k = len(means)
    if k < 2:
        return 0.0
    num = den = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            d = means[i] - means[j]
            S = 0.5 * (covs[i] + covs[j])
            dist = float(np.sqrt(d @ np.linalg.solve(S, d)))
            wpair = weights[i] * weights[j]
            num += wpair * dist
            den += wpair
    return num / den if den > 0 else 0.0


def main():
    df = pd.read_parquet(DATA / "swings_model.parquet",
                         columns=["play_id", "batter_id", "batter_full_name"] + FEATURES + ["horz_attack_angle"])
    per = df.groupby("batter_id").size()
    cohort = per[per >= MIN_SWINGS].index
    df = df[df.batter_id.isin(cohort)].copy()
    print(f"Cohort: {len(cohort)} batters, {len(df):,} swings (>= {MIN_SWINGS} each)")

    assign_rows, summary_rows, batter_rows = [], [], []
    for i, (bid, g) in enumerate(df.groupby("batter_id", sort=False)):
        X = g[FEATURES].to_numpy(float)
        labels, resp_max, mu, sd, k, bic, disp = fit_batter(X)
        name = g["batter_full_name"].iloc[0]

        gg = g.assign(cluster=labels)
        assign_rows.append(pd.DataFrame({
            "play_id": g["play_id"].values, "batter_id": bid,
            "cluster": labels, "resp_max": resp_max.round(3),
        }))

        weights = np.bincount(labels, minlength=k) / len(labels)
        for c in range(k):
            sub = gg[gg.cluster == c]
            row = {"batter_id": bid, "batter_full_name": name, "cluster": c,
                   "n": len(sub), "weight": round(weights[c], 4)}
            row.update({f"{col}_mean": round(sub[col].mean(), 3) for col in RAW_CENTROID_COLS})
            summary_rows.append(row)

        w = weights[weights > 0]
        entropy = float(-(w * np.log(w)).sum())
        batter_rows.append({"batter_id": bid, "batter_full_name": name, "n_swings": len(labels),
                            "k": k, "bic": round(bic, 1), "min_weight": round(weights.min(), 4),
                            "min_comp_n": int(np.bincount(labels).min()),
                            "usage_entropy": round(entropy, 3),
                            "effective_shapes": round(float(np.exp(entropy)), 2),
                            "shape_dispersion": round(disp, 3)})
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(cohort)} batters")

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
    w("# Swing-shape cluster catalog (per-batter GMM)\n")
    w(f"- Cohort: **{len(rep)} batters**, **{int(rep.n_swings.sum()):,} swings**")
    w(f"- Repertoire size (k) — mean {rep.k.mean():.2f}, median {int(rep.k.median())}")
    w(f"- Effective shapes (exp usage-entropy) — mean {rep.effective_shapes.mean():.2f}")
    w(f"- Shape dispersion (usage-wtd mean pairwise within-cluster Mahalanobis) — "
      f"mean {rep.shape_dispersion.mean():.2f}, median {rep.shape_dispersion.median():.2f} "
      f"(k=1 hitters contribute 0; within-batter scale, not yet cross-batter comparable)\n")

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
    w(top[["batter_full_name", "n_swings", "k", "effective_shapes", "shape_dispersion"]]
      .to_markdown(index=False) + "\n")

    w("## Most distinct repertoires (shapes most separated from each other, >=800 swings)")
    disp = rep[rep.n_swings >= 800].sort_values("shape_dispersion", ascending=False).head(10)
    w(disp[["batter_full_name", "n_swings", "k", "effective_shapes", "shape_dispersion"]]
      .to_markdown(index=False) + "\n")

    w("## Most one-note hitters (fewest effective shapes, >=800 swings)")
    mono = rep[rep.n_swings >= 800].sort_values("effective_shapes").head(10)
    w(mono[["batter_full_name", "n_swings", "k", "effective_shapes"]].to_markdown(index=False) + "\n")

    # one worked example: highest-k hitter's cluster centroids in real units
    ex_id = rep.sort_values(["k", "n_swings"], ascending=False).iloc[0].batter_id
    ex = summary[summary.batter_id == ex_id].sort_values("cluster")
    ex_name = ex.batter_full_name.iloc[0]
    w(f"## Example repertoire — {ex_name} (raw-unit cluster centroids)")
    cols = ["cluster", "n", "weight"] + [f"{c}_mean" for c in RAW_CENTROID_COLS]
    w(ex[cols].to_markdown(index=False))

    (DATA / "cluster_catalog.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
