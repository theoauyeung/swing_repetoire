"""Finds each hitter's distinct swings. Nobody has one swing — sorting a season of tracked swings
by their measured shape shows most hitters have two or three they actually repeat, and this is
what separates them.

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
    Returns labels, resp_max, feature_means, feature_sds, n_clusters, bic_score."""
    feature_means = X_raw.mean(axis=0)
    feature_sds   = X_raw.std(axis=0)
    # Guard against zero-variance features so standardization doesn't divide by zero
    feature_sds[feature_sds == 0] = 1.0
    X_scaled = (X_raw - feature_means) / feature_sds
    n_swings = len(X_scaled)
    # Identifiability bound: need at least 1 swing per free parameter per component
    max_n_clusters = max(1, n_swings // PARAMS_PER_COMP)

    best_model      = None
    best_bic_score  = np.inf
    stagnant_rounds = 0

    for n_clusters in range(1, max_n_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                              n_init=N_INIT, reg_covar=1e-5, random_state=SEED).fit(X_scaled)
        bic_score = gmm.bic(X_scaled)
        if bic_score < best_bic_score - 1e-6:
            best_model      = gmm
            best_bic_score  = bic_score
            stagnant_rounds = 0
        else:
            stagnant_rounds += 1
            if stagnant_rounds >= PATIENCE:
                break

    # BIC over-splits at large n, so merge near-duplicate components before finalizing.
    # n_clusters is the POST-merge shape count (the reported repertoire size).
    responsibilities = best_model.predict_proba(X_scaled)
    labels, resp_max, usage_weights = merge_components(
        X_scaled, responsibilities, best_model.means_, best_model.covariances_, MERGE_SEP)
    n_clusters = len(usage_weights)

    # relabel by descending usage weight -> cluster 0 = primary swing
    usage_order = np.argsort(-usage_weights)
    label_remap = np.empty(n_clusters, dtype=int)
    label_remap[usage_order] = np.arange(n_clusters)
    labels = label_remap[labels]
    return labels, resp_max, feature_means, feature_sds, n_clusters, best_bic_score


def _pair_maha(mean_i, cov_i, mean_j, cov_j):
    """Bhattacharyya-style Mahalanobis distance between two components, measured against their
    pooled within-cluster covariance (separation in units of the components' own scatter)."""
    mean_diff   = mean_i - mean_j
    pooled_cov  = 0.5 * (cov_i + cov_j)
    return float(np.sqrt(mean_diff @ np.linalg.solve(pooled_cov, mean_diff)))


def _group_stats(group_id, surviving_groups, X_scaled, hard_assignments, means0, covs0):
    """Return (mean, cov) for a surviving merge group.

    Singletons reuse the GMM's own fitted (regularized) params — robust for small components.
    Merged groups recompute empirically from all pooled swings in their member components.
    """
    member_components = surviving_groups[group_id]
    if len(member_components) == 1:
        component_index = member_components[0]
        return means0[component_index], covs0[component_index]
    group_points = X_scaled[np.isin(hard_assignments, member_components)]
    reg = 1e-5 * np.eye(X_scaled.shape[1])
    return group_points.mean(0), np.cov(group_points.T) + reg


def merge_components(X_scaled, resp, means0, covs0, thresh):
    """Collapse component pairs closer than `thresh` into one shape, closest pair first, until all
    surviving pairs clear the bar (or one remains). Un-merged singletons keep the GMM's fitted
    (regularized) params; a merged group's params are recomputed empirically from its pooled swings.
    Merged responsibility = sum of member components' responsibilities. Returns
    labels, resp_max, weights in X_scaled's (z-scored) frame."""
    hard_assignments = resp.argmax(axis=1)  # original hard component per swing
    surviving_groups = {group: [group] for group in range(len(means0))}  # group_id -> original component ids

    while len(surviving_groups) > 1:
        group_ids = list(surviving_groups)

        # Find the closest pair of surviving groups
        closest_dist = np.inf
        group_a = group_b = None
        for i, first_group in enumerate(group_ids):
            for second_group in group_ids[i + 1:]:
                stats_first  = _group_stats(first_group,  surviving_groups, X_scaled, hard_assignments, means0, covs0)
                stats_second = _group_stats(second_group, surviving_groups, X_scaled, hard_assignments, means0, covs0)
                distance = _pair_maha(*stats_first, *stats_second)
                if distance < closest_dist:
                    closest_dist = distance
                    group_a = first_group
                    group_b = second_group

        if closest_dist >= thresh:
            break
        surviving_groups[group_a] += surviving_groups.pop(group_b)  # absorb group_b into group_a

    # Build a mapping from original component index to final cluster label
    component_to_final_label = {}
    for final_label, (_, member_components) in enumerate(surviving_groups.items()):
        for component_index in member_components:
            component_to_final_label[component_index] = final_label

    labels = np.array([component_to_final_label[comp] for comp in hard_assignments])
    n_final_clusters = len(surviving_groups)

    # Sum member component responsibilities into the final merged cluster
    merged_responsibilities = np.zeros((len(X_scaled), n_final_clusters))
    for component_index in range(len(means0)):
        final_label = component_to_final_label[component_index]
        merged_responsibilities[:, final_label] += resp[:, component_index]

    resp_max = merged_responsibilities[np.arange(len(X_scaled)), labels].round(3)
    usage_weights = np.array([(labels == label).mean() for label in range(n_final_clusters)])
    return labels, resp_max, usage_weights


def main():
    swings = pd.read_parquet(DATA / "swings_model.parquet",
                             columns=["play_id", "batter_id", "batter_full_name", "batter_stand"]
                                     + FEATURES + ["horz_attack_angle"])
    KEY = ["batter_id", "batter_stand"]
    swings_per_unit  = swings.groupby(KEY).size()
    qualifying_units = swings_per_unit[swings_per_unit >= MIN_SWINGS].index
    swings = swings[pd.MultiIndex.from_frame(swings[KEY]).isin(qualifying_units)].copy()

    # Switch hitters qualify with both stances; only they get the (L)/(R) suffix on their display
    # label — one-way hitters keep their bare name.
    stands_per_batter = swings.groupby("batter_id")["batter_stand"].nunique()
    switch_hitter_ids = set(stands_per_batter[stands_per_batter == 2].index)
    print(f"Cohort: {len(qualifying_units)} (batter, stand) units, {len(swings):,} swings (>= {MIN_SWINGS} each); "
          f"{len(switch_hitter_ids)} switch hitters clustered as two units each")

    assign_rows, summary_rows, batter_rows = [], [], []
    for unit_index, ((batter_id, stand), unit_swings) in enumerate(swings.groupby(KEY, sort=False)):
        X_scaled = unit_swings[FEATURES].to_numpy(float)
        labels, resp_max, _, _, n_clusters, bic_score = fit_batter(X_scaled)
        batter_name   = unit_swings["batter_full_name"].iloc[0]
        # Switch hitters get a stance suffix so their two rows are distinguishable in leaderboards
        display_label = f"{batter_name} ({stand})" if batter_id in switch_hitter_ids else batter_name

        unit_swings_with_cluster = unit_swings.assign(cluster=labels)
        assign_rows.append(pd.DataFrame({
            "play_id":      unit_swings["play_id"].values,
            "batter_id":    batter_id,
            "batter_stand": stand,
            "cluster":      labels,
            "resp_max":     resp_max.round(3),
        }))

        cluster_counts = np.bincount(labels, minlength=n_clusters)
        usage_weights  = cluster_counts / len(labels)
        for cluster_id in range(n_clusters):
            cluster_swings = unit_swings_with_cluster[unit_swings_with_cluster.cluster == cluster_id]
            row = {
                "batter_id": batter_id, "batter_stand": stand, "batter_full_name": batter_name,
                "label": display_label, "cluster": cluster_id,
                "n": len(cluster_swings), "weight": round(usage_weights[cluster_id], 4),
            }
            for col in RAW_CENTROID_COLS:
                row[f"{col}_mean"] = round(cluster_swings[col].mean(), 3)
            summary_rows.append(row)

        # Usage entropy: higher = more evenly spread across shapes; exp(entropy) = effective shapes
        nonzero_weights = usage_weights[usage_weights > 0]
        usage_entropy = float(-(nonzero_weights * np.log(nonzero_weights)).sum())
        batter_rows.append({
            "batter_id": batter_id, "batter_stand": stand, "batter_full_name": batter_name,
            "label": display_label, "n_swings": len(labels),
            "k": n_clusters, "bic": round(bic_score, 1),
            "min_weight": round(usage_weights.min(), 4),
            "min_comp_n": int(cluster_counts.min()),
            "usage_entropy":   round(usage_entropy, 3),
            "effective_shapes": round(float(np.exp(usage_entropy)), 2),
        })
        if (unit_index + 1) % 100 == 0:
            print(f"  ...{unit_index + 1}/{len(qualifying_units)} units")

    assignments = pd.concat(assign_rows, ignore_index=True)
    summary      = pd.DataFrame(summary_rows)
    repertoire   = pd.DataFrame(batter_rows)

    assignments.to_parquet(DATA / "cluster_assignments.parquet", index=False)
    summary.to_parquet(DATA / "cluster_summary.parquet", index=False)
    repertoire.to_parquet(DATA / "batter_repertoire.parquet", index=False)
    write_catalog(repertoire, summary)
    print(f"\nWrote cluster_assignments / cluster_summary / batter_repertoire / cluster_catalog.md")


def write_catalog(repertoire, summary):
    lines = []
    lines.append("# Swing-shape cluster catalog (per-(batter, stand) GMM)\n")
    lines.append(f"- Cohort: **{len(repertoire)} (batter, stand) units**, **{int(repertoire.n_swings.sum()):,} swings** "
                 f"(switch hitters contribute one unit per stance)")
    lines.append(f"- Repertoire size (k) — mean {repertoire.k.mean():.2f}, median {int(repertoire.k.median())}")
    lines.append(f"- Effective shapes (exp usage-entropy) — mean {repertoire.effective_shapes.mean():.2f}\n")

    lines.append("## BIC-selection sanity: are any components degenerate? (no occupancy floor is imposed)")
    lines.append(f"- smallest mixture weight across hitters — min {repertoire.min_weight.min():.3f}, "
                 f"median {repertoire.min_weight.median():.3f}")
    lines.append(f"- smallest component size across hitters — min {int(repertoire.min_comp_n.min())}, "
                 f"median {int(repertoire.min_comp_n.median())}")
    lines.append(f"- hitters with a component < 15 swings: {int((repertoire.min_comp_n < 15).sum())} "
                 f"| < 3% usage: {int((repertoire.min_weight < 0.03).sum())}\n")

    lines.append("## Distribution of repertoire size (k) across hitters")
    k_distribution = repertoire.k.value_counts().sort_index()
    lines.append(k_distribution.to_frame("batters").to_markdown() + "\n")

    lines.append("## Widest repertoires (most effective shapes)")
    top_wide = repertoire.sort_values("effective_shapes", ascending=False).head(10)
    lines.append(top_wide[["label", "n_swings", "k", "effective_shapes"]]
                 .to_markdown(index=False) + "\n")

    lines.append("## Most one-note hitters (fewest effective shapes, >=800 swings)")
    most_monotone = repertoire[repertoire.n_swings >= 800].sort_values("effective_shapes").head(10)
    lines.append(most_monotone[["label", "n_swings", "k", "effective_shapes"]].to_markdown(index=False) + "\n")

    # one worked example: highest-k unit's cluster centroids in real units
    example_unit = repertoire.sort_values(["k", "n_swings"], ascending=False).iloc[0]
    example_summary_rows = summary[
        (summary.batter_id == example_unit.batter_id) &
        (summary.batter_stand == example_unit.batter_stand)
    ].sort_values("cluster")
    lines.append(f"## Example repertoire — {example_unit.label} (raw-unit cluster centroids)")
    centroid_cols = ["cluster", "n", "weight"] + [f"{col}_mean" for col in RAW_CENTROID_COLS]
    lines.append(example_summary_rows[centroid_cols].to_markdown(index=False))

    catalog_text = "\n".join(lines)
    (DATA / "cluster_catalog.md").write_text(catalog_text, encoding="utf-8")
    print(catalog_text)


if __name__ == "__main__":
    main()
