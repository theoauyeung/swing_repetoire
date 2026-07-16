"""Context-responsiveness (adjustability) — Facet-2 core metric.

How much does a hitter's SHAPE CHOICE depend on the PRE-SWING context? High = they reshape the
swing to the pitch/situation; low = same swing regardless. This is the "adjustability" the paper
tests, and it is DISTINCT from repertoire width (Repertoire+): a hitter can carry 5 far-apart shapes
but deploy them at random (wide, not adjustable), or 2 shapes switched sharply by count (narrow,
adjustable). Repertoire+ can't tell those apart; this metric is what does.

Per (batter, stand) unit with >= MIN_SWINGS competitive swings and k >= 2 shapes, over 2024-25:
  shape S      = the per-swing cluster id (from cluster_assignments)
  context axes = count_state (ahead/even/behind/2K), pitch_group (FB/breaking/offspeed), loc_zone
                 (3x3 inside/mid/outside x low/mid/high), all PRE-swing/exogenous.

Two independent estimators (they should agree):
  1. Normalized mutual information (uncertainty coefficient) U = I(C;S) / H(S), the fraction of a
     hitter's shape-choice uncertainty explained by context. Dividing by H(S) nets out repertoire
     entropy so "responsive" != "has many even shapes". Per-axis + overall (joint context).
     Bias-corrected with a within-unit permutation null (shuffle context, subtract the null mean),
     because raw MI is upward-biased with sparse cells.
  2. Classifier skill: out-of-fold log-loss improvement of a multinomial logit predicting S from
     context over the usage-prior baseline. Handles continuous location + interactions; the prior
     baseline auto-controls for entropy and sample size. Reported as a cross-check.

Headline `responsiveness` = null-adjusted overall U. Per-axis columns let you separate *volitional*
adjustment (count-responsiveness — a flatter 2-strike swing is a choice) from geometry that is
partly *forced* (location-responsiveness — the pitch location mechanically moves the intercept, and
horz_attack_angle is the most pitch-reactive feature). GUARDRAIL: this measures shape<->context
DEPENDENCE, agnostic to intent; whether it is *good* is decided by the adjustment-payoff regression
(does responsiveness predict higher xRV after controlling for hitter quality + repertoire).

Input:  data/cluster_assignments.parquet, data/swings_model.parquet
Output: data/context_response.parquet, data/context_response_catalog.md

Run:  python src/context_response.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KEY = ["batter_id", "batter_stand"]
SEASONS = [2024, 2025]
MIN_SWINGS = 300      # per (batter, stand) unit, for a stable MI/skill estimate
NULL_PERM = 200       # permutation-null resamples for MI bias correction
SEED = 7

PITCH_GROUP = {"FF": "FB", "SI": "FB", "FC": "FB",
               "SL": "breaking", "CU": "breaking", "KC": "breaking", "ST": "breaking",
               "SV": "breaking", "CS": "breaking", "SC": "breaking", "KN": "breaking",
               "CH": "offspeed", "FS": "offspeed", "FO": "offspeed"}


def add_context(df):
    """Attach the pre-swing context axes (all exogenous; no post-contact info)."""
    out = df.copy()
    out["pitch_group"] = out["pitch_type"].map(PITCH_GROUP).fillna("other")
    s, b = out["strikes"].to_numpy(), out["balls"].to_numpy()
    out["count_state"] = np.select([s == 2, b > s, b < s], ["2K", "ahead", "behind"], default="even")
    pull = np.where(out["batter_stand"].to_numpy() == "L", 1.0, -1.0)
    out["plate_x_pull"] = out["plate_x"] * pull                      # + = pull-side / inside, both hands
    out["plate_z_norm"] = (out["plate_z"] - out["sz_bot"]) / (out["sz_top"] - out["sz_bot"])
    loc_h = np.select([out["plate_x_pull"] > 0.3, out["plate_x_pull"] < -0.3], ["in", "out"], default="mid")
    loc_v = np.select([out["plate_z_norm"] > 0.66, out["plate_z_norm"] < 0.33], ["hi", "lo"], default="mid")
    out["loc_zone"] = pd.Series(loc_h, index=out.index) + "_" + pd.Series(loc_v, index=out.index)
    out["same_hand"] = (out["batter_stand"] == out["pitcher_throws"]).astype(int)
    return out


def mi_bits(s, c, ns, nc):
    """Mutual information I(S;C) in bits from integer-coded label arrays."""
    joint = np.bincount(s * nc + c, minlength=ns * nc).reshape(ns, nc).astype(float)
    n = joint.sum()
    pij = joint / n
    pi, pj = pij.sum(1), pij.sum(0)
    ii, jj = np.where(pij > 0)
    return float(np.sum(pij[ii, jj] * np.log2(pij[ii, jj] / (pi[ii] * pj[jj]))))


def entropy_bits(s, ns):
    p = np.bincount(s, minlength=ns) / len(s)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def axis_responsiveness(s, c, Hs, rng):
    """Normalized MI U=I/H(S), plus a permutation-null-adjusted MI and its z. c = integer context."""
    ns, nc = s.max() + 1, c.max() + 1
    I = mi_bits(s, c, ns, nc)
    null = np.array([mi_bits(s, rng.permutation(c), ns, nc) for _ in range(NULL_PERM)])
    mu, sd = null.mean(), null.std()
    U = I / Hs if Hs > 0 else 0.0
    U_adj = max(I - mu, 0.0) / Hs if Hs > 0 else 0.0        # null-corrected fraction of H(S) explained
    z = (I - mu) / sd if sd > 0 else 0.0
    return U, U_adj, z


def classifier_skill(g):
    """OOF log-loss improvement of a multinomial logit (context -> shape) over the usage prior.
    >0 = context predicts shape beyond the hitter's overall usage mix. NaN if a shape is too rare
    to stratify."""
    feat = ["plate_x_pull", "plate_z_norm", "balls", "strikes", "same_hand", "pitch_group"]
    g = g.dropna(subset=["plate_x_pull", "plate_z_norm"])   # missing strike-zone / location rows
    y = pd.factorize(g["cluster"], sort=True)[0]            # contiguous 0..k-1
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() < 5:
        return np.nan
    X = pd.get_dummies(g[feat], columns=["pitch_group"]).to_numpy(float)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    ll_model = log_loss(y, proba, labels=range(len(counts)))
    prior = np.tile(counts / counts.sum(), (len(y), 1))
    ll_prior = log_loss(y, prior, labels=range(len(counts)))
    return float(ll_prior - ll_model)                       # nats; >0 = context helps


def main():
    ca = pd.read_parquet("data/cluster_assignments.parquet", columns=["play_id"] + KEY + ["cluster"])
    ctx = pd.read_parquet(DATA / "swings_model.parquet",
                          columns=["play_id", "game_year", "batter_full_name", "pitcher_throws",
                                   "balls", "strikes", "plate_x", "plate_z", "sz_top", "sz_bot",
                                   "pitch_type", "batter_stand", "batter_id"])
    df = add_context(ctx[ctx["game_year"].isin(SEASONS)].merge(ca, on=["play_id"] + KEY, how="inner"))
    rng = np.random.default_rng(SEED)

    rows = []
    for (bid, stand), g in df.groupby(KEY, sort=False):
        if len(g) < MIN_SWINGS or g["cluster"].nunique() < 2:
            continue
        s = pd.factorize(g["cluster"], sort=True)[0]
        Hs = entropy_bits(s, s.max() + 1)
        axes = {ax: axis_responsiveness(s, pd.factorize(g[col])[0], Hs, rng)
                for ax, col in [("count", "count_state"), ("pitch", "pitch_group"), ("loc", "loc_zone")]}
        joint = pd.factorize(g["count_state"] + "|" + g["pitch_group"] + "|" + g["loc_zone"])[0]
        U_all, U_all_adj, z_all = axis_responsiveness(s, joint, Hs, rng)
        rows.append({
            "batter_id": bid, "batter_stand": stand, "label": g["batter_full_name"].iloc[0],
            "n_swings": len(g), "k": int(g["cluster"].nunique()), "shape_entropy": round(Hs, 3),
            "responsiveness": round(U_all_adj, 4), "resp_z": round(z_all, 2),
            "resp_raw_U": round(U_all, 4), "resp_clf": round(classifier_skill(g), 4),
            "resp_count": round(axes["count"][1], 4), "resp_pitch": round(axes["pitch"][1], 4),
            "resp_loc": round(axes["loc"][1], 4)})

    out = pd.DataFrame(rows)
    out["resp_pctile"] = (out["responsiveness"].rank(pct=True) * 100).round(1)
    out = out.sort_values("responsiveness", ascending=False).reset_index(drop=True)
    out.to_parquet(DATA / "context_response.parquet", index=False)
    write_catalog(out)
    print(f"Wrote context_response.parquet ({len(out)} units, >= {MIN_SWINGS} swings & k>=2, {SEASONS})")


def write_catalog(a):
    L = ["# Context-responsiveness (adjustability) catalog\n",
         "Per (batter, stand) unit: how much shape choice depends on pre-swing context — the paper's "
         "adjustability metric, **distinct from Repertoire+ width**. Headline `responsiveness` = "
         "null-adjusted normalized mutual information `U = (I(C;S) - null) / H(S)` over the joint "
         f"context (count x pitch-group x location), 2024-25, units with >= {MIN_SWINGS} swings & "
         "k>=2 shapes.\n",
         "**Guardrail:** this measures shape<->context *dependence*, not intent. A high value can be "
         "volitional adjustment OR the pitch mechanically forcing a different intercept geometry — "
         "the adjustment-payoff regression (does it raise xRV, net of quality + repertoire) is what "
         "adjudicates. Read `resp_count` (cleanest volitional signal) vs `resp_loc` (most contaminated "
         "by forced geometry / pitch-reactive horz_attack_angle) to gauge which.\n",
         f"- Units scored: **{len(a)}**  ·  mean responsiveness {a.responsiveness.mean():.3f}, "
         f"median {a.responsiveness.median():.3f}, max {a.responsiveness.max():.3f}",
         "- Columns: `responsiveness` (headline, null-adj U), `resp_z` (SDs above the shuffle null), "
         "`resp_raw_U` (uncorrected), `resp_clf` (classifier-skill cross-check, nats), "
         "`resp_count`/`resp_pitch`/`resp_loc` (per-axis null-adj U).\n"]
    show = ["label", "batter_stand", "n_swings", "k", "responsiveness", "resp_z", "resp_clf",
            "resp_count", "resp_pitch", "resp_loc"]
    L.append("## Most context-responsive (most adjustable)")
    L.append(a.head(20)[show].to_markdown(index=False) + "\n")
    L.append("## Least context-responsive (same swing regardless)")
    L.append(a.tail(20)[show].sort_values("responsiveness")[show].to_markdown(index=False) + "\n")
    L.append("## Most COUNT-responsive (cleanest volitional-adjustment signal)")
    L.append(a.sort_values("resp_count", ascending=False).head(15)[show].to_markdown(index=False) + "\n")
    (DATA / "context_response_catalog.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
