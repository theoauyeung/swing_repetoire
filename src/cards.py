""" Layer 2 — per-hitter swing ID cards.

Turns each per-(batter, stand) cluster from a bare index into a self-describing shape: an archetype
tag (Layer 1), a name relative to the hitter's own primary swing, the situations it's over-used in,
and its run value in-context + versus the primary in matched situations. One row per (unit, cluster).

Card fields per shape:
  - archetype_name  (Layer 1 tag)
  - role            'primary' (cluster 0, the highest-usage swing) or 'secondary'
  - name_delta      readable geometry delta. Primary: vs the league (cohort) average. Secondary:
                    vs THIS hitter's primary. Top features by standardized magnitude, e.g.
                    "+5deg uppercut, -3mph slower, +6deg pull". This is where the bat_speed gap
                    between same-archetype shapes (Ohtani's two Uppercut Pulls) surfaces.
  - when_label      contexts where the shape is OVER-indexed vs the hitter's own baseline usage
                    (lift = P(shape | context)/P(shape) - 1), across count / location / pitch type /
                    base-out. Distinctive deployment, not raw share. Base-out is deployment-only.
  - grade           mean xrv_grade (0-100, 50 = league-avg swing) over the shape's swings: value of
                    deploying this shape across the contexts it is actually used in (design Part C.1).
  - matched_runs100 within-batter matched contrast (design Part C.2): mean xRV of this shape MINUS
                    the primary's, over count x height x pitch-type strata where BOTH appear, usage-
                    weighted, x100 = runs per 100 swings. NaN for the primary (it is the reference).
                    Base-out is NOT a value stratum (xRV excludes game state by design).
  - matched_n / matched_thin  support behind the contrast; thin flag when it's too sparse to trust.

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
    summ = pd.read_parquet(DATA / "cluster_summary.parquet")
    arch = pd.read_parquet(DATA / "shape_archetypes.parquet",
                           columns=["batter_id", "batter_stand", "cluster", "label", "archetype_name",
                                    "arch_confidence", "weight", "n"] + GEO)
    assign = pd.read_parquet(DATA / "cluster_assignments.parquet",
                             columns=["play_id", "batter_id", "batter_stand", "cluster"])
    xrv = pd.read_parquet(DATA / "xrv_swings.parquet", columns=["play_id", "xrv", "xrv_grade"])
    sm = pd.read_parquet(DATA / "swings_model.parquet",
                         columns=["play_id", "batter_stand", "balls", "strikes", "plate_x", "plate_z",
                                  "sz_top", "sz_bot", "pitch_type", "outs_when_up",
                                  "on_1b_id", "on_2b_id", "on_3b_id"])
    return summ, arch, assign, xrv, sm


def build_frame(assign, xrv, sm):
    """Per-swing frame: cluster + value + context bucket columns (count / location / pitch / base-out
    booleans for the when-label, and a coarse count x height x pitch stratum for the matched contrast)."""
    df = assign.merge(xrv, on="play_id", how="inner").merge(
        sm.drop(columns="batter_stand"), on="play_id", how="left")
    b, s = df.balls, df.strikes
    znorm = (df.plate_z - df.sz_bot) / (df.sz_top - df.sz_bot)
    # plate_x is absolute (catcher frame): pull-side/inside = negative for RHH, positive for LHH,
    # so flip RHH. + = pull-side / inside for both hands (validated vs bearing_angle).
    xpull = df.plate_x * np.where(df.batter_stand == "L", 1.0, -1.0)
    is_fb = df.pitch_type.isin(FASTBALLS)
    runners = df[["on_1b_id", "on_2b_id", "on_3b_id"]].notna()

    # when-label context buckets (rich; each needs only marginal support)
    df["ctx"] = None  # placeholder; buckets held in a dict of masks returned separately
    buckets = {
        "0-0": (b == 0) & (s == 0), "ahead": b > s, "behind": (s > b) & (s < 2),
        "2-strk": s == 2, "3-2": (b == 3) & (s == 2),
        "up": znorm > 0.66, "down": znorm < 0.33, "inside": xpull > 0.33, "away": xpull < -0.33,
        "vs FB": is_fb, "vs offspd": df.pitch_type.notna() & ~is_fb,
        "RISP": runners.on_2b_id | runners.on_3b_id, "bases empty": ~runners.any(axis=1),
        "2 out": df.outs_when_up == 2,
    }
    # matched-contrast stratum (coarse, needs JOINT support of two shapes): count3 x height3 x pitch2
    count3 = np.where(s == 2, "2strk", np.where(b > s, "ahead", "other"))
    height3 = np.where(znorm > 0.66, "up", np.where(znorm < 0.33, "down", "mid"))
    pitch2 = np.where(is_fb, "FB", "OS")
    df["strat"] = pd.Series(count3, index=df.index) + "|" + height3 + "|" + pitch2
    return df, buckets


def name_delta(cent, ref, sds):
    """Readable geometry phrase: top features (by |delta| in centroid-SD units) of cent vs ref.
    Always emits at least the single biggest delta, even if it clears no threshold, so the phrase
    is never uninformative (a shape can be far from the reference with each feature diffusely
    under the notable bar)."""
    scored = []
    for col, pos, neg, unit, fmt in PHRASE:
        d = cent[col] - ref[col]
        z = abs(d) / sds[col] if sds[col] > 0 else 0.0
        scored.append((z, f"{fmt.format(d)}{unit} {pos if d > 0 else neg}"))
    scored.sort(reverse=True)
    keep = [p for z, p in scored if z >= DELTA_Z][:3] or [scored[0][1]]
    return ", ".join(keep)


def over_index(g, cluster, base_weight, buckets):
    """Sorted (lift, bucket-name) list of contexts where the shape is over-indexed vs the hitter's
    own baseline usage (lift = P(shape | ctx)/P(shape) - 1 >= LIFT_MIN, with support)."""
    if base_weight <= 0:
        return []
    is_c = (g.cluster == cluster).to_numpy()
    hits = []
    for name, mask in buckets.items():
        m = mask.loc[g.index].to_numpy()
        n = int(m.sum())
        nc = int((m & is_c).sum())
        if n < MIN_CTX or nc < MIN_CTX_SHAPE:
            continue
        lift = (nc / n) / base_weight - 1.0
        if lift >= LIFT_MIN:
            hits.append((lift, name))
    hits.sort(reverse=True)
    return hits


def when_label(hits):
    """Full when-phrase with lift percentages (top 3)."""
    return ", ".join(f"{name} {lift * 100:+.0f}%" for lift, name in hits[:3])


def context_tag(hits):
    """Compact situational tag (top 3 over-indexed bucket names) to enrich the archetype label so
    same-archetype shapes separate. Top-2 collides when two shapes share their strongest contexts
    (e.g. Raleigh L's two Uppercut Pulls are both down/offspeed) — the 3rd bucket pulls them apart
    ('...down, offspd, full' vs '...down, offspd, inside'). Shapes deployed identically will still
    tie here; their geometry (name_delta) and value (grade) are what separate them then."""
    return ", ".join(name for _, name in hits[:3])


def matched_contrast(g, cluster):
    """Within-batter matched xRV contrast of `cluster` vs the primary (0), over strata where both
    appear. Returns (runs_per_100, matched_n, thin_flag)."""
    cur = g[g.cluster == cluster]
    prim = g[g.cluster == 0]
    num = den = matched_n = 0.0
    nstrat = 0
    for s, a in cur.groupby("strat"):
        b = prim[prim.strat == s]
        if len(a) >= MIN_STRAT and len(b) >= MIN_STRAT:
            diff = a.xrv.mean() - b.xrv.mean()
            num += diff * len(a)
            den += len(a)
            matched_n += len(a)
            nstrat += 1
    if den == 0:
        return np.nan, 0, True
    return (num / den) * 100.0, int(matched_n), (matched_n < MATCHED_THIN_N or nstrat < 2)


def build_cards(summ, arch, frame, buckets):
    # centroid-SD per feature (spread of shapes across the league) — the scale for "notable" deltas
    sds = {c: arch[c].std() for c in GEO}
    cohort_mean = {c: arch[c].mean() for c in GEO}   # league reference for naming the primary
    cent = arch.set_index(["batter_id", "batter_stand", "cluster"])

    grade = frame.groupby(["batter_id", "batter_stand", "cluster"]).xrv_grade.mean()
    rows = []
    for (bid, stand), g in frame.groupby(["batter_id", "batter_stand"], sort=False):
        prim = cent.loc[(bid, stand, 0)]
        weights = g.cluster.value_counts(normalize=True)
        for c in sorted(g.cluster.unique()):
            cc = cent.loc[(bid, stand, c)]
            ref = cohort_mean if c == 0 else prim
            r100, mn, thin = (np.nan, 0, False) if c == 0 else matched_contrast(g, c)
            hits = over_index(g, c, float(weights.get(c, 0.0)), buckets)
            tag = context_tag(hits)
            arch = cc["archetype_name"]
            rows.append({
                "batter_id": bid, "batter_stand": stand, "label": cc["label"], "cluster": int(c),
                "role": "primary" if c == 0 else "secondary",
                "archetype_name": arch,
                "context_tag": tag,
                # cluster 0 is by definition the hitter's primary swing, so label it "Primary"
                # rather than by archetype+situation (archetype_name still holds the true archetype).
                "archetype_detailed": "Primary" if c == 0 else (f"{arch} · {tag}" if tag else arch),
                "arch_confidence": cc["arch_confidence"],
                "usage": round(float(weights.get(c, 0.0)), 3), "n": int((g.cluster == c).sum()),
                "name_delta": name_delta(cc, ref, sds),
                "when_label": when_label(hits),
                "grade": round(float(grade.loc[(bid, stand, c)]), 1),
                "matched_runs100": None if c == 0 else round(r100, 2),
                "matched_n": mn, "matched_thin": thin,
                "bat_speed": round(float(cc["bat_speed"]), 1),
            })
    return pd.DataFrame(rows)


def write_catalog(cards):
    L = ["# Swing ID cards (Layer 2) — worked examples\n"]
    for name in ["Aaron Judge", "Shohei Ohtani", "Luis Arraez"]:
        d = cards[cards.label.str.startswith(name)].sort_values("cluster")
        if d.empty:
            continue
        u = d.iloc[0]
        L.append(f"\n## {u.label}  ·  {int(d.n.sum()):,} swings  ·  {len(d)} shapes\n")
        for _, r in d.iterrows():
            star = "* PRIMARY" if r.role == "primary" else "  secondary"
            ref = "vs league" if r.role == "primary" else "vs primary"
            desc = r.archetype_name if r.role == "primary" else r.archetype_detailed  # star already says PRIMARY
            L.append(f"**{star} — {desc}**  ({r.usage*100:.0f}% usage, {r.bat_speed:.0f} mph)")
            L.append(f"  - shape: {r.name_delta} ({ref})")
            L.append(f"  - when : {r.when_label or '(no strong over-index)'}")
            val = f"grade {r.grade:.0f}"
            if r.role == "secondary" and pd.notna(r.matched_runs100):
                flag = " *(thin)*" if r.matched_thin else ""
                val += f"  ·  vs primary {r.matched_runs100:+.2f} runs/100 (n={r.matched_n}){flag}"
            L.append(f"  - value: {val}\n")
    (DATA / "shape_cards_catalog.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    summ, arch, assign, xrv, sm = load()
    frame, buckets = build_frame(assign, xrv, sm)
    print(f"Frame: {len(frame):,} swings with value + context")
    cards = build_cards(summ, arch, frame, buckets)
    cards.to_parquet(DATA / "shape_cards.parquet", index=False)
    print(f"Wrote shape_cards.parquet — {len(cards)} shape rows across "
          f"{cards.groupby(['batter_id','batter_stand']).ngroups} units")
    write_catalog(cards)


if __name__ == "__main__":
    main()
