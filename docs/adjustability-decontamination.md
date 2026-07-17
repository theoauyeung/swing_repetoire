# Adjustability & location decontamination — design options

**Status:** v2 BUILT (2026-07-17) — `context_response.py` now implements the cluster-free,
per-dial-slope approach (options #4 + the conditioning logic of #1). Options #2 (residualize) and
#3 (approach-shape clustering) remain unbuilt alternatives, kept here for reference. See also
`docs/research-design.md` Part D.

## The problem
`context_response.py` (v1) measures `I(context; shape)` and found the dependence is **almost entirely
pitch location** (`resp_loc` ≈ 0.22) with **near-zero count-responsiveness** (`resp_count` ≈ 0.006).

**Mechanism (the leakage):** shape = 5 mechanics features *at contact*, and two of them —
`vert_attack_angle` and especially `horz_attack_angle` — are **kinematically forced** by where the
ball is. To meet a low-away pitch the bat *must* be at a certain angle; the hitter doesn't choose it.
So `I(location; shape)` is inflated by a physical identity and reads "the pitch dictated the bat
angle" as if it were "the hitter adjusted." The variance decomposition (research-design Part A)
predicts this: `horz_attack_angle` batter-ICC = 0.054, `vert_attack_angle` = 0.106 (least trait-like,
most location-driven), and they have the **largest raw SDs** (17°, 10°), so they dominate the
clustering. The shape labels are substantially *location labels in a mechanics costume*.

## The key identification insight
On the **location axis itself, "forced" and "chosen" are not separable** — a hitter who *chooses* to
go oppo on outside pitches produces the same shape↔location correlation as pure physics. So
"decontaminated location-responsiveness" is not identifiable from this data.

**Count is the clean instrument.** Count does not mechanically change the geometry required to hit a
pitch at location X, so any shape change that covaries with **count, holding location (and pitch type)
fixed**, is necessarily volitional. Reframe the estimand:

> Adjustability = does shape change with the *strategic* context (count) that does **not** dictate
> geometry, **after** the geometry-dictating context (location/pitch) is held fixed?

Base-out / game state (runners, outs) is a second strategic axis with the same logic — volitional if
it moves shape within fixed location — but empirically weak (see diagnostic).

## Diagnostic verdict (experiments/adjustability_count_diagnostic.py, 2024-25)
Measured the count effect **directly on the raw dials**, holding hitter + location bin fixed
(within-FE slope, `d` = slope / cohort SD):

| feature | 2-strike effect (d) | RISP effect (d) |
|---|---|---|
| bat_speed | **−0.34** (~−1.9 mph) | −0.055 |
| swing_length | **−0.15** (shorter) | −0.040 |
| vert_attack_angle | −0.12 | −0.037 |
| swing_path_tilt | −0.08 | −0.013 |
| horz_attack_angle (forced) | −0.06 | −0.027 |

**Conclusion: the volitional signal is REAL but HIDDEN.** Hitters clearly ease bat speed and shorten
the swing at 2 strikes (holding location fixed); contact types (Kwan, Hoerner, Freeman) do it most
(bat_speed d −0.33 to −0.57). `resp_count ≈ 0` in v1 is a **substrate artifact** — the count signal
lives in bat_speed/swing_length, which the 5-feature clustering under-weights vs. the 17°/10° angle
axes. So the fix is to change *what shape is made of / how we measure the count effect*, not to
abandon adjustability. RISP is ~5× weaker than count → keep as exploratory, not headline.

## Options to decontaminate (come back to these when implementing)

### 1. Partial out location with a nested model  — *cheapest, do first*
Compare a classifier predicting shape from `{location, pitch_type}` vs `{location, pitch_type, count}`;
the **out-of-fold log-loss lift from adding count** = `I(count; shape | location, pitch)` =
decontaminated adjustability. Model-based conditional MI, so it dodges the cell-sparsity of hard
stratification. Reuses the `context_response.py` classifier. **Make this the headline estimand.**
- Pros: cheap, principled, sample-efficient, reuses infra. Cons: relies on the model capturing the
  location→shape surface (use continuous plate_x/z).

### 2. Residualize the shape substrate first
Fit `feature ~ f(location, pitch_type)` league-wide; keep the **residual** ("deviation from the
geometry the pitch dictated"), then cluster / measure responsiveness on residuals.
- Pros: removes the location channel before anything else. Cons: for `horz_attack` almost no batter
  signal remains after residualizing (ICC 0.054) → residual clusters may collapse (itself informative).

### 3. Change what "shape" is made of for adjustability  — *strongest per diagnostic*
Define a parallel **"approach-shape"** on the trait-stable, less-forced dials only —
`swing_path_tilt` (ICC 0.44), `swing_length` (0.27), `bat_speed` (0.13) — dropping the location-forced
attack angles. The diagnostic shows count-adjustment lives exactly here (bat_speed, length). Precedent:
the archetype lexicon already uses 4 features, not the clustering's 5 — a purpose-specific feature set
is acceptable.
- Pros: attacks the root; surfaces the count signal the current clusters bury. Cons: new
  clustering/def just for this question; more moving parts.

### 4. Reframe from clusters to approach dials  — *most interpretable*
Drop the multivariate cluster for adjustability; measure it directly: *does `swing_length` shorten /
`bat_speed` ease / `swing_path_tilt` flatten with two strikes (and with base-out), holding location
fixed?* Per-hitter within-location FE regression per dial (exactly the diagnostic, productionized).
- Pros: dead-simple, unambiguously volitional, coach-legible ("he shortens up with 2 strikes"). Cons:
  leaves the cluster framing behind (could complement it).

## Recommended sequencing
1. **Done:** diagnostic → signal is real but hidden (substrate problem).
2. **Done (v2):** the diagnostic reordered the plan — since the *clusters* are the wrong substrate,
   option #1-on-clusters would still be muted, so we went straight to **#4 (direct per-dial slopes),
   cluster-free**, carrying option #1's conditioning logic (within location × pitch type). Count is
   the headline (`count_adj` + per-dial `cnt_*_d`); base-out is a secondary axis. Only bat_speed /
   swing_length / swing_path_tilt are used; the forced attack angles are excluded outright.
3. **Optional next:** a residualized-angle robustness variant (#2) to show the forced features, net
   of location, still don't move with count — strengthens the "we didn't just cherry-pick dials" point.
4. **Optional:** an approach-shape clustering (#3) only if a cluster-based narrative is wanted to sit
   beside Facet-1 for continuity.

## Robustness: residualize on location, re-measure (2026-07-17, option #2 run)
League-residualized each feature on continuous location + pitch (`feat ~ px + pz + px² + pz² + px·pz +
pitch_group`), then re-measured the within-hitter count slope. Result (d per strike):

| feature | within-loc slope | residualized-on-location | reading |
|---|---|---|---|
| bat_speed | −0.20 | −0.16 | volitional ✓ (survives) |
| swing_length | −0.13 | −0.13 | volitional ✓ (survives) |
| swing_path_tilt | −0.03 | ~0.00 | trait/plane — **stable, not adjusted** |
| vert_attack_angle | −0.13 | −0.13 | survives → **ambiguous** (maybe volitional plane change, not pure forced) |
| horz_attack_angle | −0.10 | −0.09 | least trait — likely partly forced |

Two takeaways: (1) **count effects are NOT a location artifact** — they survive residualizing on
location+pitch, which *validates* the conditional design (v2 is measuring something real). (2) The
crisp "dials volitional / angles forced" split is only partly true: `swing_path_tilt` barely moves
with count (hitters keep their swing-plane identity), while `vert_attack_angle`'s count-shift is NOT
explained by location. **Feature-set implication (open decision):** the composite could lean on
`bat_speed` + `swing_length` (the two clear dials), consider adding `vert_attack_angle` (ambiguous
but real signal), and `swing_path_tilt` ≈ 0 could be dropped or kept as a "plane-stability" descriptor.
Also note: controlling for pitch type (v2's `loc_pitch` cells) shrinks bat_speed's raw count slope
from ~−0.34 (location-only) to ~−0.20 — some of the count↔bat_speed link ran through 2-strike
pitch-mix; v2 correctly nets that out.

## Meta-point for the paper
This is not just a metric bug: **at-contact geometry may be the wrong substrate for "adjustability."**
Volition shows up earlier (decision, timing, effort) and in the trait dials, not the pitch-forced
angles. Decide whether the paper's adjustability claim is about *swing shape* (contaminated, modest)
or *approach* (cleaner, and closer to what a coach means by "he adjusts"). Ties to Limitation #1
(contact-point-only geometry).
