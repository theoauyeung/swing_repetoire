# Swing Repertoire Research — Design & Implementation Plan

## Context

Statcast bat tracking (2024+) exposes swing *geometry at ball-bat intercept* for every
competitive swing. This lets us move past aggregate "attack angle" averages and ask two
questions Driveline has not been able to answer before:

1. **Swing-shape value.** Each hitter deploys a *repertoire* of distinct swing shapes.
   What is the run value of each shape, conditioned on the situation it's used in (count,
   pitch location, pitch type, base-out state)? This yields a composite metric that grades
   a swing purely by its shape-in-context, independent of whether that particular ball
   happened to be squared up.
2. **Repertoire diversity.** Is a *wider* repertoire beneficial? Does carrying more swing
   shapes give a hitter more *adjustability* to different pitches/situations — and does that
   adjustability actually translate into better outcomes, or is "diversity" just
   inconsistency wearing a nicer name?

This document is the pre-implementation design: data reality, methodology, limitations, and
the conditions we must satisfy before writing modeling code. It reflects four design
decisions the user has confirmed (see below).

---

## Confirmed design decisions

| Fork | Decision | Consequence for the plan |
|---|---|---|
| Cluster scope | **Strictly per-(batter, stand) GMM** (independent, per-unit *k*) | Clusters are idiosyncratic and *not* comparable across units. The clustering unit is `(batter_id, batter_stand)`: a switch hitter's L and R swings are different movements, so each stance is clustered — and enters Facet 2 — as its own "player" (Cal Raleigh L vs Cal Raleigh R). Only `horz_attack_angle` is handedness-mirrored; the other four features are not, so an unsplit switch hitter's GMM separates on stance rather than shape. The diversity analysis (Facet 2) must use unit-level scalar metrics + a post-hoc reference map, not shared cluster IDs. |
| Value target | **Bespoke xRV built from scratch** | We define and fit our own per-swing run-value model. `delta_run_exp` / `pitch_values.ipv` become *validation anchors*, not the target. |
| Causal ambition | **Conditional xRV + within-batter matching** | Predictive xRV surface, plus a within-hitter quasi-experiment comparing a batter's own shapes in matched contexts. Causal-flavored, honest about limits. |
| Shape features | **Include bat speed; exclude ball-bat intercept location** | Shape = 5 mechanics features (`swing_path_tilt, swing_length, bat_speed, vert_attack_angle, horz_attack_angle`). Intercept coordinates are pitch-location/timing artifacts (see variance decomposition below), not mechanics, and are excluded from the shape vector. |

---

## Data foundation (verified against live `mlb_db`, 2026-07-01)

All swing-shape data lives in **`pbp_raw`** and is **MLB-only** (`level_id = 1`). No minor-league
or amateur bat tracking exists. Confirmed non-null coverage of tracked swings:

| Season | Tracked swings (`bat_speed` non-null) | Notes |
|---|---|---|
| 2023 | ~149,665 | Partial — treat as pilot/exclude by default |
| 2024 | ~322,577 | First full season |
| 2025 | ~343,282 | Full season |
| 2026 | ~172,794 | Partial (through ~July 1) |

**Bat tracking is present on whiffs and fouls, not only on contact** — this is the single
most important enabling fact. In 2025 MLB: `pitch_outcome='S'` (swinging strikes + fouls +
called strikes) has 220K tracked swings; balls-in-play (`'X'`) 123K; takes/`'B'` 0. So swing
shape can be modeled largely independent of the contact outcome, which is what makes the
value question tractable.

**Per-batter feasibility (2025 MLB):** 535 hitters ≥100 tracked swings, **450 ≥200, 389 ≥300,
298 ≥500** (median 144 across all 991 who swung at least once). Pooling 2024–2026 roughly
triples per-hitter counts. Per-batter GMM is viable for regulars; bench/call-up hitters will
be sample-limited.

### Columns we will use

**Shape features (`pbp_raw`, at intercept):**
`bat_speed`, `swing_length`, `vert_attack_angle`, `horz_attack_angle`, `swing_path_tilt`,
`ball_bat_intercept_x`, `ball_bat_intercept_y`, `ball_intercept_z`.
(`ball_bat_intercept_y` = contact depth → a **timing proxy**; `ball_bat_miss` = miss distance.)

**Pitch/situation context:**
`balls`, `strikes` (`pbp_raw`); `plate_x`, `plate_z`, `plate_zone` (`pbp_raw`); `pitch_type`
(`pbp_raw`); `outs_when_up` (`pbp_raw`); `on_1b_id`, `on_2b_id`, `on_3b_id`
(`pbp_descriptions`); `batter_stand`, `batter_id` (`pbp_raw`); `players.sz_top`/`sz_bot`,
`height` for zone normalization.

**Outcome / value signals:**
`delta_run_exp` (`pbp_descriptions`, per-pitch RE change — available on every swing),
`woba`/`xwoba` (`pbp_raw`, BIP only), `exit_velo`/`launch_angle`/`bb_type`/`bearing_angle`
(`pbp_raw`), `pa_outcome`/`pitch_outcome`. Cross-check tables: `pitch_values.ipv*`,
`xwoba_components`, `batting_model_expectations_new`.

**Swing flags (`pbp_descriptions`):** `is_swing`, `is_whiff`, `is_contact`, `is_bip`.

**Bonus:** `swing_visuals` (per batter/year `video_link`) for qualitative cluster validation;
`hitter_bat_speed_stats` (pre-aggregated) as a sanity reference.

### Connection
`mlb-db-analysis` skill pattern — miniforge Python + `mysql-connector-python`, `BIOMECH_DB_*`
creds from `~/.claude/.env`, database `mlb_db`, read-only user. Full column reference:
`~/.claude/skills/mlb-db-analysis/docs/schema.md`.

---

## Methodology

### Part A — Per-batter swing-shape clustering (GMM)

**Feature vector (per swing) — 5 mechanics features:** `swing_path_tilt`, `swing_length`,
`bat_speed`, `vert_attack_angle`, `horz_attack_angle`.

**Excluded from shape (and why):** the ball-bat intercept coordinates
(`ball_bat_intercept_x`, `ball_intercept_z`, `ball_bat_intercept_y`) are *where/when the ball
was met*, not swing mechanics. A variance decomposition on 335K 2025 MLB swings (see below)
confirms `intercept_x` is 99.4% explained by pitch location with ~0 stable batter signal, and
`intercept_z` is 97.8% pitch height. `intercept_y` (contact depth) is a **timing/approach
signature** (only 35% pitch-driven, 22% stable batter tendency) — retained as a *separate*
descriptor for interpretation and the within-batter analysis, but **not** part of the shape
vector. (It is measured *at contact*, so it is a product of the swing, not exogenous pre-swing
context — it must not enter the xRV conditioning set as a mediator.)

#### Feature selection — variance decomposition (2025 MLB, 335K swings, 535 hitters ≥100 swings)

For each feature: *reaction share* = variance explained by pitch location + pitch type;
*mechanical fingerprint* = stable between-batter signal remaining after removing pitch context.

| Feature | Reaction (pitch R²) | Fingerprint (batter ICC, net of pitch) | Role |
|---|---|---|---|
| `swing_path_tilt` | 0.321 | **0.439** | Shape (strongest trait) |
| `swing_length` | 0.350 | 0.274 | Shape |
| `bat_speed` | 0.123 | 0.126 | Shape (note: more state than trait) |
| `vert_attack_angle` | 0.192 | 0.106 | Shape |
| `horz_attack_angle` | 0.263 | 0.054 | Shape (weakest trait) |
| `ball_bat_intercept_y` | 0.350 | 0.221 | Timing descriptor / context, not shape |
| `ball_intercept_z` | 0.978 | 0.192 | Excluded (pitch height) |
| `ball_bat_intercept_x` | 0.994 | 0.025 | Excluded (pitch location) |

Note `bat_speed`'s low trait share (0.126): included per decision, but expect it to split
"A-swings vs protective/two-strike swings" as much as it separates geometry.

**Preprocessing:**
- Normalize height-dependent features by the strike zone: express `plate_z` and
  `ball_intercept_z` relative to `sz_top`/`sz_bot`. Put horizontal features in a pull frame
  (+ = pull, both hands) so L/R hitters share a frame — but the transform differs by metric
  (validated vs `bearing_angle`, worklog 2026-07-09): `horz_attack_angle` is *batter-relative*, so
  the pull frame is a **uniform negation** (no per-hand mirror); `plate_x` is *absolute* (catcher
  frame), so it needs a **real per-hand flip** (flip RHH). An early bug applied one shared per-hand
  mirror to both, leaving RHH attack-direction inverted.
- Standardize (z-score) each feature *within batter* before clustering (a batter's own shape
  spread, not league spread, defines their clusters).
- Outlier handling: drop physically impossible values (bat_speed <30 or >95 mph, etc.) using
  `mlb-db-analysis/docs/gotchas.md` filters; robust-scale to limit leverage of extreme swings.

**Clustering:**
- `sklearn.mixture.GaussianMixture`, full covariance, per batter.
- **Model selection per hitter via BIC** over k=1..k_max, then a **post-BIC merge** of
  near-duplicate components. BIC over-segments at large n (its ln(n) penalty is too weak to stop
  it splitting trivial density bumps into large-but-near-identical components — e.g. Ohtani's twin
  "Level Center" swings at 75 vs 78 mph, each ~28% usage). We merge component pairs closer than
  `MERGE_SEP=2.0` within-cluster-SD Mahalanobis (closest pair first, iteratively) so each surviving
  cluster is a genuinely distinct shape. This is *separation*-based, deliberately **not** an
  occupancy floor — the phantom components are large, so a size floor cannot catch them. (ICL was
  ruled out: its entropy penalty overcorrects and collapses ~everyone to k=1, because swing shapes
  are a continuum.) The post-merge *k* is the headline "repertoire size" (mean ≈1.9, median 2).
- **Minimum threshold:** cluster only units with ≥150 tracked swings (2024–26 pooled) *per
  `(batter, stand)`*. Lowered from the original ≥300 pooled-per-batter rule so a switch hitter's
  weaker side still qualifies (e.g. Abraham Toro R = 282 swings); the identifiability cap
  `k_max = n // 20` still permits up to 7 shapes at n=150. Report the qualifying population;
  everyone else is described but not clustered. (Current cohort: 703 units, incl. 47 switch
  hitters split two ways.)
- Pool seasons for stability, but include a season covariate check for within-hitter drift
  (a hitter who changed their swing across years is a real signal, not noise).

**Outputs:** per-hitter cluster assignments per swing; cluster centroids + covariances;
per-hitter *k*; usage weights (mixture π). Each swing now carries a `(batter_id, cluster_id)`.

### Part B — Bespoke xRV model (per-swing run value)

**Estimand:** the expected run-value contribution *of the swing*, given the pitch and
situation — `E[Δrun_value | pitch, context, swing occurred]`. We want value attributable to
executing a swing (any shape) on this pitch, *net of situational leverage*, so shape quality
in Part C is a fair comparison rather than a reward for swinging in high-leverage counts.

**Training signal:** per-pitch run value on the swing. Primary construction is an RE24-style
delta on the swing outcome (whiff → strike cost; foul → strike/no-cost per count; BIP →
outcome run value). We build this ourselves (per the "bespoke" decision) but **validate it
row-for-row against the existing `delta_run_exp`** and calibrate aggregate hitter values
against `pitch_values.ipv` — if our xRV can't reproduce those within tolerance, the
construction is wrong.

**Model:** gradient-boosted trees (LightGBM/XGBoost) for the conditional surface
`E[xRV | context features]`, because the conditioning set (count × location × pitch type ×
base-out × handedness) is high-dimensional and interaction-heavy and cell means would be far
too sparse. Features: `balls`, `strikes`, normalized `plate_x/plate_z`, `pitch_type`,
`outs_when_up`, base-state (three occupancy flags), `batter_stand`. Report calibration and
out-of-sample (season-held-out) performance. A hierarchical Bayesian variant (via the
`pymc-bambi-hierarchical-convergence` skill) is the fallback if we need principled shrinkage
for sparse contexts.

### Part C — Swing-shape value (the composite grade)

Two layers:

1. **Conditional (predictive) grade.** For each `(batter, cluster)`, estimate mean xRV
   conditional on context, i.e. the value of *deploying that shape* across the contexts it's
   actually used in, and standardized against a context-matched baseline. This is the
   "quality of a swing based purely on its shape-in-context" metric the user described.

2. **Within-batter matched comparison (causal-flavored).** The confound: the pitch drives
   *both* the shape and the outcome (a hitter fooled badly produces an emergency shape *and*
   a bad result — common cause). To get closer to causal, compare a hitter's **own** shapes
   within matched context strata (same count bucket, location bin, pitch-type group,
   base-out). Where a hitter uses ≥2 shapes in the same stratum, the xRV difference is a
   within-hitter, within-context contrast that nets out hitter quality and situational
   leverage. Aggregate these matched contrasts to answer "is shape A actually better than
   shape B *for this hitter, here*." Report where matching support is too thin to conclude.

### Part D — Repertoire diversity & adjustability (Facet 2)

Because clusters are per-batter and non-comparable, diversity is measured with **batter-level
scalars that don't require shared cluster IDs:**

- **Repertoire size:** selected *k* (effective number of shapes).
- **Usage entropy / effective shapes:** Shannon entropy of mixture weights → "effective
  number of shapes actually used" (a hitter with k=5 but 95% in one cluster is effectively
  monolithic).
- **Shape dispersion:** total volume / mean pairwise centroid distance in standardized shape
  space (how *spread out* the repertoire is, not just how many pieces).

**Adjustability** is distinct from raw diversity and is the crux: diversity is only valuable
if shapes are deployed *appropriately*. Operationalize as:

- **Context-responsiveness:** mutual information (or a classifier's skill) between pitch
  context and which shape the hitter selects. High MI = the hitter reshapes their swing to
  the pitch; low MI = they swing the same way regardless.
- **Adjustment payoff:** does higher context-responsiveness predict higher xRV, *after*
  controlling for overall hitter quality and repertoire size? This is the "is adjustability
  actually helping" test — a hitter-level regression of performance on
  {repertoire size, usage entropy, context-responsiveness} with hitter-quality controls.

**Key guardrail:** separate *adaptive* diversity (shape covaries with context → good) from
*noise* diversity (shape varies randomly → inconsistency, likely bad). The MI/payoff split
above is exactly what distinguishes them; we report both so a wide-but-random repertoire
isn't mistaken for a skill.

---

## Limitations & threats to validity (read before building)

1. **Contact-point-only geometry.** We have swing summary metrics *at the contact/near-miss
   point*, not the full 3D bat trajectory. "Shape" = swing metrics at intercept, not the entire
   swing arc; two different arcs producing the same metrics are indistinguishable. The intercept
   *location* coordinates are excluded from shape (they are ~98–99% pitch location; see Part A
   variance decomposition), but the retained attack-angle features are still *measured at
   contact*, so they carry a residual pitch-height confound (`vert_attack_angle` moves with
   where in the arc contact occurs). Treated as shape-as-deployed and handled at the value stage
   (Part C.2), not by residualizing the clustering.
2. **Endogeneity / common-cause confounding.** Pitch recognition and timing drive both shape
   and outcome. The within-batter matched design (Part C.2) mitigates but does not eliminate
   this — unobserved "how fooled was the hitter" still lurks. `ball_bat_intercept_y` (depth)
   is our only timing proxy. State causal claims cautiously.
3. **Selection on swinging.** Everything is conditional on a swing occurring; we do not model
   take decisions. The metric grades swings, not plate discipline. A hitter could have great
   shape values and poor overall production via bad swing *decisions*.
4. **bat_speed-in-shape coupling.** Per the user's choice, effort is baked into shape, so an
   "A-swing" and a defensive version of the same geometry cluster apart. This is intended but
   means clusters partly encode intent/effort, not pure geometry — interpret accordingly.
5. **Cell sparsity in conditioning.** cluster × count × location × pitch type × base-out
   explodes combinatorially; this is why Part B uses a model, not cell means, and why Part C.2
   flags thin matching support instead of over-claiming.
6. **Per-batter incomparability (by design).** Cluster #2 for Hitter A ≠ cluster #2 for
   Hitter B. All cross-hitter analysis uses the scalar metrics in Part D, plus an optional
   post-hoc global reference embedding to *label* clusters ("this looks like a steep
   uppercut") without making them the unit of cross-hitter comparison.
7. **Small-era / drift.** Only ~2.5 full seasons exist. Year-over-year swing changes, the
   evolving measurement pipeline, and rule/ball changes limit longitudinal claims. 2023 is
   partial — exclude by default.
8. **Measurement noise & missingness.** ~18% of BIP and a larger share of swings lack
   tracking; missingness may be non-random (e.g., extreme swings). Characterize
   missingness before trusting per-cluster means.
9. **Reused-value leakage.** Since we validate xRV against `delta_run_exp`, we must not then
   treat agreement as independent confirmation — it's a calibration check, not validation of
   the causal contrast.
10. **Small handedness lean in diversity metrics.** L-stand units average slightly more effective
    shapes and dispersion than R-stand units (partial corr with an L-flag ≈ +0.11 net of
    `n_swings`; not a sample-size confound, and provably *not* a `horz_attack_angle`-mirror
    artifact — Mahalanobis distance is invariant to the sign flip). Small (~0.18 effective shapes,
    ~0.06 dispersion) but real; plausibly a platoon/approach difference or a batter-side
    measurement asymmetry. Consider including `batter_stand` as a covariate in the Facet 2 payoff
    regression rather than assuming L/R units are exchangeable.

---

## Conditions to satisfy before modeling

- [ ] **Missingness audit** — is bat-tracking-present vs absent random w.r.t. pitch type,
  location, count, exit velocity? Determines whether complete-case clustering biases results.
- [ ] **Feature sanity & units** — confirm sign conventions (attack angle up = +?), units, and
  outlier bounds for all 8 shape features against `gotchas.md` and a hand-checked hitter.
- [ ] **xRV validation harness** — reproduce `delta_run_exp` and aggregate `ipv` before using
  our xRV anywhere downstream.
- [ ] **Qualifying population frozen** — lock the ≥150-swing-per-(batter, stand) unit list and
  season pooling so every stage runs on the same cohort.
- [ ] **Matching support map** — quantify, per hitter, how many context strata contain ≥2
  shapes with enough swings; if this is thin league-wide, Part C.2 scales back to
  partially-pooled estimates.
- [ ] **Cluster stability check** — bootstrap/seed-stability of per-hitter GMM (does k and the
  centroids reproduce?) before interpreting clusters as real "shapes."

---

## Implementation structure

```
swing-repertoire/
  README.md
  docs/
    research-design.md       # this document
    worklog.md               # per docs protocol
  data/                      # cached extracts (gitignored); never commit athlete PII
  src/
    extract.py               # mlb_db → parquet; qualifying-cohort filter
    features.py              # normalization, zone-relative, handedness mirror, scaling
    cluster.py               # per-batter GMM + BIC selection + stability
    xrv.py                   # bespoke per-swing run value + validation vs delta_run_exp
    value_model.py           # E[xRV|context] GBM; conditional shape grades
    within_batter.py         # matched-context contrasts (Part C.2)
    diversity.py             # repertoire scalars + adjustability/MI + payoff regression
  notebooks/                 # EDA, missingness audit, cluster viz (uses swing_visuals links)
  reports/                   # branded outputs (driveline-baseball-design)
  tests/
```

**Pipeline order:** extract → features → (missingness audit) → cluster → xrv (+validate) →
value_model → within_batter → diversity → reports.

**Stack:** Python; `mysql-connector-python` (extract), `pandas`/`pyarrow`, `scikit-learn`
(GMM), `lightgbm` (xRV surface), `scipy`/`numpy` (entropy, MI), optionally
`pymc`/`bambi` for the hierarchical fallback. Use `uv` venv per global conventions.

**Reuse (per org "reuse before rebuild"):** `mlb-db-analysis` skill for all extraction/schema;
`pymc-bambi-hierarchical-convergence` skill if we go hierarchical; `driveline-baseball-design`
for report styling; `autoresearch` skill if we sweep clustering/model hyperparameters.

---

## Phased milestones

1. **M1 — Data & EDA:** extract qualifying cohort, missingness audit, feature distributions,
   confirm units/signs. Deliverable: EDA notebook + frozen cohort.
2. **M2 — Clustering:** per-batter GMM + BIC + stability. Deliverable: cluster catalog with
   `swing_visuals` spot-checks for 5–10 hitters.
3. **M3 — xRV:** build + validate bespoke per-swing xRV against `delta_run_exp`/`ipv`.
4. **M4 — Shape value:** conditional grades + within-batter matched contrasts (Facet 1).
5. **M5 — Diversity/adjustability:** repertoire scalars, context-responsiveness, payoff
   regression (Facet 2).
6. **M6 — Reporting:** branded writeup, limitations, hitter-level examples.

---

## Verification

- **xRV correctness:** correlation and row-level agreement with `delta_run_exp`; aggregate
  hitter xRV vs `pitch_values.ipv` within tolerance (documented in `xrv.py` tests).
- **Clustering realism:** for a handful of hitters, pull `swing_visuals.video_link` and
  confirm clusters correspond to visibly distinct swings (not measurement artifacts).
- **Diversity metric sanity:** hitters known for adjustability should score high on
  context-responsiveness; one-note power hitters should score low — eyeball against scouting
  priors before trusting the payoff regression.
- **Causal caution check:** every claim from Part C.2 reported with matching support N and
  framed as within-hitter/within-context, never as unconditional "shape X is best."
- **Held-out season:** train on 2024–25, sanity-check stability of grades/clusters on 2026.
