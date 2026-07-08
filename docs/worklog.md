# Worklog

## M2 viz — swing-card attack direction: true field orientation (2026-07-08)

- The ATTACK DIRECTION panel (`_p_dir` in `cluster_results.ipynb`) drew from `horz_attack_angle_pull`
  (the mirrored "+ = pull" frame), so it always put pull on screen-left. Correct for RHH (pull = LF),
  backwards for LHH (who pull to RF). Now flipped per handedness: `s = -1` for L, arrow angle
  `= radians(haa)*s`, and the PULL/OPPO labels swap sides so the panel reads in the hitter's real
  field orientation (RHH pull = left/LF, LHH pull = right/RF). `stand` is passed through the per-row
  frame (`g['stand'] = stand`). The magnitude and "N° PULL/OPPO" text are frame-invariant (unchanged).
  Only the direction panel changed — the side-view fan and other panels are untouched.
- Verified headlessly: arrow x-tip sign correct for all four L/R × pull/oppo cases; rendered Arraez (L,
  OPPO-left/PULL-right) and Aaron Judge (R, PULL-left/OPPO-right) cards and eyeballed both. Updated the
  column-legend markdown to describe the handedness-aware orientation.
- Follow-up: also mirror the SWING PATH fan for lefties (`_p_fan` negates the fan's x-coords when
  `stand == 'L'`), so a lefty's side view reads as the true mirror image (contact on screen-left,
  hands trailing right) instead of looking identical to a RHH. RHH fan unchanged (contact screen-right).
  Re-rendered both cards to confirm. Only left/right screen orientation changes — tilt/length/angle
  magnitudes are untouched.
- Bug fix in `_fan`: the barrel-at-contact landed on inconsistent sides across a hitter's shapes
  (some fans hooked so the contact point sat at the left end, others the right). Root cause: `flip`
  compared the contact projection only to the arc's *start* point, but these reconstructed arcs curve
  past contact, so mid-arc points exceeded it on either side. Fixed by anchoring against the arc's
  projected midpoint (`flip = +1 if proj[-1] >= 0.5*(proj.min()+proj.max()) else -1`), so the bold
  bat-at-contact is consistently on the right (then mirrored left for LHH). Verified: all 4 Judge
  shapes now barrel top-right, all 7 Arraez shapes barrel top-left. Also clarified the legend that the
  bold tan bar = bat AT contact and the faint fan = the sweep INTO contact.
- **Removed the SWING PATH fan column entirely** (per user: uninterpretable and incorrect). It was a
  stylized reconstruction of a sweep from contact-point-only metrics, not a measured trajectory, and
  read misleadingly. Dropped `_fan`/`_p_fan`, the `R0`/`BATLEN` constants, the `swing_length` aggregate,
  and the COLS entry; cards are now 4 columns (attack angle, attack direction, swing tilt, bat speed),
  all direct functions of contact-point metrics. Legend updated to drop the fan row + reconstruction caveat.

## Tooling — single shared `driveline` Python env (2026-07-07)

- **Problem:** recurring "import X could not be resolved" in the IDE. Root cause was config, not
  packages: no project venv existed (scripts only ran via directly-invoked global miniforge), and
  `.vscode/settings.json` set `python-envs.defaultEnvManager: system` (IDE grabbed a bare system
  Python) plus a dead macOS `extraPaths` (`/Users/UserName/...`). Pylance resolved against the
  wrong interpreter.
- **Fix (per user: one shared env for all workflow):** created `~/.venvs/driveline` (uv, miniforge
  CPython 3.13, `--prompt driveline`); installed pandas, pyarrow, scikit-learn, lightgbm, scipy,
  numpy, mysql-connector-python, requests, matplotlib, **tabulate** (needed by `cluster.py`
  `.to_markdown()`), ipykernel. Registered Jupyter kernel "Python (driveline)". Rewrote
  `.vscode/settings.json` to `python.defaultInterpreterPath = ${userHome}/.venvs/driveline/...`
  and removed the dead `extraPaths` / `system` env-manager lines. Updated README + CLAUDE.md env
  sections (previously referenced a conda `swing_repertoire` env / project-local `uv venv`).
- Verified all imports resolve on that interpreter. Other kernels (miniforge3, pitcher-bat-path)
  left intact; just select "Python (driveline)".

## M2 — handedness-lean check + label polish (2026-07-07)

- **Investigated the L-stand skew in the leaderboards.** Not a mirror artifact: `shape_dispersion`
  is computed on within-unit z-scored features, and Mahalanobis distance is invariant to a
  coordinate sign flip, so the `horz_attack_angle` mirror provably cannot affect it. L vs R
  dispersion distributions are nearly identical (mean 2.22 vs 2.16; max 3.28 vs 3.36). There is a
  *small, genuine* L-lean that survives sample-size control — partial corr(effective_shapes, L |
  n_swings) = +0.107, and it is NOT an n_swings confound (L-flag vs n_swings corr +0.04). So
  L-stand units carry ~0.18 more effective shapes / ~0.06 more dispersion; the all-but-one-L
  top-10 is the tail amplifying that small mean shift (both leader tables actually include one R:
  Goldschmidt rank 10 widest, Carson Kelly rank 7 most-distinct). Plausibly real platoon/approach
  difference (lefties face mostly RHP) or a batter-side measurement asymmetry — flagged, not a
  blocker; revisit if it distorts cross-unit comparisons.
- **Label polish:** the `(L)/(R)` suffix on the display `label` is now applied **only to switch
  hitters** (batters with both stances in the cohort); one-way hitters keep their bare name.
  `batter_stand` is still on every row for keying — only the human-readable label changed.

## M2 fix — switch hitters: cluster per (batter, stand) (2026-07-07)

- **Bug:** clustering keyed on `batter_id` alone, pooling a switch hitter's L and R swings into
  one GMM. Only `horz_attack_angle` is handedness-mirrored (`features.py`); the other four shape
  features are not, so a switch hitter's two stances are genuinely different movements and the
  dominant axis of variation was L-vs-R stance — "cluster 0 vs 1" decoded to handedness, not
  swing shape (the Cal Raleigh L / Cal Raleigh R problem).
- **Fix:** clustering unit is now `(batter_id, batter_stand)` (`cluster.py`). One-way hitters
  unaffected (single stance); switch hitters split into two independent units. `cluster_assignments`
  / `cluster_summary` / `batter_repertoire` all gain a `batter_stand` column + a `label`
  = "Name (L/R)"; catalog and worked-example keyed on the unit. Also corrected the stale
  "per-batter constant" mirror comment in `features.py` (the sign flips between a switch
  hitter's two units — which is *why* they must be split).
- **Threshold:** lowered MIN_SWINGS from 300 pooled-per-batter to **150 per (batter, stand)** so a
  switch hitter's weaker side still qualifies (e.g. Abraham Toro R = 282). `k_max = n // 20` still
  permits up to 7 shapes at n=150. Per-side-300 would have dropped weak sides; 150 keeps them.
- **Facet 2 decision (user):** each stance is an independent "player" in cross-batter analysis —
  Raleigh L and Raleigh R get separate `batter_repertoire` rows and enter the diversity study as
  two observations. (Noted: this double-counts one human; revisit aggregation at the diversity stage.)
- **Result:** cohort 703 units (was ~389 batters), 780,328 swings, 47 switch hitters split two ways.
  Raleigh L = k5 / 2167 swings (top of "most distinct") vs Raleigh R = k2 / 892 swings — previously
  merged into one repertoire.

## M2 — Cluster viz: swing-shape cards (2026-07-07)

- Added `swing_cards(name)` to `src/cluster_results.ipynb`: a **static, Baseball-Savant-style**
  comparison matrix (matplotlib) — one row per cluster, columns = [swing-path fan | attack angle |
  attack direction | swing tilt | bat speed], each with a cohort MLB-average reference. Read down a
  column to compare a hitter's shapes. `save=` writes a PNG. Dark navy Savant theme (not
  Driveline-branded, per user).
- Geometry (`_fan`): honest reconstruction from the per-cluster means of the 5 shape features,
  since Statcast only exposes swing geometry *at* contact (research-design Limitation #1), not the
  true 3D arc. The fan's contact-point tangent points exactly along the measured
  (`vert_attack_angle` elevation, `horz_attack_angle_pull` azimuth); the arc lies in a plane tilted
  at `swing_path_tilt`; arc length ∝ `swing_length`, swept at a **fixed schematic radius R0=3.0 ft**
  (the one assumption — curvature/extent behind contact is not identifiable from the metrics). Arc
  projected into the swing plane's side view, flipped so contact is always on the right.
- **Batter skeleton omitted**: Savant's central figure uses Hawk-Eye body-pose tracking we don't
  have; only the four angle/length/speed diagrams (direct functions of our data) are drawn.
- History: first cut was an interactive 3D plotly overlay (`swing_paths_3d`, rotatable); replaced
  per user request for a static Savant-card style. `plotly` was installed into the miniforge base
  env during that pass but the notebook no longer uses it.

## M1 — Data extraction (initial)

- Added `src/extract.py`: pulls all MLB competitive swings (`is_swing=1`), 2024–2026,
  joining `pbp_raw` + `pbp_descriptions` (1:1 on `play_id`) + `players` (anthropometry).
  Keeps untracked swings too (bat-tracking cols null) so the missingness audit can compare
  tracked vs untracked; `has_bat_tracking` = `bat_speed` not null.
- Output (`data/`, gitignored): `swings_2024_2026_mlb.parquet` (1,042,291 rows / 2,562 batters
  / 109 MB), `sample_1000.csv`, `profile.md`.
- Findings:
  - 835,043 tracked swings (80.1%). Cohort: 575 batters ≥300 pooled tracked swings, 470 ≥500.
  - Tracking missingness ~roughly outcome-neutral (S 81.4% vs X 77.9%); 2026 lower (72.3%).
  - `delta_run_exp` populated on ~all tracked swings (per-swing value signal is complete).
    `woba`/`xwoba` ~62% null (BIP/PA only). Base-runner id nulls = empty base, not missing.
  - **Outliers to trim before clustering:** `bat_speed` min 1.3, `swing_length` min 0.2 (bunts/
    checked swings); `vert_attack_angle` ±90 and `horz_attack_angle` ±180 tails are artifacts
    (IQRs are sane). Intercept cols 38% null vs 20% for angles — reinforces excluding them.

### Next (M1 remainder)
- Formal missingness audit: is `has_bat_tracking` random w.r.t. pitch type / location / count /
  exit velo? (`src/` audit or notebook.)
- Feature sanity + outlier bounds; confirm sign conventions (attack angle up = +) on a
  hand-checked hitter.

## M2 — Competitive-swing filter + per-batter GMM clustering

- `src/features.py`: competitive-swing filter → `data/swings_model.parquet`.
  Operational "competitive swing" = bat-tracked + not bunt (`is_bunt`, computed in extract via
  `pitch_outcome_explanation`/`pa_outcome`) + `bat_speed >= 50`. Added a 4th filter (user-approved)
  dropping physically-impossible angle artifacts: `|horz_attack_angle| > 45` or
  `vert_attack_angle` outside [-45, 75]. Funnel: 1,042,291 → 834,993 tracked → 833,465 (−bunts)
  → 809,622 (−sub-50) → **795,723** (−13,899 angle artifacts). 1,111 batters.
  Also adds `horz_attack_angle_pull` (handedness-mirrored, + = pull).
- `src/cluster.py`: per-batter full-cov GMM, k=1..8, BIC selection among non-degenerate models
  (weight ≥0.03, ≥25 swings/comp), k capped at n//60. Cohort = ≥300 swings (**565 batters**).
  Features = 5 shape cols using `horz_attack_angle_pull`, z-scored within batter. Clusters
  relabeled by descending usage (cluster 0 = primary). Outputs: `cluster_assignments`,
  `cluster_summary` (raw-unit centroids), `batter_repertoire`, `cluster_catalog.md`.
- Results: k median 3 (mean 3.15, range 1–7), effective_shapes mean 2.83. Assignment confidence
  median resp_max 0.85 (57% >0.8). Clusters interpretable (verified on Cal Raleigh centroids).
- **CONFOUND to control before Facet 2:** `k` vs `n_swings` r=0.71 (mean k 2.25→4.20 across
  swing-count quartiles). Raw k / effective_shapes ≠ adjustability until sample size is controlled.
  Plan: re-cluster on fixed-N per hitter and/or residualize diversity metrics on n_swings for the
  diversity analysis; keep full-data clusters for the value (Facet 1) work.

### M2 refinement — BIC-driven k selection (removed arbitrary knobs)
- Replaced `K_MAX=8` + `MIN_WEIGHT` + `MIN_COMP_N` guards with pure minimum-BIC selection
  (early-stop, PATIENCE=3, N_INIT=5). Only remaining bound is the identifiability cap
  `k_max = n // PARAMS_PER_COMP` where PARAMS_PER_COMP = D + D(D+1)/2 = 20 for D=5 (a full-cov
  component needs >= its free-param count in points). Statistical necessity, not arbitrary.
- Verified equivalence: mean k 3.15→3.18, median 3, k-distribution nearly unchanged — BIC was
  already the selector; the floors barely bound. Degeneracy check: only 3/565 hitters have a
  component <15 swings, 8/565 <3% usage → BIC does not over-segment. Guards were unnecessary.
- `batter_repertoire.parquet` now also stores `bic`, `min_weight`, `min_comp_n`.

### M2 addition — shape_dispersion (within-batter shape distinctness)
- Added `shape_dispersion` to `batter_repertoire`: usage-weighted mean pairwise Mahalanobis
  distance between a hitter's cluster centroids, each pair measured against its own
  within-cluster covariance `(Σ_i+Σ_j)/2` (the Bhattacharyya-style local metric). Computed
  from the fitted GMM (`gm.means_/covariances_/weights_`) in the within-batter z-scored frame,
  so it reads as "separation in units of the batter's own swing scatter." **Within-batter scale
  — deliberately NOT cross-batter comparable yet** (per user: measure internal distinctness now,
  worry about batter-to-batter normalization later). k=1 → 0.
  - Rejected a league-covariance (cross-batter comparable) frame after discussion: the question
    is how distinct a hitter's own shapes are, so the metric frame is the hitter's own scatter.
- Results: dispersion mean 2.14, median 2.17, range ~0–2.8. Adds info beyond count-based scalars
  (Yoshida k=3/eff 2.78 but dispersion 2.65 — few but very separated shapes; Willi Castro tops
  distinctness at 2.81). Caveat: BIC only splits already-separated clusters, so multi-cluster
  hitters compress into a 2.0–2.8 band — modest dynamic range by construction.
  - Catalog gains a header line + a "Most distinct repertoires" table; existing widest/one-note
    tables unchanged except the widest table now shows `shape_dispersion`.

- **Confound check (done):** `shape_dispersion` vs n_swings Pearson +0.091 / Spearman +0.018
  (~0 excluding k=1) — effectively orthogonal to sample size, UNLIKE k/effective_shapes
  (r≈0.71). So dispersion does NOT need the n_swings residualization the count-based scalars do.
  vs k: Pearson +0.351 / Spearman +0.273 (partial dispersion~k|n_swings +0.39;
  dispersion~n_swings|k −0.29). Reason: dispersion is in within-cluster-SD units, and BIC holds
  split clusters at ~constant separation regardless of sample size — more swings buys more
  clusters, not more-separated ones. Dispersion is a genuinely distinct axis from repertoire count.

### Next
- Sample-size control for the *count-based* diversity metrics (fixed-N re-cluster or residualize
  k/effective_shapes on n_swings). shape_dispersion is exempt (confound check above).
- Cluster stability check (bootstrap/seed reproducibility of k and centroids).
- Season-drift check (are multi-season clusters real repertoire vs swing changes over time?).
