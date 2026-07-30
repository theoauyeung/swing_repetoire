# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research project on MLB swing shapes using Statcast bat tracking (2024+). Two questions:

1. Swing-shape value (Facet 1): a per-batter GMM clusters a hitter's swing shapes, and a custom xRV model grades each shape's run value conditioned on the pitch (location, type, platoon) but not the count or game state, so Swing+ measures what a swing shape is worth on a given pitch while staying count-neutral (count variation is reserved for the adjustability value analysis).
2. Repertoire diversity (Facet 2): batter-level scalars (repertoire size, usage entropy, repertoire expansiveness, context-responsiveness) test whether a wider, more adjustable repertoire improves outcomes.

`docs/research-design.md` is the source of truth for methodology, confirmed design decisions, limitations, and milestones. Read it before touching modeling code. The decisions there are settled: strictly per-batter GMM, custom xRV, and a shape defined as 5 mechanics features that exclude intercept location. `docs/worklog.md` tracks what's actually been built.

## Environment and commands

The shared `driveline` venv lives at `~/.venvs/driveline` (uv, CPython 3.13). It's reused across Driveline workflow, not a project-local `.venv`. The IDE points at it via `.vscode/settings.json` (`python.defaultInterpreterPath`), and a matching Jupyter kernel is registered as "Python (driveline)" (`--name driveline`). Select that kernel for `.ipynb` files.

```
# one-time setup (already done on this machine):
uv venv ~/.venvs/driveline --python <miniforge python> --prompt driveline
VIRTUAL_ENV=~/.venvs/driveline uv pip install pandas pyarrow scikit-learn lightgbm scipy numpy \
    mysql-connector-python requests matplotlib tabulate jinja2 ipykernel dataframe_image

# activate for a terminal session (Git Bash):
source ~/.venvs/driveline/Scripts/activate
# to add a package later:
VIRTUAL_ENV=~/.venvs/driveline uv pip install <pkg>
```

`tabulate` is required (`cluster.py`'s `.to_markdown()`); `matplotlib` for the viz notebook; `jinja2` for pandas `.style` (heatmaps in `cluster_results.ipynb`); `dataframe_image` to save table/Styler outputs as PNGs (see `results/plots/` note below).

### Pipeline

Run from repo root with the `driveline` env active:

```
python src/extract.py        # mlb_db -> swings_2024_2026_mlb.parquet + profile.md (slow: full DB pull)
python src/features.py       # competitive-swing filter -> swings_model.parquet
python src/cluster.py        # per-batter GMM -> cluster_* + batter_repertoire + cluster_catalog.md
python src/xRV_model.py      # custom per-swing xRV -> xrv_swings.parquet (+ xrv_grade)
python src/interpret.py      # Layer-1 archetype lexicon -> shape_archetypes + archetype_lexicon.md
python src/cards.py          # Layer-2 swing ID cards -> shape_cards.parquet + shape_cards_catalog.md
python src/repertoire.py     # Repertoire+ -> repertoire_scores.parquet + repertoire_catalog.md
python src/adjustability.py  # adjustability -> adjustability.parquet (headline column: adjustability)
```

Order: extract → features → cluster → xrv (built). `interpret.py` and `cards.py` are the interpretability overlay consuming cluster + xrv outputs. `shape_card(name)` in cluster_results.ipynb renders a hitter's cards. `repertoire.py` and `adjustability.py` are the two built Facet-2 stages; `value_model → within_batter → diversity → reports` remain unbuilt. Each stage is a standalone script with a `main()`; no test suite or build step yet.

### R leaderboards

`src/leaderboard_table.R` writes presentation-grade Swing+ / Repertoire+ / Adjustability leaderboard PNGs using R (gt + mlbplotR). `batter_id` is the MLBAM id `gt_fmt_mlb_headshot()` keys on. Run `Rscript src/leaderboard_table.R` from repo root; it reads `data/*.parquet` and writes `results/plots/{swingplus,repertoire,adjustability}_leaderboard_gt.png` (+ `*_bottom_gt.png`).

`swing+_results.ipynb` displays the Swing+/Repertoire+ PNGs. `adjustability_results.ipynb` opens with a 2-panel location-confound defense figure (`adjustability_location_confound.png`: (A) mean swing tilt across the zone shows location sets the swing; (B) 2-strike-minus-early pitch density shows location shifts with the count), then displays the adjustability leaderboard and the DML value finding as a 4-bar coefficient chart (`adjustability_payoff_coeffs.png`: count- vs pitch-adjustment standardized DML θ on the two-strike whiff and run-value outcomes, parsed from `results/adjustability_value.md` — count axis protective on whiff (θ=−0.365, p<0.001), run value null under DML, pitch n.s. on both).

R 4.6.0 lives at `/usr/local/bin/Rscript` (on PATH on macOS); packages arrow/dplyr/gt/gtExtras/mlbplotR/scales/webshot2 are installed. `gtsave()` PNG export needs headless Chrome (webshot2), which works in this env.

### Repertoire+

`repertoire.py` → `data/repertoire_scores.parquet`. A purely descriptive, count-aware measure of repertoire width: `expansiveness = mean_pairwise_dist × √effective_shapes`, where `mean_pairwise_dist` is the usage-weighted mean pairwise Euclidean distance between a unit's cluster centroids (each of the 5 shape features standardized by cohort (league) swing-level SD for cross-hitter comparability) and `effective_shapes = 1/Σweight²` (inverse-Simpson, the usage-effective shape count).

Design history: mean pairwise distance alone is count-blind (average dissimilarity of two random swings), so it ranked 2-extreme-shape hitters above 6-moderate-shape ones (corr with `k` ≈ −0.12). Multiplying by `effective_shapes` (eff¹) overcorrected — count then drove 84% of the ranking (corr with `k` ≈ +0.82). The `√` on the count term (eff^0.5) balances them: spread and count each contribute ~half the ranking variance (corr with `k` ≈ +0.59; corr with spread ≈ corr with eff ≈ 0.66), so a wide 2-shape repertoire can still out-rank a mediocre 5-shape one. Chosen over MST-length (pure geometry, drops usage) and Rao's Q (count reward saturates).

Geometry only: no run value, quality, or adjustability; all 5 features equal-weighted (incl. `bat_speed`); k=1 → 0 floor. Use `repertoire_pctile` as the headline, not `repertoire_plus`: 24% of units are single-shape and pile up at the 0 floor, which skews the "50 = average" reference. `repertoire_plus` is on the same scale as Swing+: `50 + 10·z` clipped to [0, 100]. Diagnostic columns `mean_pairwise_dist` and `effective_shapes` are retained in `repertoire_scores.parquet`. It reuses cluster_summary's raw centroids directly, since the horz_attack_angle pull-mirror is distance-invariant.

Pegged to a frozen 2024-25 baseline (2026-07-16): the feature SDs, the `50+10·z` mean/SD, and the percentile grid are computed once from the 2024-25 cohort and persisted to `src/repertoire_reference.json` (committed; league aggregates only, no PII), then reused on every later run so repertoire_plus/pctile stay comparable as seasons are added (like OPS+/wRC+). Delete that JSON to re-peg. Caveat: `cluster_summary` centroids are still pooled across all clustered seasons, so a true per-season cross-season plot also needs per-season centroids (unbuilt) — the peg removes scale drift, not centroid pooling.

### Adjustability

`adjustability.py` → `data/adjustability.parquet`, v4 single-regression (2026-07-22); renamed from `context_response.py`, a stale v1 label — headline column has always been `adjustability`. Measures how much a hitter's swing depends on the situation, distinct from Repertoire+ width (a hitter can be wide-but-random or narrow-but-adjustable). Measured directly on the volitional trait dials (`bat_speed`, `swing_length`, `swing_path_tilt`) — not on shape clusters (v1 measured MI over clusters, a substrate artifact) and no longer v2's single signed count slope. Per (batter, stand), 2024-25, ≥400 swings (471 units). Unsigned magnitude.

Method (v4): one joint regression per hitter per dial, `dial ~ location surface (px, pz, squares, interaction) + situation dummies`. The headline `adjustability` = the incremental adjusted R² the whole situation (count+gamestate+pitch) adds over a location-only baseline, averaged over the 3 dials, floored at 0. Each per-axis column is that axis's unique contribution — variance it adds net of location and the other two axes — so `adj_count` holds pitch type fixed. This replaced the v3 two-stage build (global location residualization, then a separate per-hitter situation R²): one model, and each hitter's own location relationship is fit rather than a single league-average surface.

Why v4 is better: v3's global surface let pitch-movement-correlated-with-location leak into `adj_pitch`; per-hitter location control removed it (adj_pitch mean 0.058→0.030), so the headline is now count-led (corr with `adj_count` 0.72 > `adj_pitch` 0.57), pointing at the axis that actually pays off. Per-axis means: count 0.021, pitch 0.030, gamestate 0.002. `adj_count` is rank-stable across v3→v4 (r=0.94); `adj_pitch` moved most (r=0.75). YoY reliability (v4, recomputed 2026-07-23): `adjustability` r=0.67, `adj_count` r=0.69, `adj_pitch` r=0.64 — all repeatable; `adj_gamestate` r=0.28 is noise (v3 read 0.75/0.72/0.78/0.19, same conclusion). The v4 recompute lives in `adjustability_results.ipynb`. v2's directional construction is in git history (988a0f1). See research-design.md Part D + docs/adjustability-decontamination.md.

### Adjustability value

`src/adjustability_value.py` → `results/adjustability_value.md`. DML causal estimates (Robinson 1988 partial linear model; XGBoost nuisance models, 5-fold cross-fitting; SE from influence-function sandwich estimator). Outcome is realized RV (`delta_run_exp`); `xrv` excluded to avoid circularity. Confounders: swing quality, log playing time, repertoire width, batter handedness, pitcher quality faced (swing-weighted mean pitcher delta_run_exp), contact rate.

Season-wide: null across all axes and both outcomes (n=450) — rules out "adjusters are just good hitters." Two-strike (n=471): `adj_count` → far fewer whiffs (θ=−0.365, p<0.001); run value null under DML once pitcher quality is controlled (θ=+0.025, p=0.52). `adj_pitch`, `adj_gamestate`, `adj_platoon` all null on both outcomes. The protective signal is whiff reduction at two strikes via count adjustment specifically.

Adjustability is a two-strike contact-preservation skill. The whiff finding is robust; run value at two strikes cannot be separated from pitcher quality in the current data.

### Archetype lexicon

`interpret.py`. Archetypes are defined on the 4 geometry features only (tilt, length, VAA, HAA_pull). `bat_speed` is a reported descriptor, not a defining axis, because its "state not trait" ICC drags a 5-feature carve into an effort bin. This is a naming overlay only; `cluster.py` still defines shapes with all 5 features. Three archetypes (`K_ARCH=3`): Level Oppo / Level Center / Uppercut Pull. MLB geometry sits on a level-oppo ↔ uppercut-pull diagonal.

`K_ARCH=3` is a deliberate interpretability choice, not the raw BIC minimum: after MERGE_SEP moved 2.0→1.75 (2026-07-13) the finer cluster pool makes BIC marginally prefer K=2 (13188.8 vs 13222.2), and at K=3 the two level components collide in the same naming cell. We keep 3 for the useful middle band and moved the `HAA_OPPO` naming boundary −5.0→−6.5 so they name apart (Level Center at haa_pull ≈−5.6 vs Level Oppo ≈−7.6). `cards.py` (Layer 2) enriches each with a `context_tag` (top-3 over-indexed situations) to produce `archetype_detailed`, so same-archetype shapes read apart. Cluster 0 (the primary swing) is labeled `"Primary"` in `archetype_detailed`, while the true archetype stays in `archetype_name`.

### Notebook plot theme

Standard analytical charts use `plt.style.context('fivethirtyeight')` with a white-background override (`figure/axes/savefig.facecolor='white'`, plus `grid.color='#cbcbcb'`, because fivethirtyeight's default grid is white and vanishes on a white bg). The `usage_heatmap` pandas Styler (cell 10) is a white-bg table with a fivethirtyeight blue→white→red diverging gradient (`FT_DIV`) to match. Only the Baseball-Savant swing cards in `cluster_results.ipynb` (cell 8, dark navy `BG/INK/MUT/GRID` palette plus hand-drawn art) stay dark by design. Both notebooks run on the `driveline` Jupyter kernel. One notebook-authoring gotcha: don't build cell source via a triple-quoted string with `\n` + `splitlines()`, because the escapes become real newlines and split string literals. Use an explicit line list instead.

### Handedness convention

Validated vs `bearing_angle`; we got this wrong once, see worklog 2026-07-09. `horz_attack_angle` is batter-relative (raw + = opposite field for both hands), so the pull frame is a uniform negation: `horz_attack_angle_pull = -horz_attack_angle` (+ = pull, both hands). There is NO per-hand mirror. `plate_x` is absolute (catcher frame), so pull-side/inside needs a real per-hand flip: `plate_x * (L? +1 : -1)`. Don't apply the same mirror to both; they differ. `vert_attack_angle`, tilt, length, and bat_speed are handedness-neutral.

DB credentials (`BIOMECH_DB_HOST/PORT/USER/PASS`) resolve from `~/.claude/.env` via the `get_secret` helper in `extract.py`. Read-only user, database `mlb_db`. Full schema: `~/.claude/skills/mlb-db-analysis/docs/schema.md`.

## Architecture and conventions

- `data/` is gitignored and holds all extracts. Never commit it; it contains athlete data. Parquet is the interchange format between stages, and markdown (`profile.md`, `cluster_catalog.md`) is the human-readable output of each stage.
- `results/plots/` holds rendered notebook outputs as PNGs (committed, unlike `data/`). Both notebooks write there via `PLOTS = ROOT / 'results' / 'plots'`: matplotlib figures via `fig.savefig`, and table/Styler outputs via `dataframe_image.export(obj, str(path), table_conversion='matplotlib')`. Use the `matplotlib` backend, not the default `chrome`/`selenium` (no browser in this env; `matplotlib` renders the gradient heatmap fine and needs no extra binary). Figures aggregate cohort-level results only, so no athlete PII gate is needed.
- Competitive swing (no DB flag exists, so `features.py` defines it): bat-tracked (5 shape features present) + not a bunt + `bat_speed >= 50` + angle artifacts dropped (`|horz_attack_angle| <= 45`, `vert_attack_angle` in [-45, 75]).
- Shape feature vector: `swing_path_tilt`, `swing_length`, `bat_speed`, `vert_attack_angle`, `horz_attack_angle`. Clustering uses `horz_attack_angle_pull` (handedness-mirrored, + = pull) so L/R hitters share a frame. Intercept location coords are deliberately excluded from shape (98–99% pitch-location artifact). `ball_bat_intercept_y` is kept only as a separate timing descriptor, never in the shape vector or as an xRV mediator.
- Clustering unit is `(batter_id, batter_stand)`, not batter alone. A switch hitter's L and R swings are different movements, so each stance clusters (and enters Facet 2) as its own "player": Cal Raleigh L vs Cal Raleigh R. Only `horz_attack_angle` is handedness-mirrored; pooling both stances would make the GMM separate on stance instead of shape. All three outputs carry `batter_stand`. `batter_repertoire` and `cluster_summary` also carry a display `label` that suffixes the stance only for switch hitters ("Cal Raleigh (L)"), leaving one-way hitters bare ("Aaron Judge").
- Clusters are strictly per-unit and not comparable across units. Cluster 0 = that unit's primary (highest-usage) swing. All cross-unit analysis must use unit-level scalars (`batter_repertoire.parquet`), never shared cluster IDs.
- GMM k selection is minimum-BIC (early-stop, no occupancy floors) followed by a post-BIC merge. BIC over-segments at large n into large-but-near-duplicate components, so `cluster.py` merges component pairs closer than `MERGE_SEP=1.75` (within-cluster-SD Mahalanobis) into one shape. Reported `k` is post-merge. This is not an occupancy floor: the phantom components are large (~28% usage), so the problem is separation, not size. Identifiability cap is `k_max = n // 20`. Cohort is ≥150 competitive swings per `(batter, stand)` unit (lowered from pooled-300 to keep switch hitters' weaker side). Post-merge: mean k ≈2.26, median 2, max 6 (MERGE_SEP=1.75; was ≈1.9 at 2.0). 13% of units are single-shape.
- Count-based diversity metrics (`k`, `effective_shapes`) correlate with `n_swings` (r≈0.71) and must be sample-size-controlled before Facet 2.

## xRV status

`src/xRV_model.py` is built and committed. Predictors: 5 shape features + `same_hand`, `plate_x_pull`, `plate_z_norm`, `pitch_type` (no count, no game state). Three XGBoost models (`p_bip`, `p_foul`, `v_bip`); hyperparameters fixed from `experiments/sweep.py` (2024 train / 2025 val); trained on 2024-25, held-out test on 2026. Outputs `data/xrv_swings.parquet` with `xrv` and `xrv_grade` (0-100, 50 = league average). Validation: realized_rv vs `delta_run_exp` corr=0.957. `data/re24.csv`, `count_values.csv`, `count_transitions.csv`, `linear_weights.csv` are the run-value tables it reads. Aggregate hitter xRV vs `pitch_values.ipv` calibration is still TODO.

## Documentation protocol

When you change behavior, update `docs/worklog.md` (append what you built + findings), and `docs/research-design.md` if a methodology decision changes. Keep this file's conventions current if you discover new project-specific patterns.
