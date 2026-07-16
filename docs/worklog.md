# Worklog

## Table outputs saved as PNGs via dataframe_image (2026-07-13)

- **User: wire up dataframe_image to save the tables as images.** Installed `dataframe_image` into
  the `driveline` venv and re-added saving to both notebooks, now as PNGs:
  `cluster_results.ipynb` cell 10 exports the usage-heatmap Styler, and `swing+_results.ipynb`
  exports all four leaderboards (restored the `PLOTS` def + a `dfi` import in its setup cell).
- **Backend = `matplotlib`, not chrome/selenium.** `dfi.export(obj, str(path),
  table_conversion='matplotlib')` needs no browser binary (the default `chrome` backend would fail
  headless here) and renders the heatmap's `background_gradient` correctly. Verified all 5 table PNGs
  plus the existing figure PNGs. Note `dfi.export` wants a `str` path, not a `Path`.
- `results/plots/` is now 9 PNGs (4 figures + 5 tables). Docs updated (env dep list, results/plots
  convention, README).

## Remove HTML table saves from both notebooks (2026-07-13)

- **User: drop the HTML saves, they'll export tables as images later.** Removed the `.to_html()`
  saving code from `cluster_results.ipynb` (cell-10 usage heatmap) and all four `swing+_results.ipynb`
  leaderboard cells, and dropped the now-unused `PLOTS` def from the swing+ setup cell. `git rm`'d the
  5 HTML files (`usage_heatmap_arraez`, `swingplus_leaderboard`, `swingplus_by_cluster`,
  `swingplus_clusters_raleigh`, `repertoire_leaderboard`). PNG saves in cluster_results (swing cards,
  archetype scatter, repertoire-size dist) are untouched. `results/plots/` is now PNG-only: don't
  re-add `.to_html()` dumps; table→image export is the pending approach.
- (User-side, via auto-commits: added `swing_cards_baldwin.png`, then deleted `swing_cards_ohtani.png`
  and `mergesep_sensitivity.png` from the repo.)

## Keep 3 archetypes (retune HAA_OPPO) + swing+ notebook saves tables (2026-07-13)

- **User: keep three archetypes.** Set `K_ARCH` back to 3 (a deliberate interpretability override of
  the raw BIC min, which is 2 on the MERGE_SEP=1.75 pool — a slim 13188.8 vs 13222.2 margin). At K=3
  the two level components collided in the "Level Oppo" naming cell (they split on horz attack:
  haa_pull ≈−5.6 vs ≈−7.6), so retuned the naming boundary `HAA_OPPO` −5.0→−6.5 to name them apart.
  Restores **Level Oppo (573) / Level Center (446) / Uppercut Pull (573)**. Reran interpret → cards;
  regenerated cluster_results plots.
- **User: save plots in swing+_results.ipynb too.** That notebook has no matplotlib figures — it's
  four leaderboard tables (Swing+ top 25, Swing+ by cluster, a per-hitter helper demo, Repertoire+).
  Added a `PLOTS = ROOT / 'results' / 'plots'` and each cell now writes its table via `.to_html()`
  (same HTML approach as the cluster_results heatmap, since there's no image backend):
  `swingplus_leaderboard.html`, `swingplus_by_cluster.html`, `swingplus_clusters_raleigh.html`,
  `repertoire_leaderboard.html`.

## MERGE_SEP 2.0 -> 1.75 + archetype K 3 -> 2 (2026-07-13)

- **User asked whether MERGE_SEP=2.0 was too aggressive** (mean 1.94 shapes felt low). Ran a
  threshold sweep: BIC-select each of the 703 units once, then apply the merge at 1.0..3.5 and also
  dump the pre-merge pairwise-separation distribution (2,194 component pairs). Saved
  `results/plots/mergesep_sensitivity.png`.
- **Finding: no natural gap.** The BIC-component separations are a gapless continuum peaking right
  at ~2.0 (median 2.15), so MERGE_SEP is a judgment dial, not a data-pinned value (same continuum
  reason ICL was rejected). 2.0 sat on the steepest part of the mean-k curve. The true near-duplicate
  mass (BIC phantom splits) is only the <1.5 band (~10% of pairs); everything >=1.5 is at least
  modestly separated. Sweep: T=1.5 -> mean 2.50 / 11% k=1; T=1.75 -> 2.26 / 13%; T=2.0 -> 1.94 / 24%;
  T=2.25 -> 1.55 / 49%; T=2.5 -> 1.20 / 81%.
- **Decision (user): loosen to 1.75.** Keeps every pair below 1.75 SD merged (~19% overlap) but
  recovers 1.75-2.0 shapes. New cluster.py output: mean k 2.26, median 2, max 6, k-dist
  91/408/147/42/14/1, effective_shapes 2.10, single-shape units 24%->13%. Reran cluster -> interpret
  -> cards -> repertoire; xrv untouched (per-swing, cluster-independent).
- **Cascade into Layer 1 (interpret.py): archetype count 3 -> 2.** The finer cluster pool (1,592
  centroids) moved the BIC minimum for K_ARCH to 2 (BIC 13188.8 at K=2 vs 13222.2 at K=3), and every
  K>=3 now also trips the unique-name guard (two components land in the same "Level Oppo" naming
  cell, differing only in tilt). So the lexicon is now **Level Oppo (1006) / Uppercut Pull (586)**;
  the old middle "Level Center" no longer earns its own component. Set K_ARCH=2 per the documented
  BIC-min rule. cards.py `archetype_detailed` still works (Arraez now k=6).
- Regenerated all notebook plots against the new data. Updated CLAUDE.md + research-design.md
  (threshold, mean k, single-shape %, archetype count/names).

## cluster_results.ipynb: save plots to results/plots/ (2026-07-13)

- **Added plot-saving to the notebook (user ask).** Cell 1 now defines `PLOTS = ROOT / 'results'
  / 'plots'` and `mkdir(parents=True, exist_ok=True)`. Each visual cell writes its output there:
  cell 8 `swing_cards('Arraez', save=PLOTS / 'swing_cards_arraez.png')` (the function already had a
  `save=` param), cell 13 `archetype_batspeed_vs_vaa.png`, cell 15 `repertoire_size_distribution.png`
  (both `fig.savefig(..., dpi=130, bbox_inches='tight')` inside the fivethirtyeight-white context so
  the white facecolor carries into the file).
- **Heatmap is a pandas Styler, not a matplotlib fig**, and `dataframe_image` is not installed, so
  cell 10 saves `usage_heatmap_arraez.html` via `_hm.to_html()` (jinja2 already present) instead of a
  PNG. The other three are PNGs. If we want image parity, add `dataframe_image` to the venv and swap
  to `dfi.export`.
- **`results/` is not gitignored**, so the folder is committable. Generated all four files headlessly
  by exec-ing the notebook's own cell sources (setup 1/2/3/12 then plot cells 8/10/13/15) under the
  Agg backend, so the notebook stays the single source of the plotting code.

- **Added a distribution plot (user ask):** a 2-panel figure of how many distinct swing shapes each
  `(batter, stand)` unit carries — left = repertoire size `k` (bar, 168/24% k=1, 428/61% k=2,
  91/13% k=3, 13/2% k=4, 3/0% k=5), right = `effective_shapes` = `exp(usage entropy)` histogram
  (spike at 1.0 for one-shape units, band near 2, mean 1.81). Both read from `rep`
  (batter_repertoire), appended as a new markdown + code cell at the end of the notebook.
- **Theme change (user): standard analytical charts → fivethirtyeight with a WHITE background.**
  Converted the archetype scatter (bat_speed vs VAA) and built the new distribution plot in
  `plt.style.context('fivethirtyeight')` with an rcParams override to white
  (`figure/axes/savefig.facecolor='white'`, `grid.color='#cbcbcb'` so the grid shows on white —
  fivethirtyeight's default grid is white, invisible on a white bg; that recolor is the key step).
  **Left the bespoke Baseball-Savant swing cards (cell 8) in their dark theme** per user scope
  choice — fivethirtyeight doesn't compose with the hand-drawn card art. Kept the archetype palette
  (`ARCH_PAL`) but bumped point alpha to 0.7 so the tan Level-Center points read on white.
- **Gotcha (notebook authoring):** writing cell source via a triple-quoted Python string containing
  `\n` + `.splitlines(keepends=True)` turned the intended in-string `\n` escapes into real newlines,
  splitting string literals (SyntaxError). Fixed by building source as an explicit line list and
  inserting real backslash-n via `chr(92)+'n'` for the multi-line titles/labels.
- Set the notebook kernelspec to `driveline` (was generic `python3`). Re-executed headlessly
  (`nbconvert --execute --kernel driveline`, exit 0); both new/updated plots render with white bg.
- **Follow-up (user): rethemed the `usage_heatmap` pandas Styler (cell 10) too.** Swapped its custom
  Savant blue→off-white→red colormap for a fivethirtyeight-consistent `FT_DIV` blue (`#30a2da`) →
  white → red (`#fc4f30`) diverging gradient, and added explicit `set_table_styles` (white bg, dark
  `#3c3c3c` text, light `#cbcbcb`/`#ececec` borders) so the table matches the plots. Re-executed clean.
  Now only the bespoke Savant swing cards (cell 8) remain dark by design.

## swing+_results.ipynb — fixed Repertoire+ leaderboard + merged Swing+ (2026-07-09)

- **Fixed the "Swing Repertoire" cell (cell 7).** It was `print()`ing the DataFrame (plain text, no
  rich table) and its `.round({'repertoire_plus': 0})` targeted a column already renamed to
  `Repertoire+`, so it silently didn't round. Now returns the DataFrame as the cell value (rich HTML
  table) with correct per-column rounding, sorted by Repertoire+ desc, leading with the percentile.
- **Merged Swing+ into the leaderboard (user ask).** Added a unit-level **Swing+** column = mean
  `xrv_grade` over each `(batter_id, batter_stand)` unit's 2024-25 swings (join `cluster_assignments`
  play_ids → `xrv_swings`), merged onto `repertoire_scores` by (batter_id, batter_stand). Columns:
  Batter / R/L / Shapes (k) / Repertoire pctile / Repertoire+ / Swing+. Confirms the two axes are
  independent — the widest repertoires (Repertoire+ ~111-114) carry ~league-average Swing+ (~48-50),
  i.e. profile *width* is orthogonal to swing *quality*.
- **Set the notebook kernelspec to `driveline`** (was the generic `python3` / "defaultInterpreterPath"
  display). Re-executed the whole notebook headlessly (`nbconvert --execute --kernel driveline`,
  exit 0); all cells clean, 703 units in the leaderboard.

## Renamed Arsenal+ → Repertoire+ (2026-07-09)

- **Per user: renamed the expansiveness metric and its stage from "Arsenal" to "Repertoire"**
  everywhere in scripts and current-state docs. `src/arsenal.py` → `src/repertoire.py`; columns
  `arsenal_plus`/`arsenal_pctile` → `repertoire_plus`/`repertoire_pctile`; outputs
  `arsenal_scores.parquet` + `arsenal_catalog.md` → `repertoire_scores.parquet` +
  `repertoire_catalog.md` (old files deleted, new ones regenerated — cohort/values identical, pure
  rename). Docs updated: `research-design.md` Part D, `CLAUDE.md` (pipeline cmd + Repertoire+
  section), `README.md`, and the prior worklog entries' references to our metric.
- **Kept unchanged (proper nouns, NOT our metric):** the pitcher "paint mixer" **arsenal score**
  concept and the source repo `drivelineresearch/arsenal_construction` — those are real external
  names, not the swing metric.

## Removed the `shape_dispersion` metric (2026-07-09)

- **Per user: dropped `shape_dispersion` entirely** from scripts and current-state descriptions.
  It is superseded by Repertoire+ (the cross-hitter-comparable, league-frame expansiveness metric);
  `shape_dispersion`'s within-batter Mahalanobis frame was deliberately not rankable, so it had no
  consumer once Repertoire+ existed. No downstream code read the column (verified: only `cluster.py`
  produced it; the notebook reference was stale render output).
- **`cluster.py`:** removed the `shape_dispersion()` function, its call in `fit_batter`, the
  `batter_repertoire` column, and the two catalog sections that used it ("Shape dispersion" header
  line, the column in the widest-repertoires table, and the "Most distinct repertoires" table).
  **Preserved `_pair_maha` + the `merge_components` post-BIC merge loop** — those use Mahalanobis
  distance internally for the MERGE_SEP over-split fix and are unrelated to the reported metric.
  (A concurrent botched edit had deleted `_pair_maha` and the merge `while` loop while leaving the
  callers, which would have silently disabled merging and ballooned k back to the BIC over-split
  values; caught and restored.) `merge_components` now returns `labels, resp_max, weights` (dropped
  the post-merge `means`/`covs` that only fed `shape_dispersion`).
- Re-ran `cluster.py`: cohort + k distribution **unchanged** (703 units; k mean 1.94, median 2,
  max 5; dist 168/428/91/13/3) — confirms the merge logic still behaves identically. Regenerated
  `batter_repertoire.parquet` (no `shape_dispersion` col) + `cluster_catalog.md`. Re-ran
  `repertoire.py` (unaffected — reads cluster_summary). Cleared the stale `rep.head()` output cell in
  `cluster_results.ipynb`.
- Docs scrubbed: `research-design.md` (Part D bullet + Limitation #10 dispersion mention),
  `CLAUDE.md` (Facet-2 scalar list, Repertoire+ contrast, confound note), `README.md`. Historical
  worklog entries left intact (append-only audit log — they were accurate when written).

## Facet 2 — Repertoire+ (repertoire expansiveness), `src/repertoire.py` (2026-07-09)

- **New Facet-2 stage (user brainstorm, modeled loosely on the pitcher "paint mixer" arsenal score
  in `drivelineresearch/arsenal_construction`).** Walked the user through the paint-mixer method
  (portfolio-theory pitch-mix optimizer: run-value returns + overuse decay curves + pairwise
  "buyback" interaction + entropy penalty + SLSQP usage optimizer) and the mapping to swings. The
  central mismatch: **pitchers freely choose usage; a hitter's swing shape is largely reactive** to
  the pitch (design Limitation #2), so the prescriptive optimizer/decay/buyback machinery does not
  port. After iterating, the user narrowed the metric all the way down to **purely descriptive
  geometric spread** — nothing about run value, shape quality, or deployment. Wider profile (bigger
  gaps between clusters in bat speed / angles) = bigger repertoire.
- **`repertoire.py`** (consumes cluster_summary + swings_model): per (batter, stand), expansiveness =
  **usage-weighted mean pairwise Euclidean distance between cluster centroids**, each of the 5 shape
  features standardized by the **cohort swing-level SD** (makes mph/deg commensurable AND rankable
  across hitters). `repertoire_plus = 100 + 10·z`; also `repertoire_pctile` + per-feature raw-unit spread
  breakdown. Outputs `repertoire_scores.parquet` (703 units) + `repertoire_catalog.md`.
- **Key distinction from `shape_dispersion`:** that metric is scaled in each hitter's *own*
  within-cluster scatter (Mahalanobis), deliberately NOT cross-hitter comparable (worklog 2026-07-07).
  Repertoire+ uses the **league frame** so it's rankable — the whole point of a score. Reuses
  cluster_summary's raw centroids directly: the `horz_attack_angle` pull-mirror is a uniform negation
  and pairwise distances use differences, so the mirror is distance-invariant (verified reasoning).
- **Design decisions (all user-confirmed via option walkthrough):** geometry only (no value); all 5
  features equal-weighted incl. bat_speed (archetype lexicon excludes bat_speed, but it's wanted
  here); usage-weighted (π_i·π_j), not capability; mean pairwise, not max; keep k=1 units in the
  distribution but **lead with `repertoire_pctile`** since 168 single-shape units (24%) pile up at the
  0-spread floor and drag the Repertoire+ mean below the multi-shape median (mean exp 1.80 vs median
  2.26 → Repertoire+ 100 sits below the k≥2 median of 105.4). Repertoire+ is a monotone transform of the
  same ranking, so pctile and + agree on order.
- **Findings:** expansiveness mean 1.80 / median 2.26 / max 3.28. k dist 168/428/91/13/3 (k=1..5).
  Per-feature contribution in league-SD units (multi-shape): swing_length 1.34, horz_attack_angle
  1.30, vert_attack_angle 1.22, bat_speed 0.41, swing_path_tilt 0.35. **Caveat flagged:** width is
  driven by length + attack angles; the biggest angle contributor (`horz_attack_angle`) is also the
  most pitch-reactive feature (ICC 0.054), so a horz-driven wide repertoire partly reflects
  pitch-location variety, not genuine swing change. Widest: Johnathan Rodríguez, Michael Toglia (L),
  Spencer Jones, Bo Bichette (all big horz + VAA gaps). Narrowest multi-shape: Willie Calhoun,
  Ryan Kreidler, Royce Lewis.

## swing+_results.ipynb — archetype column on the cluster leaderboards (2026-07-09)

- Added an **Archetype** column (Layer-1 label from `shape_archetypes.parquet`) to the per-shape
  Swing+ leaderboard and the `cluster_table(name)` helper, merged on
  (batter_id, batter_stand, cluster). A bare "cluster 2" now reads as e.g. "Uppercut Pull". Markdown
  note updated. Re-executed clean on the post-merge clusters (numbers shifted from the old committed
  outputs because cluster.py was re-run with the over-split merge). Illustrative: Cal Raleigh (L)'s
  two Uppercut Pull shapes split 55.2 (best) vs 40.3 (worst) Swing+ — same archetype, A-swing vs
  fooled version.

## cards.py — cluster 0 archetype label = "Primary" (2026-07-09)

- Per user: `archetype_detailed` is now `"Primary"` for cluster 0 (the highest-usage swing) instead
  of archetype+situation, so the swing+ leaderboard / `cluster_table` read "Primary" on the primary
  row. `archetype_name` still holds the true archetype (analysis unaffected). The card renderers
  (`shape_cards_catalog.md`, notebook `shape_card`) show the primary's real archetype_name in the
  header since the role star already says PRIMARY (avoids a redundant "PRIMARY — Primary").
  Re-ran cards.py + both notebooks (clean); all 1,364 cluster-0 rows = "Primary".

## Handedness fix + archetype situational enrichment (2026-07-09)

- **BUG FOUND (user flagged pull/oppo correctness): the pull frame was wrong.** Validated every
  metric against `bearing_angle` (established absolute convention via pull%>oppo%: RHH 59% to LF /
  bearing<0, LHH 60% to RF / bearing>0; Paredes dead-pull RHH confirms). Two distinct problems:
  - **`horz_attack_angle` is BATTER-RELATIVE** (raw + = toward opposite field for both hands;
    corr(raw, pull) = -0.47 RHH / -0.45 LHH). So the pull frame is a **uniform negation**
    (`-horz_attack_angle`), NOT a per-hand mirror. The old `*(L?-1:1)` left **RHH inverted** —
    RHH "pull" was actually oppo; LHH was accidentally correct. (Visible symptom: Aaron Judge, a
    pull hitter, was mislabeled "Uppercut Oppo" → now correctly "Uppercut Pull".)
  - **`plate_x` is ABSOLUTE** (catcher frame; corr(raw, pull) = -0.15 RHH / +0.16 LHH) → needs a
    REAL per-hand flip (flip RHH: `*(L?+1:-1)`). The old shared `side` mirror had it wrong for both.
  - Fixes: `features.py` (horz → uniform negation), `xRV_model.build_features` + `cards.py`
    (plate_x → flip RHH). **No xRV retrain needed** — feature sign is tree-invariant, so xrv_swings
    is unchanged. Re-validated post-fix: corr(horz_attack_angle_pull, pull) = +0.47/+0.45 both hands.
- **Lexicon → K_ARCH=3** (was 4). Correcting the pull frame collapsed the geometry onto a
  **level-oppo ↔ uppercut-pull diagonal** (uppercut swings are pull-side), so BIC now prefers 3:
  **Level Oppo / Level Center / Uppercut Pull** (578/478/308; HAA thresholds ±5). k≥4 splits the
  level band into same-named near-duplicates. Assignment confidence median 0.85 (lower — the
  corrected geometry is more of a continuum). The fail-loud name-collision assert caught the k=4
  duplicate ("two Level Oppo") before it shipped.
- **Archetype situational enrichment (user ask).** `cards.py` adds `context_tag` (top-3 over-indexed
  buckets) and `archetype_detailed` = "archetype · situation" so same-archetype shapes separate:
  e.g. Ohtani's two Uppercut Pulls → "· down, vs offspd" vs "· away, vs offspd"; Raleigh L's two →
  "· down, vs offspd, 3-2" (grade 55) vs "· down, vs offspd, inside" (grade 40, -11.6 runs/100).
  Shapes deployed identically still tie on the tag — then geometry (name_delta) + value (grade)
  separate them. swing+_results.ipynb leaderboard + cluster_table now show `archetype_detailed`;
  cluster_results.ipynb palette/map/scatter/markdown updated to the 3 archetypes + corrected frame.
- Re-ran features → cluster → interpret → cards + both notebooks (all clean).

## Interpretability Layer 2 — per-hitter swing ID cards (2026-07-09)

- **Added `src/cards.py`** (consumes cluster_summary + shape_archetypes + cluster_assignments +
  xrv_swings + swings_model). One row per (batter, stand, cluster) → `data/shape_cards.parquet`
  (1,364 rows / 703 units) + `shape_cards_catalog.md`. Notebook renderer `shape_card(name)` added to
  cluster_results.ipynb (renders each shape as a rich-Markdown card). Fields per shape:
  - **name_delta** — geometry phrase; primary vs the league (cohort) mean, secondaries vs THIS
    hitter's primary. Top features by |delta| in centroid-SD units (always emits ≥1 so it's never
    blank). This is where same-archetype shapes separate — e.g. Ohtani's two "Uppercut Pull"
    shapes read "+17deg uppercut/+26deg pull" (78 mph A-swing) vs "−8mph slower/+20deg pull" (67 mph
    defensive version).
  - **when_label** — contexts where the shape is over-indexed vs the hitter's own baseline
    (lift = P(shape|ctx)/P(shape) − 1 ≥ +15%, ≥25 unit swings & ≥5 shape swings in the bucket),
    across count / location(up/down/inside/away, pull-framed) / pitch(FB vs offspeed) / base-out.
    Base-out is deployment-only.
  - **grade** — mean xrv_grade over the shape's swings (Part C.1 conditional grade).
  - **matched_runs100** — within-batter matched contrast vs the primary (Part C.2): mean xRV(shape) −
    xRV(primary) over shared count3×height3×pitch2 strata (≥5 each), usage-weighted, ×100. NaN for
    the primary. Base-out deliberately excluded (xRV excludes game state). + matched_n + matched_thin.
- **Design decisions realized (from the brainstorm):** naming = delta-vs-own-primary; when = over-
  index/lift; value = context-conditioned grade + matched contrast vs primary (cluster 0); context
  dims count/location/pitch/base-out with base-out deployment-only. Matched strata kept coarse
  (3×3×2) so two shapes share support; the when-label uses richer buckets (only needs marginal
  support). Location pull-framed to match features.py (`plate_x_pull`, mirror + = pull).
- **Sanity:** grade mean 49.3 / sd 2.5 (compressed → the matched contrast, range −12.1..+7.0
  runs/100, carries the value discrimination). Only 6% of secondaries beat their primary in matched
  spots (expected — cluster 0 is the highest-usage, usually-best swing); those 40 are the
  situational-upgrade cases. 6% of contrasts flagged thin. Ohtani's 67 mph defensive Uppercut Pull
  (used inside +168% / offspeed +78% / down +73%) grades 45 / −10.34 runs/100 — his worst shape,
  exactly the fooled-emergency swing. Notebook re-executed clean.
- **Next / open:** Layer 2 is the first cut of Facet-1 (Part C.1/C.2). A branded report export and
  the Facet-2 diversity/adjustability stage remain. Grade compression suggests a future within-hitter
  or percentile grade may read better than the league T-score for the card headline.

## M2 fix — post-BIC component merge to kill over-splitting (2026-07-09)

- **Problem (found via Layer-1 eyeballing):** pure minimum-BIC over-segments at large n. Diagnostic
  (220-unit sample): mean closest-pair component separation falls 2.21→1.45 across n quartiles while
  mean k rises 2.15→4.12 — BIC's ln(n) penalty is too weak to stop it splitting trivial density bumps
  into large-but-near-duplicate components (Ohtani's twin Level Centers at 75 vs 78 mph, both ~28%
  usage). 29% of multi-cluster units had a pair <1.5 Mahalanobis, 69% <2.0.
- **Ruled out two fixes:** (1) occupancy floor — the phantoms are *large* (27-28%), not tiny, so a
  size floor can't catch them; the problem is separation. (2) Switching BIC→ICL — overcorrects
  catastrophically: k_icl collapsed to ~1 for 198/222 units (swing shapes are a continuum, so ICL's
  entropy penalty refuses to split at all). Neither selection criterion works alone.
- **Fix (user chose threshold 2.0):** keep BIC selection, then `merge_components()` collapses
  component pairs closer than `MERGE_SEP=2.0` (within-cluster-SD Bhattacharyya-Mahalanobis, same
  metric as `shape_dispersion` — factored a shared `_pair_maha` helper) into one shape, closest pair
  first, iteratively. Un-merged singletons keep the GMM's PD params; merged groups get empirical
  mean/cov/weight from pooled swings; merged responsibility = sum of member comps'. Reported k is
  post-merge; `bic` is still the selected model's. `cluster.py` docstring + constants updated.
- **Result:** cohort unchanged (703 units). k: mean 3.15→**1.94**, median 3→2, max 7→5. k dist:
  168 k=1 / 428 k=2 / 91 k=3 / 13 k=4 / 3 k=5. Verified merges: Ohtani 5→3 (twins merged, the two
  genuinely-different Uppercuts kept), Arraez 7→5, Carson Kelly 6→3, Merrill 4→2. shape_dispersion
  mean 2.14→2.06 (now on empirical merged covs, not fitted — slight frame change).
- **Two consequences, both resolved (user decisions):**
  1. **Facet-2 impact — keep 2.0.** Mean k halved, 24% of units now k=1. User accepted: purity for
     Facet-1 shape value takes priority; Facet 2 works with the compressed spread. (Revisit at the
     diversity stage if the signal is too thin — dialing to ~1.8 would recover shapes without
     un-merging the flagged twins.)
  2. **Lexicon → K_ARCH=4** (BIC min on the merged pool; k=5 forced a degenerate 25-shape bin).
     Retuned the naming thresholds since the merged pool is center-heavy: VAA_FLAT 6→3 (~6° reads
     "Level", not "Flat"), HAA_OPPO/PULL ±6→±4 (so the two center archetypes split by direction
     instead of colliding on "Center"). Clean 2x2 vocabulary: **Level Oppo / Level Pull / Uppercut
     Oppo / Uppercut Pull** (528/363/254/219 shapes, confidence median 0.96). interpret.py +
     cluster_results.ipynb (ARCH_PAL/ARCH_ORDER + the 3 lexicon visuals) updated to 4 archetypes;
     notebook re-executed clean on the merged data.

## Interpretability Layer 1 — league swing-shape archetype lexicon (2026-07-09)

- **Goal (user brainstorm):** clusters are bare indices ("Ohtani Cluster 1") and require
  cross-referencing three disconnected views (swing_cards mechanics, usage_heatmap deployment,
  xrv_swings value) to interpret. Agreed a two-layer fix: **Layer 1** = a cross-unit archetype
  vocabulary so every cluster inherits a human name; **Layer 2** (next) = per-hitter "swing ID
  cards" that name each shape as a delta-vs-primary, summarize *when* it's used (over-index/lift
  vs the hitter's own baseline), and grade value context-conditioned + within-batter matched vs
  the primary (Part C.2). This entry covers Layer 1 only.
- **Added `src/interpret.py`.** Recomputes each of the 1,954 unit-cluster centroids in the
  **pull frame** (handedness-mirrored horz_attack_angle, from cluster_assignments + swings_model —
  cluster_summary stores the raw unmirrored angle which would split L/R pull swings), league-
  standardizes the centroid pool (unweighted: one obs per shape), fits a full-cov GMM, and tags
  every unit-cluster with an archetype + assignment confidence. Algorithmic naming from the
  centroid — vertical (Flat <6° VAA <13° Level < Uppercut) x direction (Oppo <−6° HAA_pull <6°
  Center < Pull) — reproducible regardless of GMM component order; asserts names come out unique
  (fail-loud). Outputs (gitignored): `shape_archetypes.parquet` (per unit-cluster tag),
  `archetype_lexicon.parquet` (5-row lexicon), `archetype_lexicon.md`.
- **Decision — archetypes defined on 4 GEOMETRY features; bat_speed is a descriptor, not a
  defining axis.** Including bat_speed (all 5 shape features) let the min-BIC 5-way carve key on
  effort: it split off a degenerate 134-shape low-bat-speed "weak contact" bin (Arraez/Kwan/soft-
  Ohtani) and dissolved the geometry grid. Motivated by research-design's own finding (bat_speed
  ICC 0.126, "state not trait"); mirrors how intercept_y is quarantined as a descriptor. Geometry-
  only recovers a balanced, stable grid (Flat Oppo 518 / Flat Pull 463 / Uppercut Oppo 349 /
  Level Center 343 / Uppercut Pull 281) and reports bat_speed as a per-archetype correlate
  (flat ~68.7, lifted ~71.3). The per-batter clustering in cluster.py is UNCHANGED (still 5 feats);
  this is a naming overlay only. (User confirmed geometry-only.)
- **k=5 chosen** (k=4 is marginally lower BIC but merges two grid cells; k≥6 splits degenerate
  near-duplicates). Exemplars pass scouting priors: Schwarber/Pasquantino→Uppercut Pull,
  Soto/Kwan→Flat Oppo, Semien/Arenado→Flat Pull, Altuve/Julio→Uppercut Oppo,
  Swanson/Judge→Level Center.
- **Stability:** N_INIT=8 sometimes settled in a worse local optimum → **N_INIT=25** reliably hits
  the global one. The 5 archetype *names* are seed-invariant; only a few boundary shapes move
  (Hungarian-matched seed agreement ~0.93). Assignment confidence median 0.94, 62% >0.8 (lower
  on genuine boundary shapes, e.g. Judge's primary 0.69 — stored so Layer 2 can flag borderline).
- **Known, expected wrinkle:** ~21% of unit-archetype pairs (328/1564) carry >1 cluster of the
  same archetype — effort/fine-geometry variants (e.g. Ohtani's two Uppercut Pull at 78 vs 67 mph).
  The archetype tag intentionally groups them; the Layer-2 delta-vs-primary text carries the
  distinguishing bat_speed/geometry gap. Nothing lost, just moved to the right layer.

## Interpretability Layer 1 — lexicon eyeball visuals in cluster_results.ipynb (2026-07-09)

- Appended an "Archetype lexicon (Layer 1)" section to `src/cluster_results.ipynb` (Savant dark-navy
  theme, matching swing_cards/usage_heatmap). Loads `shape_archetypes` + `archetype_lexicon`; adds
  `ARCH_PAL`/`ARCH_ORDER` (palette ordered by lift: cool=flat, warm=uppercut). Three visuals:
  1. **bat_speed vs vert_attack_angle** scatter over all 1,954 shape centroids, colored by archetype,
     with OLS fit + Pearson **r = 0.36** (+~0.2 mph/degree). Confirms effort rises with uppercut, i.e.
     bat_speed is partly redundant with lift — the empirical basis for making it a descriptor, not a
     defining axis.
  2. **`archetype_map(name)`** — a hitter's shapes on the league direction×vertical grid (bubble = usage
     weight, annotation = cluster id · bat speed, dotted naming-threshold lines). The tool for the
     distinguishability question: are a hitter's shapes spread out or piled in one archetype cell?
  3. **collision curve** — mean DISTINCT archetypes vs repertoire size k, vs the y=x "if every shape were
     unique" line, annotated with % of units at each k that repeat a label.
- **Distinguishability finding (answers the k=7 worry):** archetype labels saturate at ~2.8-3.0 distinct
  no matter how high k climbs (k=3 → 52% of units repeat a label; k=4 → 91%; k>=5 → 100%; 40% overall).
  A 5-word vocabulary provably can't uniquely name 7 shapes — expected, and the reason Layer 2
  (delta-vs-primary + bat_speed) is required. BUT most same-label shapes still separate on position/effort
  (e.g. Arraez's 3 Flat Oppos fan out along VAA/direction).
- **Side finding — probable GMM over-split (NOT a lexicon issue):** a few hitters carry two clusters that
  overlap on *both* geometry and bat_speed (Ohtani c0/c1 Level Center 75 vs 78 mph; Arraez c1/c6 Level
  Center). Layer 2 would name these near-identically → they may be phantom shapes the per-batter GMM
  should have merged. Ties to the open "cluster stability check" TODO in research-design.md; worth
  resolving before Layer 2 so we don't name non-shapes. Verified the whole notebook executes clean
  (nbconvert --execute, exit 0; all three new cells emit figures).

## M3 — xRV 0-100 grade + feature-set regression note (2026-07-08)

- Added `xrv_grade` to `data/xrv_swings.parquet`: z-score the expected `xrv` across all swings, map to
  50 + 10*z, clip [0,100] (T-score style; 50 = league-avg swing). Computed in `main()` before the
  parquet save. Verified: mean 50.00, sd 9.99, ~0.06% clipped low / 0.003% high.
- **Feature-set change (user): dropped plate location (`plate_x_pull`, `plate_z_norm`) and replaced
  `stand_L`+`p_throws_L` with a single `same_hand` platoon flag.** FEATURES now = balls, strikes,
  same_hand, pitch_type + 5 shape. Fixed a build_features bug (same_hand was computed to a local, not
  assigned). **Regression:** 2026 held-out p_bip AUC 0.788→0.732, p_foul 0.828→0.760, v_bip r2
  0.050→0.037 — plate location is a top contact/BIP driver; removing it costs ~0.06-0.07 AUC. PARAMS
  are also still tuned on the old feature set. Validation gate unchanged (0.957; tables-only).
  Recommendation pending user: add plate location back, then re-run the sweep to re-tune PARAMS.
- **Decision (user): plate location added back.** FEATURES now = balls, strikes, same_hand,
  plate_x_pull, plate_z_norm, pitch_type + 5 shape (11 total; kept `same_hand` over stand_L/p_throws_L).
  Rationale re: the leakage worry — plate location is NOT leakage: it is pre-swing, exogenous (pitcher-
  chosen), and absent from the count+outcome target. It is a *confounder* (common cause of both swing
  shape and outcome, design Limitation #2), so conditioning on it *reduces* shape-outcome confounding
  and de-confounds the at-contact angular features — the opposite of a leak. Only caveat is
  interpretational (location & the at-contact angles are collinear, so don't read a shape-vs-location
  credit split off feature importances; the causal shape question lives in Part C.2). Re-running the
  sweep fresh (old 90 runs were on the old feature set) to re-tune PARAMS for this feature set.
- **Resolved.** Re-ran the sweep (90 runs) on the 11-feature set; applied gap-aware picks to PARAMS
  (p_bip d4/lr.08/mcw20/ss.6/cs.8/λ1/α5/γ.5/1336t; p_foul d5/lr.05/mcw50/ss.7/cs.9/λ0/α5/γ0/1413t;
  v_bip d3/lr.1/mcw1/ss.9/cs.9/λ1/α1/γ1/412t). 2026 held-out recovered: p_bip AUC 0.786, p_foul 0.827,
  v_bip r2 0.048; validation gate corr 0.957. Matches the original design feature set — `same_hand`
  vs stand_L/p_throws_L was immaterial; plate location was the whole ~0.06 AUC. `xrv_grade` mean 50,
  sd 10, range 2-100.

## M3 — xRV script consolidation + autoresearch hparam sweep (2026-07-08)

- **Autoresearch sweep** (`experiments/bench.py` + `experiments/sweep.py`, branch
  `autoresearch/xrv-hparams-20260708`): random-searches XGBoost hyperparameters for each xRV
  sub-model. **Anti-overfit split: train 2024, select on 2025 val (season transfer), 2026 never
  loaded** during the sweep; every run logs the train-val gap; search space weighted to regularization;
  ties broken toward the simpler config. Resumable (skips logged configs, recovers best from
  `autoresearch.jsonl`) so kills only cost the in-flight config.
- **Consolidated `src/xRV_model.py`** (user: too many functions). Removed the in-script tuning/CV
  machinery (`tune_and_fit`, `cv_score`, `make_model`, `sample_configs`, `_py`) — the sweep owns
  hyperparameter search now. Hyperparameters are a single fixed `PARAMS` dict (VAL-selected;
  p_bip = sweep winner, p_foul/v_bip provisional baselines pending the running sweep). 11 functions →
  7; `main()` is now a linear train-3-models → assemble → validate → report flow with no early
  stopping (fixed `n_estimators` per model). Behavior otherwise unchanged (same features, run-value
  tables, `lw_raw` baseline). `bench.py` still imports `FEATURES`/`build_features`/`load_run_value_tables`/
  `bip_value_target` from it.
- **Sweep done (90 runs, 30/model). Headline: tree-hyperparameter tuning is near-worthless here** —
  best-vs-baseline VAL improvement was only −0.16% (p_bip), −0.22% (p_foul), −0.08% (v_bip), all
  noise-level, and the raw-val winners carried the *worst* train-val gaps. So I selected **gap-aware**:
  among configs within 0.0015 val of the best, the smallest-gap (shallow, regularized) one.
  Picks now in `PARAMS`: p_bip d4/lr.08/mcw20/ss.6/cs.8/λ1/α5/γ.5/1654t (gap 0.018 vs 0.038);
  p_foul d5/lr.03/mcw20/ss.7/cs.6/λ5/α1/γ.1/2056t; v_bip d3/lr.05/mcw5/ss.7/cs.8/λ1/α.5/γ1/993t
  (gap 0.005 vs 0.010).
- **2026 held-out confirmation** (consolidated script): p_bip logloss 0.5292 / AUC 0.788; p_foul
  0.4761 / 0.828; v_bip rmse 0.4590 / r2 0.050; realized_rv vs delta_run_exp **corr 0.957**. These
  match the earlier raw-search run to ~3 decimals, so the more-regularized picks generalize to 2026
  just as well — the anti-overfit selection cost nothing on the real holdout.
- **Autoresearch conclusion:** the tree-hyperparameter search path is exhausted (flat landscape); more
  random search on the same space is wasteful. Higher-leverage next paths (not run): feature
  engineering, and the v_bip weak-signal question (r2~0.05 is a ceiling for a mediator-free BIP-value
  model, not a tuning problem).

## M3 — xRV run-value baseline fix: use lw_raw (out in play is negative) (2026-07-08)

- **Bug (baseline):** the run-value layer used `linear_weights.lw` (the OUT-centered column, where
  `out_in_play = 0`) as the Model-3 target, then rebased with `mu` = freq-wtd mean lw (0.2477) to
  reach the ERV frame. That made the BIP target semantically wrong — an out in play read as 0 rather
  than a real ~ -0.25-run cost. `linear_weights.csv` carries both columns: `lw` (out-centered) and
  `lw_raw` (avg-PA-centered, mean event ~ 0, `out_in_play = -0.2476`). ERV from `count_values` is also
  avg-PA-centered (ERV(0,0) ~ 0), so **`lw_raw` is the column that shares ERV's frame.**
- **Fix:** switched the target and all run-value math to `lw_raw` and **dropped `mu` entirely**
  (mu_raw ~ 0.0001, so it was a no-op once on the right column). Now: BIP contribution = `lw_raw - ERV(b,s)`;
  2-strike whiff (K) = `lw_raw[K] - ERV(b,2)`; Model-3 target = `lw_raw(outcome)` (out_in_play = -0.248,
  negative). `load_run_value_tables` no longer returns `mu`; docstring/report updated.
- **No change to final xRV.** `lw = lw_raw + 0.2476` (a constant shift across all outcomes), and the
  old assembly subtracted that same constant as `mu`, so the assembled `xrv` and `v_bip` RMSE/R² are
  numerically identical — this is a correctness/clarity fix of the target scale, not an output change.
  Smoke check: BIP target min -0.248 (out rows negative), realized_rv vs delta_run_exp corr 0.959.

## M3 — xRV features: + pitcher handedness, − game state (leakage exploration) (2026-07-08)

- **Added pitcher handedness.** `pitcher_throws` (R/L) was in `pbp_raw` but not either extract. Added
  `r.pitcher_throws` to `extract.py`'s QUERY and `"pitcher_throws"` to `features.py` KEEP (source of
  truth for future runs), and **backfilled** both cached parquets from a one-off `pbp_raw` pull
  (play_id→pitcher_throws, 0 nulls: swings_2024_2026 759,083 R / 283,208 L; swings_model 579,006 R /
  216,717 L) so no slow full re-extract was needed. `xRV_model.build_features` now derives
  `p_throws_L`; kept alongside `stand_L` so the trees learn the platoon interaction. It draws ~3-5%
  in-sample gain across the three models.
- **Dropped game state (outs + base-out flags) after an empirical leakage check.** User flagged a
  leakage concern. Compared full vs lean feature sets on held-out 2026 (fixed representative configs,
  batter-grouped early stop):
  | model | lean (no game state) | full (+ game state) | game-state in-sample gain |
  |---|---|---|---|
  | p_bip  | logloss 0.5313 / AUC 0.7858 | 0.5309 / 0.7861 | 10.7% |
  | p_foul | logloss 0.4763 / AUC 0.8286 | 0.4768 / 0.8284 (worse) | 8.0% |
  | v_bip  | rmse 0.4579 / R² 0.0551 | 0.4578 / 0.0556 | 13.6% |
  Game state takes 8-14% of in-sample split gain but gives ~0 held-out lift (slightly hurts p_foul) —
  the signature of fitting situation-correlated noise, not swing signal. **Precise framing:** it is
  NOT target leakage (base-out is pre-pitch and absent from the count+outcome run-value target, so it
  can't and doesn't inflate held-out metrics). The real issues are (1) interpretational confounding —
  base-out isn't a property of the swing, so conditioning on it lets the Part C.1 per-cluster grade
  absorb *deployment* rather than shape quality, against the design's "net of situational leverage"
  estimand; (2) mild overfit. Set `USE_GAME_STATE = False` (kept as a toggleable `GAME_STATE` group +
  build_features still computes the base flags, so flipping back is one line). Re-ran the full pipeline.

## M3 — bespoke xRV: swing-outcome decomposition (structure) (2026-07-08)

- Added `src/xRV_model.py`. Per-swing expected run value via a 3-model swing-outcome tree the user
  specified, all conditioned on the SAME pre-swing predictors — pitch/situation **context + the 5
  swing-shape features** (user chose shape-aware, a deliberate departure from Part B's mediator-free
  baseline) — and **no post-contact mediators** (exit velo / launch angle excluded, incl. from the
  xwOBACON piece per user):
  - `p_bip` — XGBClassifier, P(ball in play | swing). Population: all competitive swings.
  - `p_foul` — XGBClassifier, P(foul | swing, ¬BIP); whiff = 1−p_foul. Population: ¬BIP subset
    (target = `is_contact`, which on a non-BIP swing == foul). Whiffs 185,743 / fouls 311,784.
  - `v_bip` — XGBRegressor, E[linear-weight run value of the batted ball] ("xwOBACON"). Population:
    BIP (298,196). Target = `lw` of `pa_outcome` (hits→own weight, all other BIP→out_in_play=0).
- **Run-value layer built from the three CSVs** (RE24-style: value of resulting state − ERV of the
  count left). `count_values`→ERV(b,s); `linear_weights`→outcome `lw` (out-centered) + `mu` =
  freq-weighted mean lw (rebases lw onto the ERV frame, lw'=lw−mu); `count_transitions` loaded as a
  marginal-rate reference. Whiff/<2-strike-foul = ERV(b,s+1)−ERV(b,s); 2-strike whiff = (lw_K−mu)−ERV(b,2);
  2-strike foul = 0. Assembly:
  `xRV = P(BIP)·((V_bip−mu)−ERV) + (1−P(BIP))·[P(foul)·rv_foul + (1−P(foul))·rv_whiff]`.
- **Tuning:** randomized search (N_SEARCH=8) over depth/lr/subsample/colsample/min_child_weight/lambda,
  **GroupKFold by `batter_id`** (no hitter split across folds), XGBoost early stopping on each fold;
  search runs on a 250k subsample for speed, best config refit on full train at the CV-chosen
  iteration count. **2026 held out** as the final test season (train/tune 2024-25).
- **Validation gate (design requirement):** `validate_run_value_tables()` checks the tables' *realized*
  run value against `delta_run_exp`. Smoke test (30k slice) → **corr 0.957**, so the CSV-derived tables
  reproduce the ground-truth per-pitch RE change. Aggregate hitter xRV vs `pitch_values.ipv` still TODO.
- Outputs (gitignored): `data/xrv_models/{p_bip,p_foul,v_bip}.json` + `reports.json`,
  `data/xrv_swings.parquet` (play_id, model probs, xrv, realized_rv), `data/xrv_report.md`.
- Installed `xgboost==3.3.0` into the shared `driveline` venv (was lightgbm-only). API notes for this
  env: sklearn 1.9 dropped `mean_squared_error(squared=False)` → use `root_mean_squared_error`;
  XGBoost 3.3 takes `early_stopping_rounds` as a **constructor** kwarg (not a fit arg).

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

### Validation section added to `swing+_results.ipynb` (2026-07-14)
Appended a four-part validation block after the repertoire leaderboard; notebook runs clean
end-to-end on the `driveline` kernel. All figures/tables save to `results/plots/`.
- **Year-over-year (per-season agreement).** `xrv`/`realized_rv` vs the `delta_run_exp` anchor,
  by season. Realized-RV gate holds: corr(realized, actual) ≈ **0.957** every year (2024 .958,
  2025 .957, **2026 held-out .957** — no train/test gap). Per-swing corr(predicted xRV, actual)
  ≈ 0.21 (outcome noise dominates a single swing, expected). `xrv_per_season_validation.png`.
- **Year-over-year (Swing+ reliability).** Batter Swing+ across consecutive seasons (≥200 swings
  both years): 2024→2025 r=**0.740** (n=340), 2025→2026 r=**0.712** (n=284) → Swing+ is a stable
  batter skill. `swingplus_yoy_reliability.png`.
- **Expected vs actual scatter.** Batter-level (≥300 swings): mean predicted xRV vs mean actual
  `delta_run_exp`, r=**0.373**, OLS slope 0.71, n=565. Aggregating to the batter is the honest
  view (per-swing is noise-dominated). `xrv_expected_vs_actual.png`.
- **Feature importance (per model, gain %).** Loaded the three saved boosters from
  `data/xrv_models/` via `xRV_model.FEATURES`. Each model leans on a different signal:
  p_bip → `horz_attack_angle_pull` (33%), p_foul → `plate_z_norm` (23%), v_bip → `bat_speed`
  (29%). `xrv_feature_importance.png`.
- **Predictive validity vs real hitting outcomes.** `season_stats_hitting` (MLB, R, per
  batter-season, ≥150 swings & ≥200 PA, n=967). Swing+/mean-xRV corr: OPS **0.37**, wRC+ **0.35**,
  wOBA **0.35**, WAR 0.22, BA **0.03**. Swing+ tracks damage/OBP-driven value, not batting avg.
  **`mlb_db` has no OPS+ column** — used `wrc_plus` (park/league-adjusted, 100=avg) as the analog;
  available cols: `ba, obp, slg, ops, iso, woba, wrc_plus, war` in `season_stats_hitting`.
  The two corr columns are identical by construction (Swing+ is an affine transform of xRV).
  `swingplus_predictive_corr.png`.

- **Update (same day):** replaced the predictive-validity *table* with a **faceted scatter**
  (`swingplus_predictive_scatter.png`) — one panel per outcome (BA, OPS, wRC+, wOBA, WAR), Swing+
  on x, each with Pearson r + OLS fit. Dropped the redundant `vs mean xRV` column (affine dup of
  Swing+) and `n (non-null)`. Gotcha: a plotting cell ending in `print()` (not a bare expression)
  doesn't reliably auto-flush the figure to inline `display_data` under nbconvert — add an explicit
  `plt.show()`. (The old `swingplus_predictive_corr.png` table is now stale.) Note: the M3
  expected-vs-actual scatter cell was removed from the notebook by hand during editing.

### R/gt leaderboards with headshots (2026-07-14)
Added `src/leaderboard_table.R` — Swing+ and Repertoire+ leaderboards via **gt + mlbplotR**
(MLBAM headshots, `gt_theme_538`, goldenrod→blue `data_color` gradient on the value column).
Reads the same `data/*.parquet` and writes `results/plots/{swingplus,repertoire}_leaderboard_gt.png`.
`batter_id` IS the MLBAM id, which is exactly what `gt_fmt_mlb_headshot()` keys on (headshots load
for prospects too; missing Swing+ renders as "-" via `sub_missing`). The notebook gained an
"Aesthetic leaderboards" section that shells out to the script (`subprocess.run(..., check=True)`)
and displays the PNGs inline.
- **Env:** R 4.6.0 at `C:\Users\theo.an-yeung\AppData\Local\Programs\R\R-4.6.0\bin\Rscript.exe`
  (NOT on PATH). Packages present: arrow, dplyr, gt, gtExtras, mlbplotR, scales, webshot2.
  `gtsave()` PNG export needs headless Chrome via webshot2 — works in this env.
- **Gotcha:** a quoted bash heredoc (`<<'EOF'`) does NOT expand `$SHELLVAR` inside the R source,
  so pass paths as R literals / via the working directory, not shell interpolation.

- **Update (same day):** `leaderboard_table.R` reworked. Palette flipped to diverging
  **blue(low)→white→red(high)** with `data_color` domain **fixed to the full qualified pool**
  (508 batters / 703 units / 1380 shapes) rather than the shown 25 — so Top 25 reads all-red,
  Bottom 25 all-blue. Added **Bottom-25** tables for Swing+ and Repertoire+, and **Top+Bottom-25**
  for **Swing+ by shape** (per-(batter,stand) cluster; headshots repeat per batter by design).
  Bottom tables keep global ranks via `tail()` on the full desc-sorted pool. Six PNGs total; the
  notebook cell displays all six. Refactored the six gt tables through one `make_leaderboard()`
  helper (cols_label via `.list=`, `data_color`/`cols_align` via `all_of()`, `gt_theme_538(quiet=TRUE)`).

- **Update (same day):** by-shape (archetype) tables now show **% of the hitter's (stance) swings
  in that shape** (usage share) instead of raw swing count — `UsageProp = Swings / sum(Swings)`
  within each (batter, stand) unit (denominator includes sub-100-swing clusters), rendered via a
  new optional `pct_col` arg on `make_leaderboard()` (`fmt_percent`). Decluttered `swing+_results.ipynb`:
  deleted the pandas Swing+/by-cluster/Repertoire+ leaderboard cells, the `cluster_table()` per-hitter
  drill-down, and their section-title markdowns; the first cell is now setup-only (paths + imports).
  Notebook flow is setup → R leaderboards (6 gt tables) → validation. Removed the now-orphaned pandas
  PNGs (swingplus_leaderboard/by_cluster/clusters, repertoire_leaderboard, swingplus_predictive_corr).

- **Update (same day):** rebuilt the per-hitter `cluster_table` drill-down as an **R gt** table.
  `leaderboard_table.R` now takes an optional name arg (`Rscript src/leaderboard_table.R "Arraez"`):
  it builds `cl_pool` + `pal_cl` once (shared with the by-shape leaderboards), then in drill mode
  renders only that batter's shapes (substring match, both stances for switch hitters) ranked by
  Swing+ and colored on the **league scale** (so a below-average hitter reads pale/blue), writing
  `shape_drilldown_<slug>_gt.png`, and `quit()`s before the full leaderboard build. Notebook exposes
  `shape_drilldown("name")` (returns an IPython Image) with an Arraez demo. Same headshot / usage-% /
  archetype·situation schema as the by-shape leaderboard.

- **Update (same day):** renamed the per-hitter drill-down to **breakdown** (`shape_breakdown()` in
  the notebook, `shape_breakdown_<slug>_gt.png` output). Title now shows the **actual hitter name**
  (`paste(unique(d$label))`) instead of the literal search term in quotes, and dropped the em dash
  ("Swing shapes by value - Luis Arraez"). Removed the orphaned `shape_drilldown_*` PNGs.

- **Update (same day):** dropped the `mean xRV` column from the Swing+ leaderboards (top & bottom);
  they now show #, Batter, Swings, Swing+ only. Removed the now-unused `xrv` read from `sp_pool`.

- **Update (same day):** `TOP_N` 25 -> 10; all six leaderboards (Swing+, Repertoire+, by-shape;
  top & bottom) now show 10 rows. Color domains still span the full qualified pools.

### Repertoire+ made count-aware (2026-07-16)
Reworked the Repertoire+ metric in `repertoire.py`. The old metric (usage-weighted **mean**
pairwise centroid distance) is count-blind — it measures average dissimilarity of two random
swings, so hitters with 2 extreme shapes outranked ones with 6 moderate shapes (corr with `k` was
**−0.12**; every top-25 unit had k=2). New:
`expansiveness = mean_pairwise_dist × effective_shapes`, where `effective_shapes = 1/Σweight²`
(inverse-Simpson, usage-effective shape count). Now corr with `k` ≈ **+0.82**; top-10 is
Arraez (6), Raleigh, Rafaela, Busch, Ohtani, Wood, Yelich (k=4–6) — genuinely deep repertoires.
- **Why this form:** prototyped 3 count-aware candidates against the real 703-unit data.
  `mean_dist × effective_shapes` chosen (usage-aware, decomposable into "how different" × "how
  many effectively", a 99/1 hitter's rare shape barely counts) over **MST branch length** (r=0.94
  but pure geometry, drops usage) and **Rao's Q** (weighted sum; r=0.67, count reward saturates).
  User picked this option.
- `repertoire_scores.parquet` gains diagnostic cols `mean_pairwise_dist` and `effective_shapes`;
  `expansiveness` is now the product. Renamed the helper `expansiveness()` → `mean_pairwise_dist()`.
  k=1 → 0 floor unchanged. Regenerated the parquet, catalog, and the R Repertoire+ leaderboards.
- Updated the "confirmed decision" note in research-design.md (was "**mean** (not max) pairwise
  distance") and the CLAUDE.md Repertoire+ paragraph.

- **Update (same day):** tempered the count term after checking reliance — `× effective_shapes`
  (eff¹) overshot: count drove **84%** of the ranking variance (corr with `k` +0.82) and shape-count
  groups barely overlapped (a wide k=2 couldn't beat almost any k=4+). Changed to
  `expansiveness = mean_pairwise_dist × √effective_shapes` (eff^0.5). Now balanced: corr with `k`
  +0.59, spread-corr 0.66 ≈ count-corr 0.65, log-variance split ~61% spread / 50% count, and
  k-groups overlap (Johnathan Rodríguez k=2 re-enters the top 10; Oneil Cruz k=4 is #2; Arraez k=6
  no longer auto-#1). Exponent is the single tunable knob if more/less count weight is ever wanted.

### Repertoire+ pegged to a frozen 2024-25 baseline (2026-07-16)
`repertoire.py` now computes the standardization + scaling reference **once** from the 2024-25
cohort and persists it to `src/repertoire_reference.json` (committed; feature SDs, expansiveness
mean/SD, and the full percentile grid — league aggregates only, no PII). Later runs load and reuse
it instead of re-baselining, so `repertoire_plus` / `repertoire_pctile` stay comparable when 2026+
is added (OPS+/wRC+-style fixed baseline). `repertoire_plus` = `50 + 10·z` on the frozen mean/SD;
`repertoire_pctile` = position in the frozen expansiveness distribution (`np.searchsorted`).
Refactored the per-unit loop into `compute_units()`; added `resolve_reference()`. Delete the JSON to
re-peg. Numerically a no-op right now (24-25 SDs ≈ pooled SDs; top units unchanged — Raleigh 68.9,
Cruz 68.2, Wood 67.6). **Known gap:** `cluster_summary` centroids are pooled across all clustered
seasons, so a genuine per-season cross-season visual still needs per-season centroids (unbuilt) —
the peg fixes scale drift only. Kept this out of CLAUDE.md per user preference (docs/ is the home
for methodology detail).

### Context-responsiveness (adjustability) metric built (2026-07-16)
New Facet-2 stage `src/context_response.py` → `data/context_response.parquet` +
`context_response_catalog.md` (gitignored data). Per (batter, stand) unit, 2024-25, ≥300 swings &
k≥2 (512 units): how much shape choice depends on pre-swing context — the paper's adjustability
metric, distinct from Repertoire+ width. Two estimators (both, per user): (1) null-adjusted
normalized MI / uncertainty coefficient `U=(I(C;S)-null)/H(S)` over joint + per-axis context
(count / pitch-group / location), permutation null (B=200) for bias, ÷H(S) to strip repertoire
entropy; (2) classifier skill (OOF log-loss lift of a multinomial logit over the usage prior) as a
cross-check. Headline `responsiveness` = null-adj overall U.
- **Validation:** two estimators agree (r≈0.45); corr with k ≈ −0.35 (NOT repertoire size — the
  normalization works); corr with n_swings ≈ −0.16.
- **Key finding:** dependence is almost entirely **location** (resp_loc ≈0.22, resp_pitch ≈0.17)
  with **near-zero count-responsiveness** (resp_count ≈0.006). At-contact shape is largely *forced*
  by pitch location, not volitional in-count adjustment — contamination guardrail + Limitation #1
  (contact-point-only geometry) in the data. Payoff test should lean on resp_count / a
  location-excluded variant; be cautious calling the raw headline "adjustability." Documented in
  research-design.md Part D. Not pegged yet (single-cohort percentile); pegging can mirror
  repertoire if cross-season is needed.
