# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research project on MLB swing shapes using Statcast bat tracking (2024+). It asks two questions:

1. **Swing-shape value (Facet 1).** A per-batter GMM clusters a hitter's swing shapes, and a
   bespoke xRV model grades each shape's run value conditioned on context (count, location, pitch
   type, base-out state).
2. **Repertoire diversity (Facet 2).** Batter-level scalars (repertoire size, usage entropy,
   repertoire expansiveness, context-responsiveness) test whether a wider, more *adjustable*
   repertoire improves outcomes.

`docs/research-design.md` is the source of truth for methodology, confirmed design decisions,
limitations, and milestones. Read it before you touch modeling code. The decisions there are
settled and should not be relitigated: strictly per-batter GMM, bespoke xRV, and a shape defined as
5 mechanics features that exclude intercept location. `docs/worklog.md` tracks what's actually been
built.

## Environment & commands

This project uses the **shared `driveline` venv** at `~/.venvs/driveline` (uv-managed, built on
miniforge CPython 3.13). It's one env reused across Driveline workflow, not a project-local
`.venv`. The IDE points at it via `.vscode/settings.json` (`python.defaultInterpreterPath`), and a
matching Jupyter kernel is registered as **"Python (driveline)"** (`--name driveline`). Select
that kernel for `.ipynb` files.

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

`tabulate` is required (`cluster.py`'s `.to_markdown()`); `matplotlib` for the viz notebook;
`jinja2` for pandas `.style` (heatmaps in `cluster_results.ipynb`); `dataframe_image` to save
table/Styler outputs as PNGs (see `results/plots/` note below).

Run pipeline stages from repo root with the `driveline` env active (each reads/writes `data/`):

```
python src/extract.py     # mlb_db -> swings_2024_2026_mlb.parquet + profile.md (slow: full DB pull)
python src/features.py    # competitive-swing filter -> swings_model.parquet
python src/cluster.py     # per-batter GMM -> cluster_* + batter_repertoire + cluster_catalog.md
python src/xRV_model.py   # bespoke per-swing xRV -> xrv_swings.parquet (+ xrv_grade)
python src/interpret.py   # Layer-1 archetype lexicon -> shape_archetypes + archetype_lexicon.md
python src/cards.py       # Layer-2 swing ID cards -> shape_cards.parquet + shape_cards_catalog.md
python src/repertoire.py  # Facet-2 Repertoire+ (repertoire expansiveness) -> repertoire_scores.parquet + repertoire_catalog.md
```

**R leaderboards (`src/leaderboard_table.R`):** the presentation-grade Swing+ / Repertoire+
leaderboards use **R (gt + mlbplotR)** for MLBAM headshots — `batter_id` is the MLBAM id
`gt_fmt_mlb_headshot()` keys on. Run `Rscript src/leaderboard_table.R` from repo root; it reads
`data/*.parquet` and writes `results/plots/{swingplus,repertoire}_leaderboard_gt.png`. `swing+_results.ipynb`
shells out to it and displays the PNGs. R 4.6.0 lives at
`C:\Users\theo.an-yeung\AppData\Local\Programs\R\R-4.6.0\bin\Rscript.exe` (NOT on PATH); packages
arrow/dplyr/gt/gtExtras/mlbplotR/scales/webshot2 are installed, and `gtsave()` PNG export needs
headless Chrome (webshot2), which works in this env.

**Pipeline order:** extract → features → cluster → xrv (built). `interpret.py` (Layer 1 = cross-unit
archetype lexicon) and `cards.py` (Layer 2 = per-hitter swing ID cards: name-delta-vs-primary,
over-index when-label, grade + within-batter matched contrast) are the interpretability overlay and
consume cluster + xrv outputs. `shape_card(name)` in cluster_results.ipynb renders a hitter's cards. `repertoire.py` (Facet-2
repertoire expansiveness = **Repertoire+**) consumes cluster_summary + swings_model and is the first
built Facet-2 stage; `value_model → within_batter → diversity → reports` remain unbuilt. Each stage
is a standalone script with a `main()`; there is no test suite or build step yet.

**Repertoire+ (`repertoire.py`):** a purely descriptive measure of repertoire *width*. It's the
usage-weighted mean pairwise Euclidean distance between a unit's cluster centroids, with each of the
5 shape features standardized by **cohort (league) swing-level SD** so it's cross-hitter comparable
(rankable). Geometry only: no run value, quality, or adjustability. All 5 features are
equal-weighted (incl. bat_speed). Lead with `repertoire_pctile`, not `repertoire_plus`, because 13%
of units are single-shape and pile up at a 0-spread floor that skews the "50 = average" reference.
`repertoire_plus` is on the **same scale as Swing+**: `50 + 10·z` clipped to [0, 100], where 50 is
league-average width. That replaced the OPS+-style `100 + 10·z` it used originally. It reuses
cluster_summary's raw centroids directly, since the horz_attack_angle pull-mirror is
distance-invariant.

**Archetype lexicon (`interpret.py`):** archetypes are defined on the **4 geometry features only**
(tilt, length, VAA, HAA_pull). `bat_speed` is a reported descriptor, not a defining axis, because
its "state not trait" ICC drags a 5-feature carve into an effort bin. This is only a naming
overlay; `cluster.py` still defines shapes with all 5 features. There are 3 archetypes
(`K_ARCH=3`): **Level Oppo / Level Center / Uppercut Pull**. MLB geometry sits on a level-oppo ↔
uppercut-pull diagonal. Note `K_ARCH=3` is a deliberate interpretability choice, not the raw BIC
minimum: after MERGE_SEP moved 2.0→1.75 (2026-07-13) the finer cluster pool makes BIC marginally
prefer K=2 (13188.8 vs 13222.2), and at K=3 the two level components collide in the same naming
cell. We keep 3 for the useful middle band and moved the `HAA_OPPO` naming boundary −5.0→−6.5 so
they name apart (Level Center at haa_pull ≈−5.6 vs Level Oppo ≈−7.6). `cards.py` (Layer 2) enriches each with a
`context_tag` (top-3 over-indexed situations) to produce `archetype_detailed`, so same-archetype
shapes read apart. Cluster 0 (the primary swing) is labeled `"Primary"` in `archetype_detailed`,
while the true archetype stays in `archetype_name`.

**Notebook plot theme:** standard analytical charts use `plt.style.context('fivethirtyeight')` with a
white-background override (`figure/axes/savefig.facecolor='white'`, plus **`grid.color='#cbcbcb'`**,
because fivethirtyeight's default grid is white and vanishes on a white bg). The `usage_heatmap`
pandas Styler (cell 10) is a white-bg table with a fivethirtyeight blue→white→red diverging gradient
(`FT_DIV`) to match. Only the bespoke Baseball-Savant swing cards in `cluster_results.ipynb` (cell
8, dark navy `BG/INK/MUT/GRID` palette plus hand-drawn art) stay dark by design. Both notebooks run
on the **`driveline`** Jupyter kernel. One notebook-authoring gotcha: don't build cell source via a
triple-quoted string with `\n` + `splitlines()`, because the escapes become real newlines and split
string literals. Use an explicit line list instead.

**Handedness convention (validated vs `bearing_angle`; we got this wrong once, see worklog
2026-07-09):** `horz_attack_angle` is **batter-relative** (raw + = opposite field for both hands),
so the pull frame is a **uniform negation**: `horz_attack_angle_pull = -horz_attack_angle` (+ =
pull, both hands). There is NO per-hand mirror. `plate_x` is **absolute** (catcher frame), so
pull-side/inside needs a **real per-hand flip**: `plate_x * (L? +1 : -1)`. Don't apply the same
mirror to both; they differ. `vert_attack_angle`, tilt, length, and bat_speed are
handedness-neutral.

DB credentials (`BIOMECH_DB_HOST/PORT/USER/PASS`) resolve from `~/.claude/.env` via the
`get_secret` helper in `extract.py`. Read-only user, database `mlb_db`. Full schema:
`~/.claude/skills/mlb-db-analysis/docs/schema.md`.

## Architecture & conventions

- **`data/` is gitignored** and holds all extracts. Never commit it; it contains athlete data.
  Parquet is the interchange format between stages, and markdown (`profile.md`,
  `cluster_catalog.md`) is the human-readable output of each stage.
- **`results/plots/` holds rendered notebook outputs as PNGs** (committed, unlike `data/`). Both
  notebooks write there via `PLOTS = ROOT / 'results' / 'plots'`: matplotlib figures via `fig.savefig`,
  and table/Styler outputs (the cell-10 usage heatmap, the `swing+_results.ipynb` leaderboards) via
  `dataframe_image.export(obj, str(path), table_conversion='matplotlib')`. Use the **`matplotlib`**
  backend, not the default `chrome`/`selenium` (no browser in this env; `matplotlib` renders the
  gradient heatmap fine and needs no extra binary). Figures aggregate cohort-level results only, so no
  athlete PII gate is needed.
- **Competitive swing** (no DB flag exists, so `features.py` defines it): bat-tracked (5 shape
  features present) + not a bunt + `bat_speed >= 50` + angle artifacts dropped
  (`|horz_attack_angle| <= 45`, `vert_attack_angle` in [-45, 75]).
- **Shape feature vector = 5 mechanics features:** `swing_path_tilt`, `swing_length`, `bat_speed`,
  `vert_attack_angle`, `horz_attack_angle`. Clustering uses `horz_attack_angle_pull` (handedness-
  mirrored, + = pull) so L/R hitters share a frame. Intercept *location* coords are deliberately
  excluded from shape (98–99% pitch-location artifact). `ball_bat_intercept_y` is kept only as a
  separate timing descriptor, never in the shape vector or as an xRV mediator.
- **Clustering unit = `(batter_id, batter_stand)`, NOT batter alone.** A switch hitter's L and R
  swings are different movements, so each stance clusters (and enters Facet 2) as its own
  "player": Cal Raleigh L vs Cal Raleigh R. Only `horz_attack_angle` is handedness-mirrored, and
  pooling both stances would make the GMM separate on stance instead of shape. All three outputs
  carry `batter_stand`. `batter_repertoire` and `cluster_summary` also carry a display `label` that
  suffixes the stance **only for switch hitters** ("Cal Raleigh (L)"), leaving one-way hitters
  bare ("Aaron Judge").
- **Clusters are strictly per-unit and NOT comparable across units.** Cluster 0 = that unit's
  primary (highest-usage) swing. All cross-unit analysis must use unit-level scalars
  (`batter_repertoire.parquet`), never shared cluster IDs.
- **GMM k selection is minimum-BIC (early-stop, no occupancy floors) followed by a post-BIC
  merge.** BIC over-segments at large n into large-but-near-duplicate components, so `cluster.py`
  merges component pairs closer than `MERGE_SEP=1.75` (within-cluster-SD Mahalanobis) into one
  shape. Reported `k` is post-merge. This is not an occupancy floor: the phantom components are
  large (~28% usage), so the problem is separation, not size. Identifiability cap is
  `k_max = n // 20`. Cohort is ≥150 competitive swings **per `(batter, stand)` unit** (lowered from
  pooled-300 to keep switch hitters' weaker side). Post-merge: mean k ≈2.26, median 2, max 6
  (MERGE_SEP=1.75; was ≈1.9 at 2.0). 13% of units are single-shape.
- **Known confound:** count-based diversity metrics (`k`, `effective_shapes`) correlate with
  `n_swings` (r≈0.71), so they must be sample-size-controlled before Facet 2.

## xRV status

`data/re24.csv`, `count_values.csv`, `count_transitions.csv`, `linear_weights.csv` are xRV
building blocks (RE24 matrix, count run values, linear weights) generated during exploratory M3
work in `src/cluster_results.ipynb`. The bespoke xRV (`src/xrv.py`) is not yet a committed script.
Per the design, xRV must be validated row-for-row against `delta_run_exp` and calibrated against
`pitch_values.ipv` before any downstream use.

## Documentation protocol

When you change behavior, update `docs/worklog.md` (append what you built + findings), and
`docs/research-design.md` if a methodology decision changes. Keep this file's conventions current
if you discover new project-specific patterns.
