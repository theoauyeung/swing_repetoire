# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research project on MLB swing shapes using Statcast bat tracking (2024+). Two questions:

1. **Swing-shape value (Facet 1)** — per-batter GMM clusters a hitter's swing shapes; a bespoke
   xRV model grades each shape's run value conditioned on context (count, location, pitch type,
   base-out state).
2. **Repertoire diversity (Facet 2)** — batter-level scalars (repertoire size, usage entropy,
   repertoire expansiveness, context-responsiveness) test whether a wider, more *adjustable*
   repertoire improves outcomes.

`docs/research-design.md` is the source of truth for methodology, confirmed design decisions,
limitations, and milestones. **Read it before modeling work** — the design decisions there
(strictly per-batter GMM, bespoke xRV, shape = 5 mechanics features excluding intercept location)
are settled and should not be relitigated. `docs/worklog.md` tracks what's actually been built.

## Environment & commands

This project uses the **shared `driveline` venv** at `~/.venvs/driveline` (uv-managed, based on
miniforge CPython 3.13) — one env reused across Driveline workflow, not a project-local `.venv`.
The IDE is pointed at it via `.vscode/settings.json` (`python.defaultInterpreterPath`), and a
matching Jupyter kernel is registered as **"Python (driveline)"** (`--name driveline`). Select
that kernel for `.ipynb` files.

```
# one-time setup (already done on this machine):
uv venv ~/.venvs/driveline --python <miniforge python> --prompt driveline
VIRTUAL_ENV=~/.venvs/driveline uv pip install pandas pyarrow scikit-learn lightgbm scipy numpy \
    mysql-connector-python requests matplotlib tabulate jinja2 ipykernel

# activate for a terminal session (Git Bash):
source ~/.venvs/driveline/Scripts/activate
# to add a package later:
VIRTUAL_ENV=~/.venvs/driveline uv pip install <pkg>
```

`tabulate` is required (`cluster.py`'s `.to_markdown()`); `matplotlib` for the viz notebook;
`jinja2` for pandas `.style` (heatmaps in `cluster_results.ipynb`).

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

**Pipeline order:** extract → features → cluster → xrv (built). `interpret.py` (Layer 1 = cross-unit
archetype lexicon) and `cards.py` (Layer 2 = per-hitter swing ID cards: name-delta-vs-primary,
over-index when-label, grade + within-batter matched contrast) are the interpretability overlay and
consume cluster + xrv outputs. `shape_card(name)` in cluster_results.ipynb renders a hitter's cards. `repertoire.py` (Facet-2
repertoire expansiveness = **Repertoire+**) consumes cluster_summary + swings_model and is the first
built Facet-2 stage; `value_model → within_batter → diversity → reports` remain unbuilt. Each stage
is a standalone script with a `main()`; there is no test suite or build step yet.

**Repertoire+ (`repertoire.py`):** purely descriptive repertoire *width* — usage-weighted mean pairwise
Euclidean distance between a unit's cluster centroids, each of the 5 shape features standardized by
**cohort (league) swing-level SD** so it's cross-hitter comparable (rankable). Geometry only: no run
value / quality / adjustability. All 5 features equal-weighted (incl. bat_speed). Lead with
`repertoire_pctile`, not `repertoire_plus` (24% of units are single-shape → 0-spread floor spike that skews
the "50 = average" reference). `repertoire_plus` is on the **same scale as Swing+** — `50 + 10·z`
clipped to [0, 100], 50 = league-average width — not the OPS+-style `100 + 10·z` it used originally.
Reuses cluster_summary's raw centroids directly — the horz_attack_angle pull-mirror is distance-invariant.

**Archetype lexicon (`interpret.py`):** archetypes are defined on the **4 geometry features only**
(tilt, length, VAA, HAA_pull); `bat_speed` is a reported descriptor, not a defining axis (its
"state not trait" ICC drags a 5-feature carve into an effort bin). This is a naming overlay —
`cluster.py` still defines shapes with all 5 features. 3 archetypes (BIC on the post-merge,
handedness-corrected pool): **Level Oppo / Level Center / Uppercut Pull** — MLB geometry sits on a
level-oppo ↔ uppercut-pull diagonal. `cards.py` (Layer 2) enriches each with a `context_tag`
(top-3 over-indexed situations) → `archetype_detailed`, so same-archetype shapes read apart;
cluster 0 (the primary swing) is labeled `"Primary"` in `archetype_detailed` (true archetype stays
in `archetype_name`).

**Notebook plot theme:** standard analytical charts use `plt.style.context('fivethirtyeight')` with a
white-background override (`figure/axes/savefig.facecolor='white'`, and **`grid.color='#cbcbcb'`** —
fivethirtyeight's default grid is white and vanishes on a white bg). The `usage_heatmap` pandas Styler
(cell 10) is a white-bg table with a fivethirtyeight blue→white→red diverging gradient (`FT_DIV`) to
match. Only the bespoke Baseball-Savant swing cards in `cluster_results.ipynb` (cell 8, dark navy
`BG/INK/MUT/GRID` palette + hand-drawn art) stay dark by design. Both notebooks run on the
**`driveline`** Jupyter kernel. Notebook-authoring gotcha:
don't build cell source via a triple-quoted string with `\n` + `splitlines()` (the escapes become real
newlines and split string literals) — use an explicit line list.

**Handedness convention (validated vs `bearing_angle`; got this wrong once — see worklog 2026-07-09):**
`horz_attack_angle` is **batter-relative** (raw + = opposite field for both hands), so the pull
frame is a **uniform negation**: `horz_attack_angle_pull = -horz_attack_angle` (+ = pull, both
hands). NO per-hand mirror. `plate_x` is **absolute** (catcher frame), so pull-side/inside needs a
**real per-hand flip**: `plate_x * (L? +1 : -1)`. Don't apply the same mirror to both — they differ.
`vert_attack_angle`, tilt, length, bat_speed are handedness-neutral.

DB credentials (`BIOMECH_DB_HOST/PORT/USER/PASS`) resolve from `~/.claude/.env` via the
`get_secret` helper in `extract.py`. Read-only user, database `mlb_db`. Full schema:
`~/.claude/skills/mlb-db-analysis/docs/schema.md`.

## Architecture & conventions

- **`data/` is gitignored** and holds all extracts. Never commit it — it contains athlete data.
  Parquet is the interchange format between stages; markdown (`profile.md`, `cluster_catalog.md`)
  is the human-readable output of each stage.
- **Competitive swing** (no DB flag exists — `features.py` defines it): bat-tracked (5 shape
  features present) + not a bunt + `bat_speed >= 50` + angle artifacts dropped
  (`|horz_attack_angle| <= 45`, `vert_attack_angle` in [-45, 75]).
- **Shape feature vector = 5 mechanics features:** `swing_path_tilt`, `swing_length`, `bat_speed`,
  `vert_attack_angle`, `horz_attack_angle`. Clustering uses `horz_attack_angle_pull` (handedness-
  mirrored, + = pull) so L/R hitters share a frame. Intercept *location* coords are deliberately
  excluded from shape (98–99% pitch-location artifact); `ball_bat_intercept_y` is kept only as a
  separate timing descriptor, never in the shape vector or as an xRV mediator.
- **Clustering unit = `(batter_id, batter_stand)`, NOT batter alone.** A switch hitter's L and R
  swings are different movements, so each stance clusters (and enters Facet 2) as its own
  "player" — Cal Raleigh L vs Cal Raleigh R. Only `horz_attack_angle` is handedness-mirrored;
  pooling both stances would make the GMM separate on stance instead of shape. All three outputs
  carry `batter_stand`; `batter_repertoire` / `cluster_summary` also carry a display `label` that
  suffixes the stance **only for switch hitters** ("Cal Raleigh (L)"), leaving one-way hitters
  bare ("Aaron Judge").
- **Clusters are strictly per-unit and NOT comparable across units.** Cluster 0 = that unit's
  primary (highest-usage) swing. All cross-unit analysis must use unit-level scalars
  (`batter_repertoire.parquet`), never shared cluster IDs.
- **GMM k selection is minimum-BIC (early-stop, no occupancy floors) followed by a post-BIC
  merge:** BIC over-segments at large n into large-but-near-duplicate components, so `cluster.py`
  merges component pairs closer than `MERGE_SEP=2.0` (within-cluster-SD Mahalanobis) into one
  shape. Reported `k` is post-merge. NOT an occupancy floor — the phantom components are large
  (~28% usage), the problem is separation. Identifiability cap `k_max = n // 20`. Cohort = ≥150
  competitive swings **per `(batter, stand)` unit** (lowered from pooled-300 to keep switch
  hitters' weaker side). Post-merge: mean k ≈1.9, median 2, max 5.
- **Known confound:** count-based diversity metrics (`k`, `effective_shapes`) correlate with
  `n_swings` (r≈0.71) — must be sample-size-controlled before Facet 2.

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
