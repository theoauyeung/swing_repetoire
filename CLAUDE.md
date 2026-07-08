# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research project on MLB swing shapes using Statcast bat tracking (2024+). Two questions:

1. **Swing-shape value (Facet 1)** — per-batter GMM clusters a hitter's swing shapes; a bespoke
   xRV model grades each shape's run value conditioned on context (count, location, pitch type,
   base-out state).
2. **Repertoire diversity (Facet 2)** — batter-level scalars (repertoire size, usage entropy,
   shape dispersion, context-responsiveness) test whether a wider, more *adjustable* repertoire
   improves outcomes.

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
```

**Pipeline order:** extract → features → cluster → (xrv → value_model → within_batter →
diversity → reports, not yet built). Each stage is a standalone script with a `main()`; there is
no test suite or build step yet.

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
- **GMM k selection is pure minimum-BIC** (early-stop, no arbitrary occupancy floors); the only
  bound is the identifiability cap `k_max = n // 20`. Cohort = ≥150 competitive swings **per
  `(batter, stand)` unit** (lowered from pooled-300 to keep switch hitters' weaker side).
- **Known confound:** count-based diversity metrics (`k`, `effective_shapes`) correlate with
  `n_swings` (r≈0.71) — must be sample-size-controlled before Facet 2. `shape_dispersion` is
  exempt (confirmed ~orthogonal to `n_swings`).

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
