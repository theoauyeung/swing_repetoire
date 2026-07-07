# swing-repertoire

Research on MLB swing shapes using Statcast bat tracking (2024+): (1) grading the run value of
each hitter's individual swing shapes conditioned on the situation, and (2) testing whether a
wider, more *adjustable* swing repertoire actually improves outcomes.

Data source: Driveline `mlb_db` (Statcast bat tracking, MLB 2024+). See `docs/research-design.md`
for the full design, data reality, methodology, limitations, and milestones (design doc is kept
locally / not published to this repo).

### Two facets
1. **Swing-shape value** — per-batter GMM clusters swing shapes; a bespoke xRV model assigns each
   shape a run value conditioned on count, pitch location, pitch type, and base-out state.
2. **Repertoire diversity** — batter-level metrics for repertoire size, usage entropy, shape
   dispersion, and context-responsiveness; tests whether adjustability pays off.

### Usage
Scripts are standalone pipeline stages, run in order from the repo root:
```bash
python src/extract.py     # mlb_db -> data/swings_2024_2026_mlb.parquet + profile.md
python src/features.py    # competitive-swing filter -> data/swings_model.parquet
python src/cluster.py     # per-batter GMM -> data/cluster_* + batter_repertoire + catalog
```
Pipeline order: `extract → features → cluster → (xrv → value_model → within_batter → diversity →
reports, not yet built)`.

## Details
- **Project Owner:** Theo Au-Yeung
- **Project's Notion Page:** [https://notion.so](https://notion.so) *(TBD)*
- **Project's Slack Channel:** `#proj-swing-repertoire` *(TBD)*

## Contributors
* [@theoauyeung](https://github.com/theoauyeung)

## Getting Started

### Setup the environment
Create the conda VENV and install the required packages:
```bash
conda env create -f environment.yml
conda activate swing_repertoire
```

### Setup the project
- `data/` (gitignored) holds all extracts. It is **never committed** — it contains player data.
- DB credentials resolve from `~/.claude/.env` as `BIOMECH_DB_HOST/PORT/USER/PASS` via the
  `get_secret()` helper in `src/extract.py` (the `mlb-db-analysis` skill convention). This project
  reads creds from that file rather than a repo-local `.env`; the template `.config`/`.env`
  scaffolding is retained but unused by the pipeline scripts. Read-only user, database `mlb_db`.

### Test-run
```bash
python src/extract.py
```

## ⚠️ IMPORTANT ⚠️
Repo conventions (from the Driveline project template):
- Use `snake_case` for variable names; store constants in `UPPER_CASE` near the top of the file.
- Store secrets/credentials in `.env` (or `~/.claude/.env` for this project); never in a `.py` file.
- Store machine-specific configuration in `.config`.
