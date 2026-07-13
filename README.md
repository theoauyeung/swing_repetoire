# swing-repertoire

Research on MLB swing shapes using Statcast bat tracking (2024+): (1) grading the run value of
each hitter's individual swing shapes conditioned on the situation, and (2) testing whether a
wider, more *adjustable* swing repertoire actually improves outcomes.

Data source: Driveline `mlb_db` (Statcast bat tracking, MLB 2024+). See `docs/research-design.md`
for the full design, data reality, methodology, limitations, and milestones (that design doc is
kept locally and not published to this repo).

### Two facets
1. **Swing-shape value.** A per-batter GMM clusters swing shapes, and a bespoke xRV model assigns
   each shape a run value conditioned on count, pitch location, pitch type, and base-out state.
2. **Repertoire diversity.** Batter-level metrics for repertoire size, usage entropy, repertoire
   expansiveness, and context-responsiveness test whether adjustability pays off.

### Usage
Scripts are standalone pipeline stages, run in order from the repo root:
```bash
python src/extract.py     # mlb_db -> data/swings_2024_2026_mlb.parquet + profile.md
python src/features.py    # competitive-swing filter -> data/swings_model.parquet
python src/cluster.py     # per-batter GMM -> data/cluster_* + batter_repertoire + catalog
```
Pipeline order: `extract → features → cluster → (xrv → value_model → within_batter → diversity →
reports, not yet built)`.

Both result notebooks write their outputs to `results/plots/` (committed to the repo):
`src/cluster_results.ipynb` saves its figures (PNGs, plus the usage-heatmap table as HTML), and
`src/swing+_results.ipynb` saves its Swing+ / Repertoire+ leaderboards as HTML tables.

## Details
- **Project Owner:** Theo Au-Yeung
- **Project's Notion Page:** [https://notion.so](https://notion.so) *(TBD)*
- **Project's Slack Channel:** `#proj-swing-repertoire` *(TBD)*

## Contributors
* [@theoauyeung](https://github.com/theoauyeung)

## Getting Started

### Setup the environment
This project uses the **shared `driveline` uv venv** (`~/.venvs/driveline`), one environment
reused across Driveline workflow. Activate it, or select the **"Python (driveline)"** kernel in
your IDE or for notebooks. To (re)create it:
```bash
uv venv ~/.venvs/driveline --prompt driveline
VIRTUAL_ENV=~/.venvs/driveline uv pip install pandas pyarrow scikit-learn lightgbm scipy numpy \
    mysql-connector-python requests matplotlib tabulate jinja2 ipykernel
python -m ipykernel install --user --name driveline --display-name "Python (driveline)"
source ~/.venvs/driveline/Scripts/activate   # Git Bash; Windows: ~/.venvs/driveline/Scripts/activate.bat
```

### Setup the project
- `data/` (gitignored) holds all extracts. It is **never committed**, because it contains player
  data.
- DB credentials resolve from `~/.claude/.env` as `BIOMECH_DB_HOST/PORT/USER/PASS` via the
  `get_secret()` helper in `src/extract.py` (the `mlb-db-analysis` skill convention). This project
  reads creds from that file rather than a repo-local `.env`. The template `.config`/`.env`
  scaffolding is retained but unused by the pipeline scripts. Read-only user, database `mlb_db`.

### Test-run
```bash
python src/extract.py
```


