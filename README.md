# swing-repertoire

Statcast bat tracking (MLB 2024–2026) gives us swing geometry on every competitive swing, including whiffs and fouls. That makes it possible to study what a hitter's different swing shapes are worth in context, and whether deploying them situationally actually helps.

Two questions:

1. What is the run value of each hitter's distinct swing shapes, conditioned on count, pitch location, pitch type, and base-out state?
2. Does a wider, more context-sensitive swing repertoire improve outcomes?

Data: Driveline `mlb_db`. Full methodology: `docs/research-design.md` (not in this repo).

## What's built

A per-batter GMM groups each hitter's swings into distinct shapes (median: 2, max: 6). An xRV model grades each shape in context and scales it as Swing+ (50 = league average). Repertoire+ measures how wide a hitter's shape portfolio is across 5 dimensions, weighted by usage. Adjustability measures how much bat speed, swing length, and tilt track the situation, net of where the pitch is.

The main finding: adjustability is a two-strike skill. Hitters who adjust by count give back less run value at two strikes (β = +0.17, p = 0.0001) and whiff less (β = −0.39, p < 1e-8). Season-wide, swing quality and playing time dominate. Raw repertoire width, if anything, hurts at two strikes (β = −0.15, p = 0.001).

## Pipeline

Run from repo root with the `driveline` env active, in order:

```bash
python src/extract.py        # mlb_db → data/swings_2024_2026_mlb.parquet
python src/features.py       # competitive-swing filter → data/swings_model.parquet
python src/cluster.py        # per-batter GMM → cluster_*, batter_repertoire
python src/xRV_model.py      # per-swing xRV → xrv_swings.parquet
python src/interpret.py      # archetype lexicon → shape_archetypes
python src/cards.py          # swing ID cards → shape_cards.parquet
python src/repertoire.py     # Repertoire+ → repertoire_scores.parquet
python src/adjustability.py  # adjustability → adjustability.parquet
python src/payoff.py         # payoff regression → results/payoff.md
Rscript src/leaderboard_table.R  # leaderboard PNGs via gt + mlbplotR
```

Notebooks (`src/cluster_results.ipynb`, `src/swing+_results.ipynb`, `src/adjustability_results.ipynb`) save figures as PNGs into `results/plots/` subfolders. Tables render via `dataframe_image` (matplotlib backend).

## Environment

Uses the shared `driveline` venv at `~/.venvs/driveline` (uv, Python 3.13). Select the "Python (driveline)" kernel for notebooks.

```bash
# activate (macOS/Linux)
source ~/.venvs/driveline/bin/activate

# recreate from scratch
uv venv ~/.venvs/driveline --prompt driveline
VIRTUAL_ENV=~/.venvs/driveline uv pip install -r requirements.txt
python -m ipykernel install --user --name driveline --display-name "Python (driveline)"

# add a package
VIRTUAL_ENV=~/.venvs/driveline uv pip install <pkg>
```

`data/` is gitignored and holds all extracts — never commit it. DB credentials (`BIOMECH_DB_HOST/PORT/USER/PASS`) resolve from `~/.claude/.env` via `get_secret()` in `src/extract.py`. Only needed to re-pull raw data; the resume bundle already has every extract.

## New machine

See [`docs/project_resume.md`](docs/project_resume.md).

## Contributors

[@theoauyeung](https://github.com/theoauyeung)
