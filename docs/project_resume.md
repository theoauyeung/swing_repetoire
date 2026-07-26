# Resuming on a new machine

The **code** is on GitHub. The **data** (`data/`, ~273 MB — including the raw DB extract
`swings_2024_2026_mlb.parquet`) and **internal docs** (`CLAUDE.md`, `docs/`, `results/payoff.md`) are
gitignored, so they ship separately in a **resume bundle** — `swing-repertoire-resume-<date>.zip`,
stored in **Driveline OneDrive** (work account `theo.an-yeung@drivelinebaseball.com`). Steps below are
written for **macOS**.

1. **Get the data bundle from OneDrive (browser — no app needed).**
   - Go to **onedrive.com** (or **office.com** → OneDrive) and **sign in with the Driveline work
     account** `theo.an-yeung@drivelinebaseball.com`. (The bundle is on the *work* OneDrive, not a
     personal Microsoft account.)
   - In *My files* (root), download **`swing-repertoire-resume-<date>.zip`** → it lands in
     `~/Downloads`.
2. **Clone the repo**
   ```bash
   git clone https://github.com/theoauyeung/swing_repetoire.git
   cd swing_repetoire
   ```
3. **Unpack the bundle into the repo root** so `data/`, `CLAUDE.md`, `docs/`, and `results/payoff.md`
   land in place:
   ```bash
   unzip ~/Downloads/swing-repertoire-resume-*.zip -d .
   ```
4. **Python env** (uv; deps pinned in `requirements.txt`)
   ```bash
   uv venv .venv --python 3.13
   source .venv/bin/activate               # macOS/Linux (Windows Git Bash: source .venv/Scripts/activate)
   uv pip install -r requirements.txt
   python -m ipykernel install --user --name driveline --display-name "Python (driveline)"
   ```
5. **Verify** (env active, from repo root)
   ```bash
   python src/adjustability.py             # -> data/adjustability.parquet
   python src/payoff.py                    # -> results/payoff.md
   ```
   then open `src/adjustability_results.ipynb` on the **Python (driveline)** kernel and Run All.

**Fix these machine-specific paths on the Mac:** `.vscode/settings.json`
(`python.defaultInterpreterPath` → your Mac venv, e.g. `${workspaceFolder}/.venv/bin/python`) and the
hardcoded `Rscript` path in the notebooks + `src/leaderboard_table.R` — on macOS R lives at
`/usr/local/bin/Rscript` (Intel) or `/opt/homebrew/bin/Rscript` (Apple Silicon), or just put it on
`PATH`. **R (leaderboards only, optional):** R 4.6+ with `arrow, dplyr, gt, gtExtras, mlbplotR, scales,
webshot2`. **DB access is only needed to re-pull raw data** (`extract.py`) — the bundle already contains
every extract (including the raw DB load), so analysis resumes without the VPN or `BIOMECH_DB_*` creds.
