# Template Python Project
*brief description of purpose & functionality*
A boilerplate TEMPLATE for Python projects

### Usage
```bash
python main.py <args>
```

## Details
- **Project Owner:** *name*
- **Project's Notion Page:** [https://notion.so](https://notion.so)
- **Project's Slack Channel:** `#proj-project-name`

## Contributors
* [@author](Author)

## Getting Started

To get started, **create a new repository from this template** and **clone the repository** to your local machine.

### Setup the environment
1. Open `environment.yml` and edit `name:` to match your project's name
2. Create the conda VENV for the project and install the required packages
```bash
conda env create -f environment.yml
```
### Setup teh project
3. Rename `.config.TEMPLATE` to `.config` and edit it so it works w/your local machine (edit directories/paths)
4. ~~Rename `.env.TEMPLATE` to `.env` and update it with your assigned credentials (LOCAL/DEV SETUP section)~~

### Setup the IDE
5. Edit `python.analysis.extraPaths` in `.vscode/settings.json` to point to where you cloned the `drivepy` repo

### Test-run
6. Activate the VENV in your terminal and test the project
```bash
conda activate project_name
python main.py test_argument
```

## ⚠️ IMPORTANT ⚠️
Please try your best to adhere to the following standards:
- Use `snake_case` for variable names
- Use `camelCase` for function names
  - The first word (lowercase) should be a verb that describes the action it takes. (e.g. `parseData()`)
- Define constants near the top of the file, below imports/setup, with `UPPER_CASE_LIKE_THIS`
- Store secrets, credentials, or confidential information in `.env`.
  - Use them in the script with `os.environ`.  (e.g. `os.environ['GITHUB_TOKEN']`)
- Store any configuration variables (information that changes from machine to machine) in `.config`
  - Reference them in the script with `cfg`. (e.g. `cfg.DRIVEPY_PATH`)
