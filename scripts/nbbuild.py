"""Notebook builder helpers: every chapter is generated from a script.

WHY GENERATE NOTEBOOKS FROM SCRIPTS. Notebook JSON does not diff, does not
review, and drifts. Keeping each chapter as a build script means the
narrative and the code live in version control as readable text, and the
.ipynb is a build artefact — regenerate any time with
`python scripts/nb01_geometry.py` (etc.).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

# GitHub coordinates for the "Open in Colab" badge. If you fork, change these
# and rebuild the notebooks (python scripts/nb01_geometry.py ...).
GITHUB_USER = "SaadH-077"
GITHUB_REPO = "geoverdict"
GITHUB_BRANCH = "main"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def colab_badge(filename: str) -> nbf.NotebookNode:
    """A clickable 'Open in Colab' badge cell, rendered by GitHub's viewer."""
    url = (f"https://colab.research.google.com/github/{GITHUB_USER}/{GITHUB_REPO}"
           f"/blob/{GITHUB_BRANCH}/notebooks/{filename}")
    return md(f'<a href="{url}" target="_parent">'
              f'<img src="https://colab.research.google.com/assets/colab-badge.svg" '
              f'alt="Open In Colab"/></a>')


def save(cells: list, filename: str) -> Path:
    nb = nbf.v4.new_notebook()
    # every notebook opens with its Colab badge, so it can be launched with one
    # click straight from the GitHub file view
    nb.cells = [colab_badge(filename)] + cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    }
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / filename
    nbf.write(nb, str(path))
    print("wrote", path)
    return path


BOOTSTRAP_MD = """
### Environment setup and persistence

On Colab this clones the repository, installs dependencies, and mounts Google
Drive so that **outputs survive the session**. Locally it is a no-op beyond
putting `src/` on the path.

**Why Drive.** A Colab VM is deleted when the session ends, and the notebooks
depend on each other's artefacts: 01 writes the validated plot portfolio that
every later chapter loads; 02 writes the forest baselines; 03 the time series;
04 the model predictions. `outputs/`, `figures/` and `evidence/` are therefore
redirected to Drive via environment variables that `geoverdict.config` reads
at import time — which is why they must be set *before* the import.

**Re-running this cell picks up code changes**: it hard-resets the clone to
`origin/main` and purges `geoverdict` from `sys.modules` (Python caches
imports; a `git pull` alone leaves the kernel running the old code). Treat the
clone as read-only — edit code locally and push, not inside `/content`.
"""

BOOTSTRAP_CODE = '''
# --- edit these if you are running your own fork ---------------------------
GITHUB_USER = "SaadH-077"
USE_DRIVE = True          # False -> everything stays in the ephemeral session
# ---------------------------------------------------------------------------

import os, subprocess, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO = "geoverdict"

if IN_COLAB:
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        PERSIST = Path("/content/drive/MyDrive/geoverdict")
        for sub in ("outputs", "figures", "outputs/evidence"):
            (PERSIST / sub).mkdir(parents=True, exist_ok=True)
        os.environ["GEOVERDICT_OUTPUT_DIR"] = str(PERSIST / "outputs")
        os.environ["GEOVERDICT_FIGURE_DIR"] = str(PERSIST / "figures")
        os.environ["GEOVERDICT_EVIDENCE_DIR"] = str(PERSIST / "outputs" / "evidence")
        print("persisting outputs and figures to", PERSIST)

    if not Path(REPO).exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/{GITHUB_USER}/{REPO}.git"], check=False)
    if Path(REPO).exists():
        os.chdir(REPO)
        subprocess.run(["git", "fetch", "--quiet", "--depth", "50", "origin", "main"], check=False)
        before = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        subprocess.run(["git", "reset", "--hard", "--quiet", "origin/main"], check=False)
        after = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if before != after:
            print(f"repo updated {before[:7]} -> {after[:7]}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError as exc:
        print("!! dependency install failed:", exc)
        print("!! continuing anyway - the cells below will report what is missing")

ROOT = Path.cwd() if Path.cwd().name == REPO or (Path.cwd() / "src").exists() else Path.cwd().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Purge cached geoverdict modules so a repo update takes effect in THIS kernel.
for m in [m for m in list(sys.modules) if m == "geoverdict" or m.startswith("geoverdict.")]:
    del sys.modules[m]

from geoverdict import config as cfg
from geoverdict import viz

cfg.ensure_dirs()
viz.set_style()
print(f"outputs -> {cfg.OUTPUT_DIR}")
print(f"figures -> {cfg.FIGURE_DIR}")
print(f"seed = {cfg.SEED}, AOI = {cfg.AOI_NAME} {cfg.AOI_BBOX}")
'''


def bootstrap_cells() -> list:
    return [md(BOOTSTRAP_MD), code(BOOTSTRAP_CODE)]
