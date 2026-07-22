"""Central configuration: paths, the AOI, the cutoff date, seeds, constants.

WHY A SINGLE CONFIG MODULE. Every notebook and every library module reads the
same area of interest, the same cutoff date, the same band list and the same
directories from here. The alternative — each notebook re-declaring its own
bbox or date — is how two chapters silently analyse different areas and their
results stop being comparable. One definition, imported everywhere.

WHY ENVIRONMENT-VARIABLE OVERRIDES. On Colab the notebooks redirect outputs and
figures to Google Drive so they survive the session; locally they go to the
repository. The redirect happens by setting GEOVERDICT_OUTPUT_DIR /
GEOVERDICT_FIGURE_DIR / GEOVERDICT_DATA_DIR *before* importing the package,
which is why the bootstrap cell in every notebook sets them first.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("GEOVERDICT_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("GEOVERDICT_OUTPUT_DIR", ROOT / "outputs"))
FIGURE_DIR = Path(os.environ.get("GEOVERDICT_FIGURE_DIR", ROOT / "figures"))
EVIDENCE_DIR = Path(os.environ.get("GEOVERDICT_EVIDENCE_DIR", OUTPUT_DIR / "evidence"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, EVIDENCE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# The area of interest
# --------------------------------------------------------------------------
# Novo Progresso, Pará, Brazil — the BR-163 corridor in the "arc of
# deforestation". Chosen deliberately:
#   1. It is one of the most active deforestation frontiers on Earth, so a
#      portfolio of plots scattered here contains real post-2020 clearings,
#      real stable forest, and real pre-2020 pasture — all three verdict
#      classes occur naturally.
#   2. Cloud cover is seasonal (dry season ~June-September), so the notebooks
#      can show BOTH a clean dry-season signal and the wet-season observation
#      gaps that make tropical monitoring genuinely hard.
#   3. Cattle and soy from exactly this corridor are in scope of the EUDR, so
#      the compliance framing is not decorative.
# The box is ~44 x 44 km: big enough for landscape diversity, small enough
# that windowed COG reads keep every notebook inside a free Colab session.

AOI_NAME = "novo_progresso_para_brazil"
AOI_BBOX = (-55.60, -7.40, -55.20, -7.00)  # (min_lon, min_lat, max_lon, max_lat)

# --------------------------------------------------------------------------
# Regulatory anchors (EUDR, Regulation (EU) 2023/1115)
# --------------------------------------------------------------------------
# The cutoff: commodities are non-compliant if produced on land deforested
# after 31 December 2020. Every temporal split in this project is anchored
# here, and saying "my T1 epoch is the EUDR cutoff" is the sentence that
# separates a compliance pipeline from a change-detection demo.
CUTOFF_DATE = "2020-12-31"
CUTOFF_YEAR = 2020

# EUDR Art. 9: plots > 4 ha must be described by polygons; smaller plots may
# be a single point. The validator uses this to decide whether a Point
# geometry is a defect or a legitimate submission.
POINT_MAX_AREA_HA = 4.0

# Monitoring window: everything after the cutoff, up to "now".
MONITOR_START = "2021-01-01"
MONITOR_END = "2025-12-31"
# Pre-cutoff reference period used to establish each plot's spectral baseline.
BASELINE_START = "2019-01-01"

# --------------------------------------------------------------------------
# Plot portfolio parameters (notebook 01)
# --------------------------------------------------------------------------
N_PLOTS = 600            # portfolio size: large enough for stable rates,
                         # small enough to screen on free Colab
CORRUPTION_RATE = 0.45   # fraction of plots that arrive damaged — deliberately
                         # pessimistic so every failure class has enough samples
MIN_PLOT_HA = 0.5        # below this we treat the polygon as a digitising slip
MAX_PLOT_HA = 1000.0     # above this a "farm plot" is almost certainly a unit
                         # or digitising error (1,000 ha = 10 km^2)

# --------------------------------------------------------------------------
# Sentinel-2 bands used for chips (notebooks 03-04)
# --------------------------------------------------------------------------
# Six of the twelve L2A bands, chosen for physics rather than habit:
#   B02/B03/B04 (10 m visible)  — human-checkable context, soil brightness
#   B08 (10 m NIR)              — leaf structure; the numerator of NDVI
#   B11/B12 (20 m SWIR)         — moisture; THE forest-loss bands. A cleared
#                                 plot dries out, and SWIR reflectance jumps
#                                 long before the visible bands look different.
# Dropping the 60 m atmospheric bands and the red-edge trio keeps chips small
# (6 x 32 x 32) so hundreds of plots fit in Colab RAM and training takes
# minutes, at a cost we measure rather than assume (band ablation, NB06).
CHIP_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
CHIP_SIZE = 32           # 32 px at 10 m = 320 m — matches smallholder plot scale

# T1/T2 acquisition windows for the bi-temporal chips (notebook 04). Dry
# season (June-September) in the southern Amazon: the least-cloudy scene of a
# dry-season window is usually genuinely clear, while an annual minimum can
# still be 40% cloud. T1 sits at the cutoff epoch by design — "was it forest
# at the cutoff, and what does it look like now" IS the EUDR question.
T1_WINDOW = ("2020-06-01", "2020-09-30")
T2_WINDOW = ("2024-06-01", "2024-09-30")

# Weak labels for the learned arm, from Hansen post-2020 loss fraction:
# positives are unambiguous clearings, negatives unambiguous stability, and
# the ambiguous middle (2-20%) is EXCLUDED from training — teaching a model
# with labels you do not trust in the ambiguous band injects exactly the
# noise you cannot diagnose later. The excluded band is still screened at
# inference time; exclusion is a training decision, not a coverage hole.
POS_LOSS_FRAC = 0.20
NEG_LOSS_FRAC = 0.02

# Time-series screening subset (notebook 03): server-side GEE reductions are
# fast, but the getInfo transfer and the per-plot detector fits are not free;
# 200 plots give stable rates while keeping the notebook under ~15 minutes.
TS_SUBSET = 200

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 77
SEEDS = (77, 78, 79)     # multi-seed runs: a single-seed result is an anecdote


def set_seed(seed: int = SEED, deterministic: bool = True) -> int:
    """Seed python, numpy and (if present) torch. Returns the seed it set."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# --------------------------------------------------------------------------
# The results ledger
# --------------------------------------------------------------------------
# Every notebook appends its headline numbers to one JSON file. NB06 re-reads
# the ledger and checks the claims in the README against what was actually
# measured. Numbers that only exist inside a notebook cell die with the cell.

RESULTS_PATH_DEFAULT = OUTPUT_DIR / "results.json"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def append_result(entry: dict, path: Path | str | None = None) -> Path:
    """Append one result record (a flat dict with at least 'notebook' and 'name')."""
    path = Path(path or RESULTS_PATH_DEFAULT)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
    # Same notebook + name replaces the old record: re-running a cell must not
    # duplicate the ledger, or NB06 would verify a stale number.
    key = (entry.get("notebook"), entry.get("name"))
    records = [r for r in records if (r.get("notebook"), r.get("name")) != key]
    records.append(entry)
    path.write_text(json.dumps(records, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_results(path: Path | str | None = None) -> list[dict]:
    path = Path(path or RESULTS_PATH_DEFAULT)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Provenance:
    """A provenance record for one data artefact.

    In a compliance product, provenance IS the feature: an auditor asking
    "which satellite scenes produced this verdict?" must get an answer.
    Every artefact this pipeline writes carries one of these.
    """

    source: str          # e.g. "sentinel-2-l2a via Earth Search STAC"
    identifiers: list    # e.g. STAC item ids, GEE asset ids
    created: str         # ISO timestamp
    parameters: dict     # the knobs that produced it

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "identifiers": list(self.identifiers),
            "created": self.created,
            "parameters": self.parameters,
        }
