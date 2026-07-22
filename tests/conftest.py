import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geoverdict import config as cfg  # noqa: E402


@pytest.fixture()
def rng():
    return np.random.default_rng(cfg.SEED)


@pytest.fixture()
def clean_plot():
    """A valid ~28 ha plot inside the AOI (a slightly irregular pentagon)."""
    lon, lat = -55.40, -7.20
    d = 0.0025  # ~275 m
    return Polygon([
        (lon - d, lat - d), (lon + d, lat - d), (lon + 1.4 * d, lat + 0.4 * d),
        (lon, lat + 1.3 * d), (lon - 1.2 * d, lat + 0.3 * d),
    ])


@pytest.fixture()
def portfolio(rng):
    from geoverdict import corrupt

    return corrupt.generate_portfolio(40, cfg.AOI_BBOX, seed=cfg.SEED)
