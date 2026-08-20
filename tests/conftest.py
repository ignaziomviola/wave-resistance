from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SYSSER01 = REPO / "data" / "SYSSER01_surface.igs"


@pytest.fixture(scope="session")
def sysser01_path() -> Path:
    if not SYSSER01.exists():                                   # pragma: no cover
        pytest.skip(f"{SYSSER01} not present")
    return SYSSER01


@pytest.fixture(scope="session")
def sysser01_iges(sysser01_path):
    from wave_resistance.iges import read_iges
    return read_iges(sysser01_path)


@pytest.fixture(scope="session")
def sysser01_hull(sysser01_path):
    from wave_resistance.hull import Hull
    return Hull.from_iges(sysser01_path)
