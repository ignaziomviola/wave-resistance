"""Published reference data, loaded from editable data files with their provenance."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

__all__ = ["load_reference", "sysser01_reference"]


def load_reference(name: str) -> dict:
    """Load a TOML data file shipped alongside the package."""
    path = resources.files("wave_resistance").joinpath("data", name)
    with resources.as_file(path) as p:
        return tomllib.loads(Path(p).read_text(encoding="utf-8"))


def sysser01_reference() -> dict:
    """Official model-scale hydrostatics for Sysser 01 at zero heel."""
    return load_reference("sysser01_reference.toml")["sysser01"]
