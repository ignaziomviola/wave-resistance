"""Published reference data, loaded from editable data files with their provenance."""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

__all__ = ["load_reference", "hull_reference", "sysser01_reference", "sysser50_reference"]


def load_reference(name: str) -> dict:
    """Load a TOML data file shipped alongside the package."""
    path = resources.files("wave_resistance").joinpath("data", name)
    with resources.as_file(path) as p:
        return tomllib.loads(Path(p).read_text(encoding="utf-8"))


def hull_reference(sysser: int) -> dict:
    """Official model-scale hydrostatics for one Sysser hull at zero heel.

    Each hull's published values live in their own data file, so adding a hull is a data
    change and not a code change.
    """
    key = f"sysser{sysser:02d}"
    return load_reference(f"{key}_reference.toml")[key]


def sysser01_reference() -> dict:
    """Official model-scale hydrostatics for Sysser 01 at zero heel."""
    return hull_reference(1)


def sysser50_reference() -> dict:
    """Official model-scale hydrostatics for Sysser 50 at zero heel."""
    return hull_reference(50)
