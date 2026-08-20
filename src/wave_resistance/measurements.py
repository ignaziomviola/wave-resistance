"""Delft towing-tank measurements, and the friction subtraction that reduces them.

The release gives total resistance, sinkage and trim per speed.  Residuary resistance is not
in it: it follows from an ITTC-57 friction line at form factor zero, which is what the
series' own reduction does.  That makes residuary resistance a *proxy* for wave resistance
and not a measurement of it -- it also contains nonlinear and viscous-form contributions --
and every comparison in this module is labelled accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .reference import load_reference

__all__ = ["Run", "ittc57_friction_coefficient", "fresh_water_density",
           "fresh_water_viscosity", "sysser01_runs"]

GRAVITY = 9.80665


def fresh_water_density(temperature_c: float) -> float:
    """Density of fresh water in kg/m^3, from the standard fifth-order fit."""
    t = temperature_c
    return (999.842594 + 6.793952e-2 * t - 9.09529e-3 * t ** 2 + 1.001685e-4 * t ** 3
            - 1.120083e-6 * t ** 4 + 6.536332e-9 * t ** 5)


def fresh_water_viscosity(temperature_c: float) -> float:
    """Kinematic viscosity of fresh water in m^2/s."""
    t = temperature_c
    return 1.7688e-6 / (1.0 + 0.03368 * t + 0.00021 * t ** 2)


def ittc57_friction_coefficient(reynolds: float) -> float:
    """The ITTC-57 line, C_F = 0.075 / (log10(Re) - 2)^2."""
    if reynolds <= 100.0:
        raise ValueError("the ITTC-57 line is not meaningful at this Reynolds number")
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


@dataclass(frozen=True)
class Run:
    """One measured point, with the friction subtraction applied.

    ``residuary_resistance`` carries the k = 0 assumption named in the module docstring.  It
    is reported as it comes out, including where it is negative: at the lowest speed of the
    Sysser 01 bare-hull runs the subtraction gives -0.008 N, which says something about the
    friction line at Re = 5.9e5 and is not an error to be clamped away.

    ``sinkage`` and ``trim`` are in the units :class:`~wave_resistance.hull.Attitude` takes,
    metres positive downward and **degrees** bow-down positive, so a run feeds it directly:
    ``Attitude(sinkage=run.sinkage, trim=run.trim)``.  Storing the trim in radians and
    letting the caller convert is exactly the trap that produced a 57-fold error in the
    first version of this comparison.
    """

    speed: float                 # m/s
    froude: float
    reynolds: float
    total_resistance: float      # N, measured
    friction_resistance: float   # N, ITTC-57 at k = 0
    residuary_resistance: float  # N, total minus friction
    sinkage: float               # m, positive downward
    trim: float                  # deg, bow-down positive -- the units Attitude takes

    @property
    def residuary_fraction_of_weight(self) -> float:
        return self._weight_fraction

    _weight_fraction: float = 0.0


def sysser01_runs(length: float | None = None, wetted_area: float | None = None,
                  volume: float | None = None) -> list[Run]:
    """The Sysser 01 bare-hull upright runs, reduced.

    ``length``, ``wetted_area`` and ``volume`` default to the official hydrostatics, so the
    reduction uses the same geometry the series used rather than anything this code computed.
    """
    data = load_reference("sysser01_measurements.toml")["sysser01"]
    cond = data["conditions"]
    rows = data["bare_hull_upright"]
    hydro = load_reference("sysser01_reference.toml")["sysser01"]
    length = hydro["lwl"] if length is None else length
    wetted_area = hydro["wetted_area"] if wetted_area is None else wetted_area
    volume = hydro["volume"] if volume is None else volume

    rho, nu = cond["density"], cond["viscosity"]
    weight = rho * GRAVITY * volume
    runs = []
    for speed, total, sinkage_mm, trim_deg in zip(
            rows["speed"], rows["total_resistance"], rows["sinkage_mm"], rows["trim_deg"]):
        reynolds = speed * length / nu
        friction = 0.5 * rho * speed ** 2 * wetted_area * ittc57_friction_coefficient(reynolds)
        residuary = total - friction
        runs.append(Run(
            speed=speed,
            froude=speed / math.sqrt(GRAVITY * length),
            reynolds=reynolds,
            total_resistance=total,
            friction_resistance=friction,
            residuary_resistance=residuary,
            sinkage=-sinkage_mm * 1e-3,
            trim=trim_deg,
            _weight_fraction=residuary / weight,
        ))
    return runs
