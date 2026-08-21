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

from .reference import hull_reference, load_reference

__all__ = ["Run", "ittc57_friction_coefficient", "fresh_water_density",
           "fresh_water_viscosity", "hull_runs", "sysser01_runs", "sysser50_runs"]

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


def hull_runs(sysser: int, length: float | None = None, wetted_area: float | None = None,
              volume: float | None = None) -> list[Run]:
    """The bare-hull upright runs of one Sysser hull, reduced.

    ``length``, ``wetted_area`` and ``volume`` default to the official hydrostatics, so the
    reduction uses the same geometry the series used rather than anything this code computed.

    The sinkage sign is the one discussed at length in each hull's measurements data file:
    the workbook's z is taken as negative-means-deeper, which is the classical squat
    behaviour, and which contradicts the release Info sheet's "positive down". The conflict
    is recorded there rather than resolved here.
    """
    key = f"sysser{sysser:02d}"
    data = load_reference(f"{key}_measurements.toml")[key]
    cond = data["conditions"]
    rows = data["bare_hull_upright"]
    hydro = hull_reference(sysser)
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


def trim_pivot_x(sysser: int, waterline_midpoint: float) -> float:
    """Longitudinal position of the trim pivot, in the geometry's own x coordinate.

    The release's Info sheet defines the measured sinkage as the vertical displacement of
    the centre of gravity and the trim as the pitch of the hull, so the rigid motion is a
    rotation about the centre of gravity followed by a heave of that point. The pivot is
    therefore the centre of gravity, and a model ballasted to float upright at zero trim
    carries it over the longitudinal centre of buoyancy, so its x is ``lcb`` measured from
    the midpoint of the upright waterline.

    That is confirmed by the data and not only argued: on the nine series-1 hulls the
    workbook's transducer offsets are asymmetric, and their midpoint reproduces -``lcb`` to
    five decimal places on seven of them and to 0.16 mm on the other two. From hull 10
    onwards the offsets are entered as round nominal values and carry no information.

    ``waterline_midpoint`` converts the datum: ``lcb`` is measured from the midpoint of the
    upright waterline, while meshes are built in the geometry's own coordinates.

    Only the pivot's x matters. Displacing the pivot vertically by dz changes the motion by
    a surge of dz sin(trim), which no wave resistance depends on, and a heave of
    dz (1 - cos(trim)), which is 0.19 mm for dz = 0.1 m at the largest trim in either
    hull's table.
    """
    return waterline_midpoint + hull_reference(sysser)["lcb"]


def sysser01_runs(length: float | None = None, wetted_area: float | None = None,
                  volume: float | None = None) -> list[Run]:
    """The Sysser 01 bare-hull upright runs, reduced."""
    return hull_runs(1, length, wetted_area, volume)


def sysser50_runs(length: float | None = None, wetted_area: float | None = None,
                  volume: float | None = None) -> list[Run]:
    """The Sysser 50 bare-hull upright runs, reduced."""
    return hull_runs(50, length, wetted_area, volume)
