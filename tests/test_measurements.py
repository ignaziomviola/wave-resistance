"""The Delft measurement reduction, and that the shipped data matches the primary release.

The data files carry values transcribed from 4TU releases.  Nothing here can re-download
them, so the release's own numbers appear as literals and the test asserts the transcription.
That catches an edit to the data file, which is the failure mode that matters.
"""

import math

import pytest

from wave_resistance.measurements import (fresh_water_density, fresh_water_viscosity,
                                          ittc57_friction_coefficient, sysser01_runs)
from wave_resistance.reference import load_reference, sysser01_reference


def test_reference_matches_the_primary_release():
    """Row Sysser = 1 of DSYHS_hydrostatics_modelscale.xlsx, sheet 'Canoe body hydrostatics'."""
    r = sysser01_reference()
    assert r["lwl"] == 1.6
    assert r["bwl"] == 0.5072
    assert r["tc"] == 0.12704
    assert r["volume"] == 0.0376136
    assert r["lcb"] == -0.03664
    assert r["waterplane_area"] == 0.558566
    assert r["lcf"] == -0.05328
    assert r["midship_area"] == 0.0416512
    assert r["wetted_area"] == 0.642534
    assert r["cb"] == pytest.approx(0.36484230452674, abs=1e-14)
    assert r["cm"] == pytest.approx(0.6464095967559, abs=1e-13)
    assert r["cp"] == pytest.approx(0.56441350245688, abs=1e-14)
    assert r["cwp"] == pytest.approx(0.68829599143772, abs=1e-14)


def test_the_release_table_is_internally_consistent():
    """Its coefficients must follow from its own dimensions, which is what makes it a
    description of a real hull rather than a set of independently rounded numbers.

    The tolerance is 1e-6 rather than machine precision because the release's coefficients
    were computed from unrounded dimensions while the tabulated lwl, bwl and tc are rounded
    to six or seven figures.  Measured agreement: cb to 2.6e-9, cm to 1.7e-8, cwp to 6.6e-8,
    and cp to 8e-15 because it is formed from the tabulated cb and cm themselves.
    """
    r = sysser01_reference()
    assert r["volume"] / (r["lwl"] * r["bwl"] * r["tc"]) == pytest.approx(r["cb"], rel=1e-6)
    assert r["midship_area"] / (r["bwl"] * r["tc"]) == pytest.approx(r["cm"], rel=1e-6)
    assert r["cb"] / r["cm"] == pytest.approx(r["cp"], rel=1e-12)
    assert r["waterplane_area"] / (r["lwl"] * r["bwl"]) == pytest.approx(r["cwp"], rel=1e-6)


def test_the_heeled_columns_hold_displacement_constant():
    """volc10 = volc20 = volc30 = volc0 in the release, so the heeled hydrostatics were taken
    at constant displacement with sinkage and trim solved for.  A test that rotates a hull at
    fixed sinkage and expects the volume to be conserved is therefore wrong, and the project
    plan contained one."""
    r = sysser01_reference()
    for v in r["heeled"]["volume"]:
        assert v == r["volume"]


def test_ittc57_line():
    assert ittc57_friction_coefficient(1e6) == pytest.approx(0.075 / 16.0)
    assert ittc57_friction_coefficient(1e9) == pytest.approx(0.075 / 49.0)
    # Monotone decreasing in Reynolds number.
    values = [ittc57_friction_coefficient(re) for re in (1e5, 1e6, 1e7, 1e8)]
    assert values == sorted(values, reverse=True)
    with pytest.raises(ValueError):
        ittc57_friction_coefficient(10.0)


def test_water_properties_at_the_tank_temperature():
    cond = load_reference("sysser01_measurements.toml")["sysser01"]["conditions"]
    t = cond["temperature_c"]
    assert fresh_water_density(t) == pytest.approx(cond["density"], abs=0.01)
    assert fresh_water_viscosity(t) == pytest.approx(cond["viscosity"], rel=1e-3)


def test_measured_runs_reduce_to_the_expected_residuary():
    runs = sysser01_runs()
    assert len(runs) == 11
    froudes = [r.froude for r in runs]
    assert froudes == sorted(froudes)
    assert froudes[0] == pytest.approx(0.100, abs=0.001)
    assert froudes[-1] == pytest.approx(0.600, abs=0.001)

    by_fn = {round(r.froude, 2): r for r in runs}
    # Total resistance is read straight from the release.
    assert by_fn[0.30].total_resistance == 3.01934
    # Friction and residuary follow from the ITTC-57 line at k = 0.
    assert by_fn[0.30].friction_resistance == pytest.approx(1.885, abs=0.002)
    assert by_fn[0.30].residuary_resistance == pytest.approx(1.134, abs=0.002)
    assert by_fn[0.45].residuary_resistance == pytest.approx(15.503, abs=0.005)


def test_the_low_speed_residuary_is_negative_and_is_not_clamped():
    """The k = 0 subtraction gives -0.008 N at Fn = 0.10.  That is information about the
    friction line, not an error, and clamping it would hide the one place where the reduction
    visibly fails."""
    runs = sysser01_runs()
    assert runs[0].froude == pytest.approx(0.100, abs=0.001)
    assert runs[0].residuary_resistance < 0.0
    assert runs[0].residuary_resistance == pytest.approx(-0.0076, abs=0.001)


def test_residuary_at_fn_030_is_a_poor_validation_target():
    """Recorded as a test because it governs how the comparison must be read: at Fn = 0.30 the
    quantity being predicted is 0.3 per cent of displacement weight, so it magnifies every
    error; by Fn = 0.45 it is 4 per cent."""
    by_fn = {round(r.froude, 2): r for r in sysser01_runs()}
    assert by_fn[0.30].residuary_fraction_of_weight == pytest.approx(0.0031, abs=0.0002)
    assert by_fn[0.45].residuary_fraction_of_weight == pytest.approx(0.0421, abs=0.0005)
    assert (by_fn[0.45].residuary_fraction_of_weight
            > 10 * by_fn[0.30].residuary_fraction_of_weight)


def test_measured_attitude_is_in_the_units_attitude_takes():
    """Metres positive downward and degrees bow-down positive, so a run feeds Attitude
    directly.  Storing the trim in radians and letting the caller convert produced a
    57-fold error in the first version of this comparison, so the units are asserted."""
    from wave_resistance.hull import Attitude

    by_fn = {round(r.froude, 2): r for r in sysser01_runs()}
    assert by_fn[0.30].sinkage == pytest.approx(6.56515e-3, rel=1e-6)
    assert by_fn[0.45].trim == pytest.approx(-1.62161, rel=1e-6)
    assert by_fn[0.50].sinkage > by_fn[0.30].sinkage      # sinks further at speed

    run = by_fn[0.45]
    rot, _ = Attitude(sinkage=run.sinkage, trim=run.trim).matrix()
    # Bow-down positive about +y, so a negative trim lifts the bow: a point at +x rises.
    assert (rot @ [1.0, 0.0, 0.0])[2] > 0.0
    assert math.degrees(math.asin(rot[0, 2])) == pytest.approx(run.trim, rel=1e-9)
