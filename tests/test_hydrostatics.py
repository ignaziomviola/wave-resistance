"""V7 (analytic bodies), V8 (independent volume routes), V9 (mesh convergence)."""

import numpy as np
import pytest

from _shapes import box_exact, box_mesh
from wave_resistance.hull import Hull
from wave_resistance.hydrostatics import hydrostatics, solve_reference_heave
from wave_resistance.wigley import wigley_exact_hydrostatics, wigley_mesh


def test_box_hydrostatics_are_exact():
    length, width, depth = 2.4, 0.6, 0.35
    r = hydrostatics(box_mesh(length, width, depth, 0.3, n=8), n_sections=40)
    e = box_exact(length, width, depth)
    for key in ("volume", "waterplane_area", "midship_area", "wetted_area",
                "lwl", "bwl", "draught", "cb", "cp", "cm", "cwp"):
        assert getattr(r, key) == pytest.approx(e[key], rel=1e-12), key
    assert r.lcb == pytest.approx(0.0, abs=1e-12)
    assert r.vcb == pytest.approx(e["vcb"], rel=1e-12)
    assert r.tcb == pytest.approx(0.0, abs=1e-12)


def test_all_four_volume_routes_agree_for_a_box():
    r = hydrostatics(box_mesh(1.0, 0.4, 0.2, 0.2, n=6), n_sections=20)
    assert r.volume_spread / r.volume < 1e-13
    assert np.allclose(r.volume_routes, r.volume, rtol=1e-13)


def test_wigley_hydrostatic_coefficients_converge_to_the_exact_values():
    length, beam, draught = 4.0, 0.4, 0.25
    e = wigley_exact_hydrostatics(length, beam, draught)
    r = hydrostatics(wigley_mesh(length, beam, draught, n_x=400, n_z=200), n_sections=200)
    assert r.volume == pytest.approx(e["volume"], rel=2e-5)
    assert r.waterplane_area == pytest.approx(e["waterplane_area"], rel=2e-5)
    assert r.midship_area == pytest.approx(e["midship_area"], rel=2e-4)
    assert r.cb == pytest.approx(4.0 / 9.0, rel=2e-4)
    assert r.cwp == pytest.approx(2.0 / 3.0, rel=2e-4)
    assert r.cm == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert r.cp == pytest.approx(2.0 / 3.0, rel=1e-3)
    # LCB vanishes only to O(h^2): the quad-splitting diagonal is not fore-aft
    # symmetric, so the piecewise-linear surface is very slightly asymmetric even
    # though the hull is not.
    assert r.lcb == pytest.approx(0.0, abs=1e-5 * length)
    assert r.vcb == pytest.approx(e["vcb"], rel=2e-4)


def test_wigley_volume_converges_at_second_order():
    length, beam, draught = 1.0, 0.1, 0.0625
    exact = wigley_exact_hydrostatics(length, beam, draught)["volume"]
    errors = []
    for n in (50, 100, 200):
        r = hydrostatics(wigley_mesh(length, beam, draught, n_x=2 * n, n_z=n), n_sections=60)
        errors.append(abs(r.volume / exact - 1.0))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(rt > 3.0 for rt in ratios), f"expected ~4x per halving, got {ratios}"


def test_volume_routes_agree_on_the_real_hull(sysser01_hull):
    hull = sysser01_hull.with_datum_shift(0.127078)
    r = hydrostatics(hull.mesh(32, 160), n_sections=80)
    assert r.volume_spread / r.volume < 1e-12
    assert r.tcb == pytest.approx(0.0, abs=1e-12)
    assert r.tcf == pytest.approx(0.0, abs=1e-12)


def test_reference_heave_hits_the_target_displacement(sysser01_hull):
    target = 0.0376136
    shift = solve_reference_heave(sysser01_hull, target, n_u=32, n_v=160)
    r = hydrostatics(sysser01_hull.with_datum_shift(shift).mesh(32, 160), n_sections=80)
    assert r.volume == pytest.approx(target, rel=1e-6)


def test_sysser01_hydrostatics_converge_at_second_order(sysser01_hull):
    """Successive mesh differences must shrink by roughly four per halving, which is what
    exact clipping of a second-order-accurate tessellation should give."""
    hull = sysser01_hull.with_datum_shift(0.127078)
    levels = [hydrostatics(hull.mesh(n, 5 * n), n_sections=120) for n in (24, 48, 96)]
    for attr in ("volume", "wetted_area", "waterplane_area", "lcb"):
        v = [getattr(r, attr) for r in levels]
        d1, d2 = abs(v[1] - v[0]), abs(v[2] - v[1])
        assert d1 / d2 > 2.5, f"{attr} not second order: {v}, ratio {d1 / d2:.2f}"
        assert d2 / abs(v[2]) < 5e-4, f"{attr} not yet converged: {v}"


def test_sysser01_hydrostatics_regression_pin(sysser01_hull):
    """Pins the reference-condition figures so a geometry regression cannot pass quietly.

    Values are the 48 x 240 mesh at the displacement-matched datum; Richardson
    extrapolation of the 24/48/96/192 sequence gives V = 0.0376265 m^3,
    Sc = 0.6561380 m^2, Aw = 0.5584200 m^2.
    """
    r = hydrostatics(sysser01_hull.with_datum_shift(0.127078).mesh(48, 240), n_sections=240)
    assert r.volume == pytest.approx(0.037613416, rel=1e-7)
    assert r.wetted_area == pytest.approx(0.656010546, rel=1e-7)
    assert r.waterplane_area == pytest.approx(0.558267803, rel=1e-7)
    assert r.lwl == pytest.approx(1.608142, rel=1e-5)
    assert r.bwl == pytest.approx(0.512814, rel=1e-5)
    assert r.draught == pytest.approx(0.127113, rel=1e-5)
    assert r.cp == pytest.approx(0.562297, rel=1e-4)
    assert r.lcb_from_midship == pytest.approx(-0.034338, rel=1e-3)


def test_hull_entirely_above_the_free_surface_is_rejected(sysser01_hull):
    with pytest.raises(ValueError, match="entirely above"):
        hydrostatics(sysser01_hull.with_datum_shift(-1.0).mesh(8, 40))
