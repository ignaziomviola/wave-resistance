"""V2: the far-field amplitudes and the resistance constant.

The constant C = rho g^2/(pi u^4) is derived in docs/formulation.md section 4, not fitted,
and the derivation rests on the thin-ship density being sigma = u n_x rather than the
zeroth iterate sigma = 2 u n_x.

An earlier version of this file tested C by back-computing it from the Michell oracle and
dividing by the shipped constant.  That checked only that the closure and the constant were
mutually consistent, and a fourfold error in both survived it.  The tests below instead
impose the derived density and compare the *shipped* resistance against the oracle, which
is a statement that can fail.
"""

import numpy as np
import pytest

from wave_resistance.spectrum import (
    free_wave_amplitudes, panel_quadrature, resistance_constant, wave_resistance_integral,
)
from wave_resistance.wigley import (
    michell_resistance_from_integral, wigley_mesh, wigley_michell_amplitude,
)

GRAV = 9.80665
RHO = 1000.0


def _michell_integral(fn: float, beam_ratio: float, draught_ratio: float,
                      per_cycle: int = 32, lam_cap: float = 300.0) -> float:
    """Michell's I with panels locked to the bow-stern interference phase."""
    step = 2 * np.pi * fn ** 2 / per_cycle
    t_max = np.sqrt(lam_cap ** 2 - 1.0)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    total, t = 0.0, 0.0
    while t < t_max:
        span = step * np.sqrt(1 + t * t) / max(t, 1e-2) if t > 1e-2 else 2 * step
        span = min(span, 40 * step, t_max - t)
        if span <= 1e-14:
            break
        s = t + span * 0.5 * (1 + nodes)
        lam = np.sqrt(1 + s * s)
        amp = wigley_michell_amplitude(lam, fn, beam_ratio, draught_ratio)
        total += float(np.dot(weights * span * 0.5, lam * np.abs(amp) ** 2))
        t += span
    return total


def test_resistance_constant_formula():
    assert resistance_constant(1000.0, 9.80665, 2.0) == pytest.approx(
        1000.0 * 9.80665 ** 2 / (np.pi * 16.0))


def test_panel_quadrature_weights_sum_to_panel_area():
    mesh = wigley_mesh(1.0, 0.1, 0.0625, n_x=12, n_z=6)
    _, w = panel_quadrature(mesh, order=5)
    assert w.sum(axis=1) == pytest.approx(mesh.areas(), rel=1e-13)


def test_both_branches_coincide_for_a_symmetric_hull():
    """A_minus differs from A_plus only through the sign of the transverse phase, so for a
    hull symmetric about y = 0 the two are equal.  The solver keeps both regardless,
    because a heeled or asymmetric hull breaks this."""
    mesh = wigley_mesh(1.0, 0.1, 0.0625, n_x=16, n_z=8)
    pts, w = panel_quadrature(mesh, order=3)
    sigma = 2.0 * mesh.unit_normals()[:, 0]
    lam = np.array([1.0, 1.7, 4.2, 11.0])
    a_p, a_m = free_wave_amplitudes(lam, pts.reshape(-1, 3), (w * sigma[:, None]).reshape(-1),
                                    k0=11.11)
    assert np.abs(a_p - a_m).max() < 1e-12 * max(np.abs(a_p).max(), 1e-30)


def test_amplitude_of_an_asymmetric_arrangement_differs_between_branches():
    pts = np.array([[0.1, 0.3, -0.2], [-0.4, 0.1, -0.5]])
    w = np.array([1.0, 0.7])
    a_p, a_m = free_wave_amplitudes(np.array([3.0]), pts, w, k0=5.0)
    assert abs(a_p[0] - a_m[0]) > 0.1 * abs(a_p[0])


def test_spectrum_integral_certifies_its_tail():
    mesh = wigley_mesh(1.0, 0.05, 0.0625, n_x=32, n_z=12)
    pts, w = panel_quadrature(mesh, order=3)
    sigma = 2.0 * 0.94 * mesh.unit_normals()[:, 0]
    total, diag = wave_resistance_integral(
        pts.reshape(-1, 3), (w * sigma[:, None]).reshape(-1), k0=11.11, length=1.0,
        return_diagnostics=True)
    assert diag["converged"]
    assert total > 0.0
    assert diag["lambda_reached"] > 5.0


def test_spectrum_integral_is_stable_under_refinement():
    mesh = wigley_mesh(1.0, 0.05, 0.0625, n_x=32, n_z=12)
    pts, w = panel_quadrature(mesh, order=3)
    sigma = 2.0 * 0.94 * mesh.unit_normals()[:, 0]
    args = (pts.reshape(-1, 3), (w * sigma[:, None]).reshape(-1), 11.11, 1.0)
    coarse = wave_resistance_integral(*args, refine=1.0, rtol=1e-5)
    fine = wave_resistance_integral(*args, refine=2.0, rtol=1e-5)
    assert fine == pytest.approx(coarse, rel=2e-3)


def _michell_resistance(fn: float, beam_ratio: float, draught_ratio: float,
                        length: float = 1.0) -> float:
    return michell_resistance_from_integral(
        _michell_integral(fn, beam_ratio, draught_ratio), fn, length, RHO, GRAV)


@pytest.mark.parametrize("fn", [0.30, 0.40])
@pytest.mark.parametrize("beam_ratio", [0.02, 0.005])
def test_derived_density_reproduces_michell_through_the_shipped_constant(fn, beam_ratio):
    """Impose sigma = u n_x and compare R_w against the oracle.

    No back-solving: the resistance is formed with resistance_constant as the solver forms
    it, so a wrong constant fails here.  The residual is the panel discretisation of the
    amplitude integral, which falls under refinement; measured 0.84 % at 558 panels and
    0.12 % at 1630 for B/L = 0.02, Fn = 0.30.
    """
    length, draught_ratio = 1.0, 0.0625
    speed = fn * np.sqrt(GRAV * length)
    k0 = GRAV / speed ** 2
    mesh = wigley_mesh(length, beam_ratio, draught_ratio, n_x=34, n_z=12)
    sigma = speed * mesh.unit_normals()[:, 0]
    pts, w = panel_quadrature(mesh, order=4)
    integral = wave_resistance_integral(
        pts.reshape(-1, 3), (w * sigma[:, None]).reshape(-1), k0, length, rtol=1e-5)
    predicted = resistance_constant(RHO, GRAV, speed) * integral
    oracle = _michell_resistance(fn, beam_ratio, draught_ratio, length)
    assert predicted == pytest.approx(oracle, rel=6e-3)


def test_the_zeroth_iterate_is_not_the_thin_ship_density():
    """Guard against the error that produced the fourfold constant.

    sigma = 2 u n_x overstates the thin-ship density by two and the resistance by four, so
    a future change that quietly reinstates it must break a test rather than pass one.
    """
    fn, length, beam_ratio, draught_ratio = 0.30, 1.0, 0.02, 0.0625
    speed = fn * np.sqrt(GRAV * length)
    k0 = GRAV / speed ** 2
    mesh = wigley_mesh(length, beam_ratio, draught_ratio, n_x=34, n_z=12)
    pts, w = panel_quadrature(mesh, order=4)
    oracle = _michell_resistance(fn, beam_ratio, draught_ratio, length)
    const = resistance_constant(RHO, GRAV, speed)

    def resistance(scale):
        sigma = scale * speed * mesh.unit_normals()[:, 0]
        return const * wave_resistance_integral(
            pts.reshape(-1, 3), (w * sigma[:, None]).reshape(-1), k0, length, rtol=1e-5)

    assert resistance(1.0) == pytest.approx(oracle, rel=6e-3)
    assert resistance(2.0) / oracle == pytest.approx(4.0, rel=1e-2)


@pytest.mark.slow
def test_the_nk_solve_recovers_michell_as_the_hull_thins():
    """The solve, not just the normalisation.

    This is the only test in the file that exercises the Kelvin influence matrix.  The
    density is solved for, so the error is first order in panel size (formulation section
    14) and the tolerance is correspondingly loose; what it excludes is a factor of two or
    four, not a few per cent.
    """
    from wave_resistance.nk import influence_matrix

    fn, length, beam_ratio, draught_ratio = 0.30, 1.0, 0.02, 0.0625
    speed = fn * np.sqrt(GRAV * length)
    k0 = GRAV / speed ** 2
    oracle = _michell_resistance(fn, beam_ratio, draught_ratio, length)

    mesh = wigley_mesh(length, beam_ratio, draught_ratio, n_x=14, n_z=5)
    normal_x = mesh.unit_normals()[:, 0]
    sigma = np.linalg.solve(influence_matrix(mesh, k0, wave=True, order=1), speed * normal_x)

    carries_flux = np.abs(normal_x) > 0.02
    density_ratio = (sigma[carries_flux] / (speed * normal_x[carries_flux])).mean()
    assert density_ratio == pytest.approx(1.0, abs=0.15), (
        f"solved density is {density_ratio:.3f} u n_x, not the derived u n_x")

    pts, w = panel_quadrature(mesh, order=3)
    predicted = resistance_constant(RHO, GRAV, speed) * wave_resistance_integral(
        pts.reshape(-1, 3), (w * sigma[:, None]).reshape(-1), k0, length, rtol=3e-5)
    assert predicted == pytest.approx(oracle, rel=0.15)
