"""Verification of the Kelvin Green function (V4, V5, V6 of the plan).

The chain being checked is: the closed form for the wavenumber integral, the identity that
removes a quadrature level, the reduction to a single integral over wave angle, and the
resulting Green function's defining properties.  Each link has an independent check.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import exp1

from wave_resistance import greens
from wave_resistance.greens import (
    RADIATION_SIGN, _DAMPING_REACH, _normalised_grid, _peak_position, _peak_refined,
    _phase_edges, _theta_of_xi_2d, MAX_CYCLES, calibrate_radiation_sign,
    envelope_ok, green,
    oscillation_count, p_function, q_function, rankine_part, wave_part,
    wave_part_gradient,
)

#: g_w at three points, from the raw double integral with Rayleigh damping evaluated by
#: adaptive quadrature over the non-symmetrised half range.  These are the anchor values:
#: everything else in this module is consistency, but these tie the reduction to the
#: definition.
_REFERENCE_GW = [
    (2.0, 0.8, -0.9, -0.054073),
    (-3.0, 0.5, -1.2, -0.012252),
    (0.5, -1.5, -0.7, -0.088094),
]


def _pv_brute(c: complex) -> complex:
    """PV integral over p in [0, inf) of exp(-p c)/(p - 1) dp, by singularity subtraction.

    The PV of 1/(p-1) over [0, 2] is zero, so subtracting f(1) there removes the pole and
    leaves a regular integrand.
    """
    f1 = np.exp(-c)

    def regular(p):
        d = p - 1.0
        return -c * f1 if abs(d) < 1e-13 else (np.exp(-p * c) - f1) / d

    opts = dict(limit=3000, epsabs=1e-14, epsrel=1e-13)
    re = (quad(lambda p: regular(p).real, 0.0, 1.0, **opts)[0]
          + quad(lambda p: regular(p).real, 1.0, 2.0, **opts)[0]
          + quad(lambda p: (np.exp(-p * c) / (p - 1.0)).real, 2.0, np.inf, **opts)[0])
    im = (quad(lambda p: regular(p).imag, 0.0, 1.0, **opts)[0]
          + quad(lambda p: regular(p).imag, 1.0, 2.0, **opts)[0]
          + quad(lambda p: (np.exp(-p * c) / (p - 1.0)).imag, 2.0, np.inf, **opts)[0])
    return re + 1j * im


@pytest.mark.filterwarnings("ignore:The occurrence of roundoff error")
@pytest.mark.filterwarnings("ignore:The maximum number of subdivisions")
@pytest.mark.parametrize("c", [
    0.3 - 0.7j, 0.25 + 3.5j, 1.0 - 0.05j, 1.6 + 0.8j, 0.15 - 11.0j,
    29.0 - 2.0j, 0.01 - 0.02j, 2.0 + 2.0j,
])
def test_q_function_matches_direct_principal_value(c):
    assert q_function(np.array([c]))[0] == pytest.approx(_pv_brute(c), rel=1e-9)


@pytest.mark.parametrize("c", [
    0.3 - 0.7j, 1.0 - 0.05j, 3.0 + 300.0j, 1.0 + 3000.0j, 0.05 - 900.0j, 12.0 - 45.0j,
])
def test_q_function_identity(c):
    """Q(c) = exp(-c) [E_1(-c) - i pi sgn(Im c)] is what removes a quadrature level."""
    closed = np.exp(-c) * (exp1(-c) - 1j * np.pi * np.sign(c.imag))
    assert q_function(np.array([c]))[0] == pytest.approx(closed, rel=1e-9)


def test_p_function_branches_agree_in_their_overlap():
    """The asymptotic and exp1 branches must meet at the cutoff."""
    phase = np.linspace(-1.5, 1.5, 121)
    for mag in (35.0, 60.0, 200.0):
        c = mag * np.exp(1j * phase)
        exact = np.exp(-c) * exp1(-c)
        assert np.abs(p_function(c) - exact).max() / np.abs(exact).max() < 1e-10


def test_asymptotic_series_covers_its_whole_input():
    """A gap in the magnitude bands once left values in uninitialised memory."""
    c = np.array([29.999999999999996, 30.0, 1e-3, 5.0, 1e9]) * (1.0 - 0.3j)
    assert np.isfinite(greens._asymptotic(c)).all()


@pytest.mark.parametrize("X,Y,Z,expected", _REFERENCE_GW)
def test_wave_part_matches_the_raw_double_integral(X, Y, Z, expected):
    got = wave_part(np.array([X]), np.array([Y]), np.array([Z]))[0]
    assert got == pytest.approx(expected, abs=2e-6)


def test_wave_part_is_even_in_the_transverse_offset():
    """G depends on |y - eta|, so g_w must be even in Y."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-8.0, 8.0, 25)
    Y = rng.uniform(0.2, 5.0, 25)
    Z = -rng.uniform(0.25, 2.0, 25)
    a, b = wave_part(X, Y, Z), wave_part(X, -Y, Z)
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-13


def test_wave_part_rejects_a_source_on_the_free_surface():
    with pytest.raises(ValueError, match="strictly negative"):
        wave_part(np.array([1.0]), np.array([0.0]), np.array([0.0]))


def test_wave_part_converges_under_refinement():
    rng = np.random.default_rng(11)
    X = rng.uniform(-12.0, 12.0, 60)
    Y = rng.uniform(-4.0, 4.0, 60)
    Z = -10.0 ** rng.uniform(np.log10(0.2), np.log10(2.0), 60)
    base = wave_part(X, Y, Z)
    fine = wave_part(X, Y, Z, refine=3.0)
    assert np.abs(base - fine).max() < 1e-7


def test_batch_and_single_point_evaluation_agree():
    """Points share a grid within a bin; a mis-scaled grid once cost 68 per cent."""
    rng = np.random.default_rng(5)
    X = rng.uniform(-15.0, 15.0, 40)
    Y = rng.uniform(-5.0, 5.0, 40)
    Z = -10.0 ** rng.uniform(np.log10(0.15), np.log10(2.0), 40)
    batch = wave_part(X, Y, Z)
    solo = np.array([wave_part(X[i:i + 1], Y[i:i + 1], Z[i:i + 1])[0] for i in range(40)])
    assert np.abs(batch - solo).max() < 1e-8


def test_radiation_condition_puts_the_waves_downstream():
    """Upstream is x -> +inf here.  The wave train must trail, not lead."""
    for lo, hi in ((15.0, 20.0), (60.0, 65.0)):
        xs = np.linspace(lo, hi, 40)
        zeros, zs = np.zeros_like(xs), np.full_like(xs, -0.8)
        upstream = np.ptp(wave_part(xs, zeros, zs))
        downstream = np.ptp(wave_part(-xs, zeros, zs))
        assert upstream < 0.05 * downstream


def test_upstream_field_decays_faster_than_the_downstream_wave():
    """Downstream the Kelvin waves fall off as x^(-1/2) along the track; upstream there
    are no waves at all and the local field decays algebraically."""
    def spread(x0, x1, negate):
        xs = np.linspace(x0, x1, 50)
        s = -1.0 if negate else 1.0
        return np.ptp(wave_part(s * xs, np.zeros_like(xs), np.full_like(xs, -0.8)))

    up_near, up_far = spread(15.0, 20.0, False), spread(120.0, 125.0, False)
    dn_near, dn_far = spread(15.0, 20.0, True), spread(120.0, 125.0, True)
    assert up_far / up_near < 0.1                      # observed about 0.022
    assert dn_far / dn_near > 0.25                     # observed about 0.39


def test_calibration_recovers_the_pinned_radiation_sign():
    assert calibrate_radiation_sign() == RADIATION_SIGN


def test_deep_source_tends_to_the_rigid_wall_limit():
    """As k0 |z + zeta| -> inf the free-surface condition u^2 phi_xx + g phi_z = 0 becomes
    phi_z = 0, so G tends to the rigid-wall image pair and k0 g_w -> -1/(2 pi R1).

    The often-quoted "large separation tends to Rankine" is not the right limit: the
    Kelvin waves decay as x^(-1/2) along the track while the Rankine dipole decays as
    x^(-2), so at large separation the wave part dominates instead of vanishing.
    """
    X, Y = 1.5, 0.7
    ratios = []
    for Z in (-20.0, -50.0, -120.0):
        gw = wave_part(np.array([X]), np.array([Y]), np.array([Z]))[0]
        ratios.append(gw / (-1.0 / (2.0 * np.pi * np.sqrt(X * X + Y * Y + Z * Z))))
    assert ratios[-1] == pytest.approx(1.0, abs=0.01)
    assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)


def test_rankine_part_vanishes_on_the_free_surface():
    q = np.array([0.2, -0.1, -0.3])
    p = np.array([[0.7, 0.4, 0.0], [-1.1, 0.0, 0.0]])
    assert np.abs(rankine_part(p, q)).max() < 1e-15


def test_green_function_is_harmonic():
    """The finite-difference Laplacian must fall as h^2, i.e. converge to zero."""
    k0, q = 6.946, np.array([0.3, -0.10, -0.06])
    p0 = np.array([0.62, 0.05, -0.11])
    residuals = []
    for h in (2e-3, 1e-3, 5e-4):
        lap = 0.0
        for axis in range(3):
            e = np.zeros(3)
            e[axis] = h
            lap += (green(p0 + e, q, k0) - 2 * green(p0, q, k0) + green(p0 - e, q, k0)) / h ** 2
        residuals.append(abs(float(lap)))
    ratios = [residuals[i] / residuals[i + 1] for i in range(2)]
    assert all(r > 3.0 for r in ratios), f"not second order: {residuals}"


@pytest.mark.parametrize("px,py", [(0.9, 0.02), (1.4, 0.35)])
def test_green_function_satisfies_the_linearised_free_surface_condition(px, py):
    """G_xx + k0 G_z = 0 on z = 0, which is u^2 G_xx + g G_z = 0 divided by u^2."""
    k0, q, h = 6.946, np.array([0.3, -0.10, -0.06]), 1e-3
    p0 = np.array([px, py, 0.0])
    ex, ez = np.array([h, 0.0, 0.0]), np.array([0.0, 0.0, h])
    gxx = (green(p0 + ex, q, k0) - 2 * green(p0, q, k0) + green(p0 - ex, q, k0)) / h ** 2
    gz = (green(p0 + ez, q, k0) - green(p0 - ez, q, k0)) / (2 * h)
    assert abs(float(gxx + k0 * gz)) < 1e-4 * max(abs(float(gxx)), 1.0)


def test_the_gradient_is_as_accurate_as_the_value_near_the_free_surface():
    """The regression test for the errors of ``docs/formulation.md`` section 15.

    Section 9 measured g_w and set the mesh constraint from it.  The influence matrix uses
    the *gradient*, whose integrand carries an extra sec^2(theta), and at
    (X, Y, |Z|) = (2, 0.8, 0.02) the production grid returned g_w to 1.7e-3 and its
    gradient 185 per cent wrong.  Both are now checked, and the gradient is checked first.
    """
    for X, Y, absZ in ((2.0, 0.8, 0.02), (0.5, 2.0, 0.01), (0.5, 6.0, 0.02),
                       (0.2, 6.0, 0.02), (3.5, 3.5, 0.012), (2.0, 2.0, 0.02)):
        ref = wave_part_gradient(X, Y, -absZ, refine=30.0)
        got = wave_part_gradient(X, Y, -absZ)
        err = np.abs(got - ref).max() / np.abs(ref).max()
        assert err < 3e-3, f"gradient at ({X}, {Y}, {-absZ}) off by {err:.2e}"
        rv = wave_part(X, Y, -absZ, refine=30.0)
        gv = wave_part(X, Y, -absZ)
        assert abs(gv - rv) < 1e-4 * max(abs(rv), 1e-12)


def test_the_gradient_is_exact_along_the_centreplane_offset():
    """Y = 0 puts the integrand's peak at theta = +/- pi/2, outside the range.

    This is the asymmetry that identified the peak as the mechanism: before the local
    refinement the grid was exact to machine precision along Y = 0 at every |Z| and
    grossly wrong off it.  Kept as a test because it isolates the peak from everything
    else in the quadrature.
    """
    for absZ in (0.2, 0.05, 0.01, 0.002):
        ref = wave_part_gradient(2.0, 0.0, -absZ, refine=30.0)
        got = wave_part_gradient(2.0, 0.0, -absZ)
        assert np.abs(got - ref).max() < 1e-9 * np.abs(ref).max()
        # g_w is even in Y, so d/dY vanishes on Y = 0 and only round-off survives.
        assert abs(got[1]) < 1e-10 * np.abs(got).max()


def test_peak_partition_tiles_the_panel_it_replaces():
    """The correction subtracts a coarse panel and adds a refined one, so the two rules
    must cover exactly the same interval or the difference is a bias, not a refinement."""
    _, _, edges = _normalised_grid(120, 12, 16, 300.0, 8.0)
    xi_star = np.array([-0.0559, 0.0, 0.4, 1.3, -1.9])
    _, coarse_w, _, fine_w = _peak_refined(edges, xi_star, 16)
    assert coarse_w.sum(axis=0) == pytest.approx(fine_w.sum(axis=0), rel=1e-14)


def test_phase_edges_carry_equal_phase():
    """Equation (15.1): every panel must carry the same number of cycles."""
    for cq, cl in ((300.0, 8.0), (0.0, 12.0), (1900.0, 3.6), (5.0, 0.0)):
        xi = _phase_edges(40, cq, cl)
        assert xi[0] == 0.0 and xi[-1] == pytest.approx(1.0)
        assert np.all(np.diff(xi) > 0.0)
        phase = cq * xi ** 2 + cl * xi
        per_panel = np.diff(phase)
        assert per_panel.max() == pytest.approx(per_panel.min(), rel=1e-10)


def test_peak_position_lands_on_the_vanishing_imaginary_part():
    """xi* must be where Im c = 0, i.e. tan(theta) = -X/Y, in normalised coordinates."""
    X, Y, Z = 2.0, 0.8, -0.02
    t_max = np.array([np.sqrt(_DAMPING_REACH / abs(Z) - 1.0)])
    xi_star = _peak_position(np.array([X]), np.array([Y]), t_max)
    theta, _ = _theta_of_xi_2d(xi_star[None, :], t_max)
    sec2 = 1.0 / np.cos(theta) ** 2
    imag_c = -sec2 * (X * np.cos(theta) + Y * np.sin(theta))
    assert np.abs(imag_c).max() < 1e-12
    # And that angle is where |c| is least over the range, which is what makes it a peak.
    grid, _, _ = _normalised_grid(200, 12, 16, 300.0, 8.0)
    th, _ = _theta_of_xi_2d(grid[:, None], t_max)
    s2 = 1.0 / np.cos(th) ** 2
    mag = np.abs(s2 * (abs(Z) - 1j * (X * np.cos(th) + Y * np.sin(th))))
    at_star = np.abs(sec2 * (abs(Z) - 1j * (X * np.cos(theta) + Y * np.sin(theta)))).max()
    assert at_star <= mag.min() * 1.001


def test_envelope_flags_the_pairs_the_quadrature_cannot_serve():
    """Outside the envelope the grid hits its panel cap and coarsens without saying so, so
    the count is exposed and callers check it."""
    assert oscillation_count(0.0, 0.0, -1.0) < oscillation_count(0.0, 3.0, -0.01)
    assert envelope_ok(11.1, 3.5, -0.0139)          # a Sysser mesh at Fn = 0.30
    assert not envelope_ok(8.0, 6.0, -1e-5)         # both points on the free surface
    # Sysser 01 at Fn = 0.30 with its topmost centroid 1 mm below the surface, at the
    # production damping reach of 25.  Well inside the 24 000 cap; the section-9 constraint
    # this replaced would have demanded a 7 mm top row instead.
    assert oscillation_count(11.1, 3.5, -0.0139) == pytest.approx(1077.0, rel=0.02)
    assert oscillation_count(11.1, 3.5, -0.0139) < MAX_CYCLES
