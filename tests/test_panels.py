"""V3: constant-source panel influences, and the sphere in an unbounded uniform stream.

This is the decisive gate on the jump term and on the panel integration, because it uses
an exact analytic solution and does not involve the wave kernel at all.
"""

import numpy as np
import pytest

from _shapes import icosphere, uv_sphere
from wave_resistance.panels import (
    _triangle_rule, rankine_normal_velocity_matrix, solid_angle, solve_rankine,
    source_velocity, velocity_by_quadrature,
)

_UNIT_TRIANGLE = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


def test_solid_angle_of_a_closed_surface():
    """4 pi at any interior point, 0 outside, for an outward-oriented closed surface."""
    mesh = icosphere(1.0, subdivisions=3)
    tri = mesh.triangles()
    for interior in ([0.0, 0.0, 0.0], [0.5, 0.2, -0.1], [-0.7, 0.0, 0.3]):
        assert solid_angle(tri, np.array([interior])).sum() == pytest.approx(4 * np.pi, abs=1e-9)
    for exterior in ([3.0, 0.0, 0.0], [0.0, -8.0, 1.0]):
        assert abs(solid_angle(tri, np.array([exterior])).sum()) < 1e-9


def test_solid_angle_in_the_panel_plane_but_outside_it_vanishes():
    pts = np.array([[2.0, 2.0, 0.0], [-1.0, 0.5, 0.0]])
    assert np.abs(solid_angle(_UNIT_TRIANGLE, pts)).max() < 1e-14


@pytest.mark.parametrize("offset", [4.0, 1.0, 0.4, -0.4, -1.0])
def test_analytic_velocity_matches_quadrature_over_the_panel(offset):
    """Fixes every sign and the branch of the logarithm against a direct panel integral."""
    p = np.array([[0.3, 0.25, offset]])
    analytic = source_velocity(_UNIT_TRIANGLE, p)[0, 0]
    quad = velocity_by_quadrature(_UNIT_TRIANGLE, p, order=48)[0, 0]
    assert analytic == pytest.approx(quad, rel=1e-9)


def test_analytic_velocity_matches_quadrature_off_axis():
    rng = np.random.default_rng(3)
    pts = rng.uniform(-2.0, 2.0, (12, 3))
    analytic = source_velocity(_UNIT_TRIANGLE, pts)[:, 0]
    quad = velocity_by_quadrature(_UNIT_TRIANGLE, pts, order=64)[:, 0]
    assert np.abs(analytic - quad).max() / np.abs(quad).max() < 1e-8


def test_far_field_of_a_panel_is_a_point_source():
    """A unit-density panel of area A at distance r induces A/(4 pi r^2), directed away."""
    p = np.array([[1.0 / 3.0, 1.0 / 3.0, 60.0]])
    v = source_velocity(_UNIT_TRIANGLE, p)[0, 0]
    assert v[2] == pytest.approx(0.5 / (4 * np.pi * 60.0 ** 2), rel=1e-3)
    assert v[2] > 0.0


def test_influence_matrix_diagonal_is_the_jump_term():
    a = rankine_normal_velocity_matrix(icosphere(1.0, subdivisions=1))
    assert np.diag(a) == pytest.approx(0.5, abs=1e-15)


def test_influence_matrix_is_well_conditioned():
    for sub in (1, 2, 3):
        a = rankine_normal_velocity_matrix(icosphere(1.0, subdivisions=sub))
        assert np.linalg.cond(a) < 5.0


def test_triangle_rule_integrates_polynomials_exactly():
    bary, w = _triangle_rule(8)
    # The weights integrate over the reference triangle, whose area is 1/2; that is why
    # callers scale by 2 * (actual triangle area).
    assert w.sum() == pytest.approx(0.5, rel=1e-14)
    s, t = bary[:, 0], bary[:, 1]
    assert (w * s).sum() == pytest.approx(1.0 / 6.0, rel=1e-13)
    assert (w * s * t).sum() == pytest.approx(1.0 / 24.0, rel=1e-13)


def test_sphere_source_density_converges_to_the_analytic_three_halves():
    """A surface density sigma_1 cos(theta) on r = a gives the exterior potential
    -sigma_1 a^3 cos(theta)/(3 r^2).  Matching the sphere-in-a-stream dipole
    -u a^3 cos(theta)/(2 r^2) requires sigma_1 = 3u/2, and n_x = cos(theta), so the exact
    density is sigma = 1.5 u n_x.

    Piecewise-constant collocation on a curved body is first order in the density, which is
    the expected rate: the best piecewise-constant approximation to a smooth density is
    itself only O(h).  So the test asserts the observed first-order rate and checks that
    Richardson extrapolation recovers 3/2, rather than demanding a tight absolute error at
    one panel count.
    """
    u = 1.0
    means = []
    for sub in (1, 2, 3):
        mesh = icosphere(1.0, subdivisions=sub)
        sigma = solve_rankine(mesh, speed=u)
        nx = mesh.unit_normals()[:, 0]
        keep = np.abs(nx) > 0.2
        means.append(float(np.mean(sigma[keep] / (u * nx[keep]))))
    errors = [abs(m - 1.5) for m in means]
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(1.7 < r < 2.4 for r in ratios), f"expected first order, got {ratios} from {means}"
    extrapolated = 2 * means[-1] - means[-2]
    assert extrapolated == pytest.approx(1.5, abs=0.005), f"extrapolant {extrapolated}"


def test_sphere_exterior_potential_from_the_exact_density_is_second_order():
    """Confirms the geometry and the potential evaluation independently of the solve."""
    u = 1.0
    rng = np.random.default_rng(0)
    d = rng.normal(size=(80, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    pts = d * rng.uniform(2.5, 5.0, (80, 1))
    exact = -u * pts[:, 0] / (2 * np.linalg.norm(pts, axis=1) ** 3)

    def potential(mesh, sigma, order=12):
        tri, area = mesh.triangles(), mesh.areas()
        bary, w = _triangle_rule(order)
        e0, e1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        q = (tri[:, 0:1, :] + bary[None, :, 0, None] * e0[:, None, :]
             + bary[None, :, 1, None] * e1[:, None, :])
        r = np.linalg.norm(pts[:, None, None, :] - q[None], axis=-1)
        return -(1 / (4 * np.pi)) * np.einsum("q,m,nmq,m->n", w * 2.0, area, 1.0 / r, sigma)

    errors = []
    for sub in (1, 2, 3):
        mesh = icosphere(1.0, subdivisions=sub)
        sigma = 1.5 * u * mesh.unit_normals()[:, 0]
        errors.append(float(np.abs(potential(mesh, sigma) - exact).max() / np.abs(exact).max()))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(r > 3.0 for r in ratios), f"expected second order, got {ratios} from {errors}"


def test_uv_sphere_panelling_is_worse_than_the_icosphere():
    """Recorded because it explains why the test body is an icosphere: polar slivers spoil
    the rate for reasons unrelated to the method."""
    u = 1.0

    def scatter(mesh):
        sigma = solve_rankine(mesh, speed=u)
        nx = mesh.unit_normals()[:, 0]
        keep = np.abs(nx) > 0.2
        return float(np.std(sigma[keep] / (u * nx[keep])))

    ico = icosphere(1.0, subdivisions=3)           # 1280 panels
    uv = uv_sphere(1.0, n_theta=20, n_phi=40)      # 1520 panels
    assert scatter(uv) > 3.0 * scatter(ico)
